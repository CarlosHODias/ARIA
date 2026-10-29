use anyhow::Result;
use colored::Colorize;
use tracing::{debug, info, warn};
use std::io::{self, Write};

use crate::{
    config::AriaConfig,
    inference::{InferenceEngine, Message, Role},
    memory::MemoryStore,
    voice::VoiceRecognizer,
    vision::VisionCapture,
    tts::TtsEngine,
    system::SystemController,
};

// ─── Orquestrador principal ───────────────────────────────────────────────────

pub struct Orchestrator {
    config:    AriaConfig,
    inference: InferenceEngine,
    memory:    MemoryStore,
    voice:     VoiceRecognizer,
    vision:    VisionCapture,
    tts:       TtsEngine,
    system:    SystemController,
    history:   Vec<Message>,
}

impl Orchestrator {
    pub async fn new(config: AriaConfig) -> Result<Self> {
        info!("Inicializando subsistemas ARIA...");

        let inference = InferenceEngine::new(&config.inference).await?;
        info!("  {} Motor de inferência ({})", "✓".green(), config.inference.model);

        let memory = MemoryStore::new(&config.memory).await?;
        let mem_status = if memory.qdrant_online() {
            format!("Qdrant + SQLite")
        } else {
            format!("SQLite apenas (Qdrant offline)")
        };
        info!("  {} Memória: {}", "✓".green(), mem_status);

        let voice = VoiceRecognizer::new(&config.voice).await?;
        info!("  {} STT Whisper (lang: {})", "✓".green(), config.voice.language);

        let vision = VisionCapture::new(&config.vision).await?;
        info!("  {} Visão / captura de tela", "✓".green());

        let tts = TtsEngine::new(&config.tts).await?;
        info!("  {} TTS Piper", "✓".green());

        let system = SystemController::new(&config.system).await?;
        info!("  {} Controlador Linux", "✓".green());

        Ok(Self { config, inference, memory, voice, vision, tts, system, history: Vec::new() })
    }

    // ─── Modo interativo ──────────────────────────────────────────────────────

    pub async fn run_interactive(&mut self) -> Result<()> {
        let count = self.memory.episode_count().await.unwrap_or(0);
        println!(
            "\n{} — {} episódios em memória\n{}",
            "ARIA".cyan().bold(),
            count,
            "Digite /help para comandos ou escreva sua mensagem.".dimmed()
        );

        loop {
            print!("{} ", "▶".cyan());
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim().to_string();
            if input.is_empty() { continue; }

            let result = match input.as_str() {
                "/help"  => { self.print_help(); Ok(()) }
                "/clear" => { self.clear_history(); Ok(()) }
                "/mem"   => { self.show_memory_status().await }
                "/status"=> { let s = self.system_status().await?; println!("{s}"); Ok(()) }
                "/listen"=> { self.listen_and_respond().await }
                "/see"   => {
                    println!("{}", "Capturando tela...".dimmed());
                    let d = self.capture_and_describe_screen().await?;
                    println!("\n{} {}", "ARIA:".cyan().bold(), d);
                    Ok(())
                }
                "/quit" | "/exit" | "/q" => {
                    println!("{}", "Até logo!".cyan());
                    break;
                }
                _ if input.starts_with("/exec ") => {
                    let cmd = input[6..].trim();
                    println!("{}", format!("$ {cmd}").dimmed());
                    let out = self.execute_system_command(cmd).await?;
                    println!("{out}");
                    Ok(())
                }
                _ if input.starts_with("/search ") => {
                    let q = &input[8..];
                    self.show_memory_search(q).await
                }
                _ => self.handle_message(&input).await,
            };

            if let Err(e) = result {
                eprintln!("{} {e}", "ERRO:".red().bold());
            }
        }

        Ok(())
    }

    // ─── Pipeline principal de mensagem ──────────────────────────────────────

    async fn handle_message(&mut self, input: &str) -> Result<()> {
        let response = self.process_text(input).await?;
        println!("\n{} {}\n", "ARIA:".cyan().bold(), response);

        if let Err(e) = self.tts.speak(&response).await {
            debug!("TTS (não crítico): {e}");
        }
        Ok(())
    }

    pub async fn process_text(&mut self, input: &str) -> Result<String> {
        // 1. Gerar embedding da query para busca semântica
        let query_embedding = self.inference.embed(input).await.unwrap_or_default();

        // 2. Busca semântica no Qdrant (se online)
        let semantic_hits = if !query_embedding.is_empty() {
            self.memory
                .search_semantic(query_embedding.clone(), self.config.memory.top_k)
                .await
                .unwrap_or_default()
        } else {
            vec![]
        };

        // 3. Fallback: FTS no SQLite se Qdrant offline
        let fts_hits = if semantic_hits.is_empty() {
            self.memory
                .search_episodes_fts(input, self.config.memory.top_k)
                .await
                .unwrap_or_default()
                .into_iter()
                .map(|e| format!("Q: {}\nA: {}", e.input, e.response))
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        // 4. Montar contexto enriquecido
        let context = self.build_context(input, &semantic_hits, &fts_hits);
        debug!("Contexto: {} chars, {} memórias semânticas, {} FTS",
            context.len(), semantic_hits.len(), fts_hits.len());

        // 5. Adicionar ao histórico e inferir
        self.history.push(Message { role: Role::User, content: context });

        let response = self.inference.chat(
            &self.config.inference.system_prompt,
            &self.history,
        ).await?;

        // 6. Persistir
        self.history.push(Message { role: Role::Assistant, content: response.clone() });
        self.memory.save_episode(input, &response).await?;

        // Salvar embedding da pergunta na memória semântica (async, não bloqueia)
        if !query_embedding.is_empty() {
            let _ = self.memory.save_semantic(input, query_embedding).await;
        }

        // 7. Janela deslizante de contexto (20 turnos = 40 mensagens)
        if self.history.len() > 40 {
            self.history.drain(0..2);
        }

        Ok(response)
    }

    // ─── Pipeline de voz ─────────────────────────────────────────────────────

    pub async fn listen_and_respond(&mut self) -> Result<()> {
        println!("{}", "🎤 Ouvindo... (fale agora)".yellow());
        let transcript = self.voice.listen().await?;
        println!("{} {}", "Você:".yellow().bold(), transcript);

        let response = self.process_text(&transcript).await?;
        println!("{} {}", "ARIA:".cyan().bold(), response);
        self.tts.speak(&response).await?;
        Ok(())
    }

    // ─── Visão ───────────────────────────────────────────────────────────────

    pub async fn capture_and_describe_screen(&mut self) -> Result<String> {
        let b64    = self.vision.capture_base64().await?;
        let prompt = "Descreva o conteúdo desta tela detalhadamente. \
                      Mencione janelas abertas, texto visível e o que o usuário está fazendo.";

        self.inference.chat_with_image(
            &self.config.inference.system_prompt,
            &self.history,
            prompt,
            &b64,
        ).await
    }

    // ─── Sistema ──────────────────────────────────────────────────────────────

    pub async fn execute_system_command(&self, command: &str) -> Result<String> {
        self.system.execute(command).await
    }

    pub async fn system_status(&self) -> Result<String> {
        self.system.status_report().await
    }

    // ─── Memória — utilitários interativos ────────────────────────────────────

    async fn show_memory_status(&self) -> Result<()> {
        let count  = self.memory.episode_count().await?;
        let qdrant = if self.memory.qdrant_online() {
            "online".green().to_string()
        } else {
            "offline".red().to_string()
        };
        println!(
            "\n{}\n  Episódios SQLite : {}\n  Qdrant           : {}\n  Histórico atual  : {} msgs\n",
            "Memória ARIA".cyan().bold(),
            count,
            qdrant,
            self.history.len()
        );
        Ok(())
    }

    async fn show_memory_search(&self, query: &str) -> Result<()> {
        println!("{}", format!("Buscando: «{query}»...").dimmed());

        let fts = self.memory.search_episodes_fts(query, 5).await?;
        if fts.is_empty() {
            println!("Nenhum episódio encontrado.");
        } else {
            for (i, ep) in fts.iter().enumerate() {
                println!("\n{} {}", format!("[{}]", i + 1).cyan(), ep.created_at.dimmed());
                println!("  {} {}", "Q:".yellow(), ep.input);
                println!("  {} {}", "A:".green(),  &ep.response[..ep.response.len().min(200)]);
            }
        }
        Ok(())
    }

    // ─── Helpers internos ─────────────────────────────────────────────────────

    fn build_context(
        &self,
        input:         &str,
        semantic_hits: &[crate::memory::SemanticHit],
        fts_hits:      &[String],
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        if !semantic_hits.is_empty() {
            let mems = semantic_hits
                .iter()
                .map(|h| format!("[score {:.2}] {}", h.score, h.text))
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(format!("[Memórias relevantes]\n{mems}"));
        } else if !fts_hits.is_empty() {
            parts.push(format!("[Episódios relacionados]\n{}", fts_hits.join("\n---\n")));
        }

        parts.push(input.to_string());
        parts.join("\n\n")
    }

    fn clear_history(&mut self) {
        self.history.clear();
        println!("{}", "Histórico da conversa limpo.".dimmed());
    }

    fn print_help(&self) {
        println!(
            r#"
{}
  {}   — enviar mensagem de texto
  {}  — capturar voz e responder
  {}    — capturar e descrever a tela
  {}  — executar comando Linux
  {}  — buscar na memória episódica
  {}   — status do sistema
  {}     — status da memória
  {}   — limpar histórico da conversa
  {}    — encerrar
"#,
            "Comandos ARIA".cyan().bold(),
            "<mensagem>".white(),
            "/listen".yellow(),
            "/see".yellow(),
            "/exec <cmd>".yellow(),
            "/search <texto>".yellow(),
            "/status".yellow(),
            "/mem".yellow(),
            "/clear".yellow(),
            "/quit".yellow(),
        );
    }
}