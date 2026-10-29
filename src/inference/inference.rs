use anyhow::{anyhow, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::debug;

use crate::config::InferenceConfig;

// ─── Tipos públicos ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, User, Assistant }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role:    Role,
    pub content: String,
}

/// Token individual recebido via streaming
#[derive(Debug)]
pub enum StreamToken {
    Token(String),
    Done(String),  // resposta completa acumulada
    Error(String),
}

// ─── Payloads Ollama ──────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model:    &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream:   bool,
    options:  OllamaOptions,
}

#[derive(Serialize)]
struct OllamaMessage<'a> {
    role:    &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<&'a str>>,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_ctx:     usize,
    num_predict: usize,
}

#[derive(Deserialize)]
struct OllamaChunk {
    message: OllamaChunkMsg,
    #[serde(default)]
    done: bool,
}

#[derive(Deserialize)]
struct OllamaChunkMsg {
    content: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResp {
    embedding: Vec<f32>,
}

// ─── Engine ───────────────────────────────────────────────────────────────────

pub struct InferenceEngine {
    client:   reqwest::Client,
    config:   InferenceConfig,
    chat_url: String,
    embed_url: String,
}

impl InferenceEngine {
    pub async fn new(config: &InferenceConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;

        // Verificar se o Ollama está rodando
        let tags_url = format!("{}/api/tags", config.ollama_url);
        client.get(&tags_url).send().await.map_err(|_| {
            anyhow!(
                "Ollama não está rodando em {}.\n  Inicie com: ollama serve\n  \
                 Depois baixe o modelo: ollama pull {}",
                config.ollama_url, config.model
            )
        })?;

        debug!("Ollama OK em {}", config.ollama_url);

        Ok(Self {
            chat_url:  format!("{}/api/chat",       config.ollama_url),
            embed_url: format!("{}/api/embeddings", config.ollama_url),
            client,
            config: config.clone(),
        })
    }

    // ─── Chat síncrono (aguarda resposta completa) ────────────────────────────

    pub async fn chat(&self, system_prompt: &str, history: &[Message]) -> Result<String> {
        self.inner_chat(system_prompt, history, None, None, false).await
    }

    pub async fn chat_with_image(
        &self,
        system_prompt: &str,
        history:       &[Message],
        user_message:  &str,
        image_b64:     &str,
    ) -> Result<String> {
        self.inner_chat(system_prompt, history, Some(user_message), Some(image_b64), false).await
    }

    // ─── Chat com streaming token a token ─────────────────────────────────────

    //noinspection ALL
    //noinspection ALL
    /// Retorna um channel receptor — cada mensagem é um token.
    /// Útil para imprimir no terminal em tempo real.
    pub async fn chat_streaming(
        &self,
        system_prompt: &str,
        history:       &[Message],
    ) -> Result<mpsc::Receiver<StreamToken>> {
        let (tx, rx) = mpsc::channel::<StreamToken>(256);

        let payload = self.build_payload(system_prompt, history, None, None, true);
        let client  = self.client.clone();
        let url     = self.chat_url.clone();

        // Spawn task que lê o stream e envia tokens para o channel
        tokio::spawn(async move {
            let resp = match client.post(&url).json(&payload).send().await {
                Ok(r)  => r,
                Err(e) => {
                    let _ = tx.send(StreamToken::Error(e.to_string())).await;
                    return;
                }
            };

            let mut stream    = resp.bytes_stream();
            let mut full_text = String::new();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Err(e) => {
                        let _ = tx.send(StreamToken::Error(e.to_string())).await;
                        return;
                    }
                    Ok(bytes) => {
                        // Ollama envia JSON por linha (ndjson)
                        let text = String::from_utf8_lossy(&bytes);
                        for line in text.lines() {
                            if line.is_empty() { continue; }
                            match serde_json::from_str::<OllamaChunk>(line) {
                                Ok(chunk) => {
                                    full_text.push_str(&chunk.message.content);
                                    let _ = tx.send(StreamToken::Token(
                                        chunk.message.content
                                    )).await;
                                    if chunk.done {
                                        let _ = tx.send(StreamToken::Done(full_text.clone())).await;
                                        return;
                                    }
                                }
                                Err(_) => {} // linha incompleta, ignorar
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    // ─── Embeddings ───────────────────────────────────────────────────────────

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct Req<'a> { model: &'a str, prompt: &'a str }

        let resp = self.client
            .post(&self.embed_url)
            .json(&Req { model: "nomic-embed-text", prompt: text })
            .send()
            .await?
            .json::<OllamaEmbedResp>()
            .await?;

        Ok(resp.embedding)
    }

    // ─── Lista modelos disponíveis no Ollama ──────────────────────────────────

    pub async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct TagsResp { models: Vec<ModelInfo> }
        #[derive(Deserialize)]
        struct ModelInfo { name: String }

        let url  = format!("{}/api/tags", self.config.ollama_url);
        let resp = self.client.get(&url).send().await?.json::<TagsResp>().await?;
        Ok(resp.models.into_iter().map(|m| m.name).collect())
    }

    // ─── Internos ────────────────────────────────────────────────────────────

    //noinspection ALL
    async fn inner_chat(
        &self,
        system_prompt: &str,
        history:       &[Message],
        extra_msg:     Option<&str>,
        image_b64:     Option<&str>,
        stream:        bool,
    ) -> Result<String> {
        let payload = self.build_payload(system_prompt, history, extra_msg, image_b64, stream);

        let bytes = self.client
            .post(&self.chat_url)
            .json(&payload)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;

        // Sem streaming: resposta única ou ndjson com done=true
        // Tentamos parsear linha por linha e acumular
        let text = String::from_utf8_lossy(&bytes);
        let mut full = String::new();

        for line in text.lines() {
            if line.is_empty() { continue; }
            if let Ok(chunk) = serde_json::from_str::<OllamaChunk>(line) {
                full.push_str(&chunk.message.content);
                if chunk.done { break; }
            }
        }

        Ok(full.trim().to_string())
    }

    fn build_payload<'a>(
        &'a self,
        system_prompt: &'a str,
        history:       &'a [Message],
        extra_msg:     Option<&'a str>,
        image_b64:     Option<&'a str>,
        stream:        bool,
    ) -> serde_json::Value {
        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({ "role": "system", "content": system_prompt }),
        ];

        for msg in history {
            let role = match msg.role {
                Role::User      => "user",
                Role::Assistant => "assistant",
                Role::System    => "system",
            };
            messages.push(serde_json::json!({ "role": role, "content": msg.content }));
        }

        if let Some(text) = extra_msg {
            let mut m = serde_json::json!({ "role": "user", "content": text });
            if let Some(img) = image_b64 {
                m["images"] = serde_json::json!([img]);
            }
            messages.push(m);
        }

        serde_json::json!({
            "model":    self.config.model,
            "messages": messages,
            "stream":   stream,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx":     self.config.context_size,
                "num_predict": self.config.max_tokens,
            }
        })
    }
}