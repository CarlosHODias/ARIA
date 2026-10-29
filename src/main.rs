use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use tracing::info;

use aria_lib::{config::AriaConfig, orchestrator::Orchestrator, tui};

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "aria",
    about   = "ARIA — Autonomous Rust Intelligence Assistant",
    version = env!("CARGO_PKG_VERSION"),
    author  = "github.com/you/aria-assistant"
)]
struct Cli {
    #[arg(short, long, default_value = "~/.config/aria/config.toml")]
    config: String,

    #[arg(short, long, default_value = "info",
        help = "Nível de log: trace | debug | info | warn | error")]
    log_level: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Inicia ARIA em modo interativo CLI (padrão)
    Run,
    /// Inicia ARIA em modo TUI (interface visual no terminal)
    Tui,
    /// Envia uma mensagem de texto e imprime a resposta
    Ask {
        #[arg(help = "Mensagem para a ARIA")]
        message: String,
    },
    /// Captura a tela e descreve o conteúdo via Phi-3 Vision
    See,
    /// Executa um comando Linux e retorna o output
    Exec {
        #[arg(help = "Comando shell a executar")]
        command: String,
    },
    /// Exibe informações do sistema (CPU, RAM, processos)
    Status,
    /// Lista os modelos disponíveis no Ollama local
    Models,
    /// Exibe o status da memória (SQLite + Qdrant)
    Memory,
    /// Mostra todos os plugins registrados
    Plugins,
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Logging colorido
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log_level.parse().unwrap_or_else(|_| "info".parse().unwrap()))
        )
        .with_target(false)
        .compact()
        .init();

    print_banner();

    let config = AriaConfig::load(&cli.config).await?;
    info!("Configuração: {}", cli.config);

    match cli.command.unwrap_or(Commands::Run) {

        Commands::Run => {
            let mut orc = Orchestrator::new(config).await?;
            orc.run_interactive().await?;
        }

        Commands::Tui => {
            // Iniciar orquestrador e expor via closure para a TUI
            let mut orc = Orchestrator::new(config).await?;
            tui::run_tui(|input| async move {
                orc.process_text(&input).await
            }).await?;
        }

        Commands::Ask { message } => {
            let mut orc = Orchestrator::new(config).await?;
            let resp = orc.process_text(&message).await?;
            println!("{} {}", "ARIA:".cyan().bold(), resp);
        }

        Commands::See => {
            let mut orc = Orchestrator::new(config).await?;
            println!("{}", "Capturando tela...".dimmed());
            let desc = orc.capture_and_describe_screen().await?;
            println!("{} {}", "ARIA:".cyan().bold(), desc);
        }

        Commands::Exec { command } => {
            let orc = Orchestrator::new(config).await?;
            let out = orc.execute_system_command(&command).await?;
            println!("{}", out);
        }

        Commands::Status => {
            let orc = Orchestrator::new(config).await?;
            let s = orc.system_status().await?;
            println!("{s}");
        }

        Commands::Models => {
            use aria_lib::inference::InferenceEngine;
            let engine = InferenceEngine::new(&config.inference).await?;
            let models = engine.list_models().await?;
            println!("\n{}", "Modelos disponíveis no Ollama:".cyan().bold());
            for m in &models {
                println!("  • {m}");
            }
            if models.is_empty() {
                println!("  Nenhum modelo encontrado. Baixe com: ollama pull phi3:14b-vision");
            }
        }

        Commands::Memory => {
            use aria_lib::memory::MemoryStore;
            let store = MemoryStore::new(&config.memory).await?;
            let count = store.episode_count().await?;
            println!("\n{}", "Status da Memória ARIA:".cyan().bold());
            println!("  Episódios armazenados : {count}");
            println!("  Qdrant               : {}",
                     if store.qdrant_online() { "online".green() } else { "offline".red() });
            println!("  SQLite               : {}", shellexpand::tilde(&config.memory.sqlite_path));

            if count > 0 {
                println!("\n{}", "Últimos 5 episódios:".dimmed());
                let recent = store.recent_episodes(5).await?;
                for ep in &recent {
                    println!("  [{}] {}", ep.created_at.dimmed(), &ep.input[..ep.input.len().min(80)]);
                }
            }
        }

        Commands::Plugins => {
            use aria_lib::PluginRegistry;
            let reg = PluginRegistry::default();
            println!("\n{}", "Plugins registrados:".cyan().bold());
            for (name, desc, triggers) in reg.list() {
                println!("  {} — {}", name.yellow().bold(), desc);
                println!("    Triggers: {}", triggers.join(", ").dimmed());
            }
        }
    }

    Ok(())
}

fn print_banner() {
    println!(
        "\n{}\n{}\n",
        r#"  █████╗ ██████╗ ██╗ █████╗ "#.cyan(),
        r#" ██╔══██╗██╔══██╗██║██╔══██╗"#.cyan(),
    );
    println!(
        "  {} v{}\n",
        "ARIA — Autonomous Rust Intelligence Assistant".cyan().bold(),
        env!("CARGO_PKG_VERSION").dimmed()
    );
}