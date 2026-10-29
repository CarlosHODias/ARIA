//! Sistema de plugins para estender as capacidades da ARIA.
//!
//! Um plugin é uma struct que implementa o trait `AriaPlugin`.
//! O `PluginRegistry` descobre e despacha comandos para plugins registrados.
//!
//! # Adicionando um plugin
//! 1. Crie um arquivo em `src/plugins/meu_plugin.rs`
//! 2. Implemente `AriaPlugin`
//! 3. Registre em `PluginRegistry::default()`

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use tracing::{debug, info};

pub mod weather;
pub mod calculator;
pub mod reminder;

pub use weather::WeatherPlugin;
pub use calculator::CalculatorPlugin;
pub use reminder::ReminderPlugin;

// ─── Trait base ───────────────────────────────────────────────────────────────

#[async_trait]
pub trait AriaPlugin: Send + Sync {
    /// Nome único do plugin (ex: "weather")
    fn name(&self) -> &str;

    /// Descrição curta (exibida no /help)
    fn description(&self) -> &str;

    /// Prefixos de comando que ativam este plugin (ex: ["/clima", "/weather"])
    fn triggers(&self) -> Vec<&str>;

    /// Executa o plugin. `args` é tudo depois do trigger.
    async fn execute(&self, args: &str) -> Result<String>;

    /// Retorna true se este plugin deve interceptar mensagens de linguagem
    /// natural (não apenas comandos com prefixo)
    fn handles_natural_language(&self) -> bool { false }

    /// Chamado para mensagens sem prefixo — só se `handles_natural_language` = true
    async fn handle_natural(&self, _input: &str) -> Result<Option<String>> {
        Ok(None)
    }
}

// ─── Registry ────────────────────────────────────────────────────────────────

pub struct PluginRegistry {
    plugins: Vec<Box<dyn AriaPlugin>>,
    /// Mapa de trigger → índice no Vec para lookup O(1)
    trigger_map: HashMap<String, usize>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self { plugins: Vec::new(), trigger_map: HashMap::new() }
    }

    /// Registra todos os plugins built-in
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(WeatherPlugin::new()));
        reg.register(Box::new(CalculatorPlugin::new()));
        reg.register(Box::new(ReminderPlugin::new()));
        reg
    }

    pub fn register(&mut self, plugin: Box<dyn AriaPlugin>) {
        let idx = self.plugins.len();
        info!("Plugin registrado: {} — {:?}", plugin.name(), plugin.triggers());
        for trigger in plugin.triggers() {
            self.trigger_map.insert(trigger.to_lowercase(), idx);
        }
        self.plugins.push(plugin);
    }

    /// Despacha um input para o plugin correto.
    /// Retorna None se nenhum plugin interceptou.
    pub async fn dispatch(&self, input: &str) -> Result<Option<String>> {
        let lower = input.to_lowercase();

        // 1. Verificar prefixos de trigger
        for (trigger, &idx) in &self.trigger_map {
            if lower.starts_with(trigger.as_str()) {
                let args = input[trigger.len()..].trim();
                debug!("Plugin '{}' ativado por trigger '{trigger}'", self.plugins[idx].name());
                return Ok(Some(self.plugins[idx].execute(args).await?));
            }
        }

        // 2. Processar linguagem natural em todos os plugins que suportam
        for plugin in &self.plugins {
            if plugin.handles_natural_language() {
                if let Some(resp) = plugin.handle_natural(input).await? {
                    debug!("Plugin '{}' interceptou linguagem natural", plugin.name());
                    return Ok(Some(resp));
                }
            }
        }

        Ok(None)
    }

    /// Lista todos os plugins registrados (para /help)
    pub fn list(&self) -> Vec<(&str, &str, Vec<&str>)> {
        self.plugins.iter().map(|p| (p.name(), p.description(), p.triggers())).collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self { Self::with_defaults() }
}