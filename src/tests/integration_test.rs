//! Testes de integração da ARIA.
//! Execute com: cargo test -- --nocapture

use aria_lib::config::AriaConfig;

// ─── Config ───────────────────────────────────────────────────────────────────

#[test]
fn test_config_default_is_valid() {
    let cfg = AriaConfig::default();
    assert!(!cfg.inference.model.is_empty(), "model não pode ser vazio");
    assert!(cfg.inference.context_size > 0, "context_size deve ser > 0");
    assert!(cfg.inference.temperature >= 0.0 && cfg.inference.temperature <= 2.0);
    assert!(!cfg.memory.collection_name.is_empty());
    assert!(cfg.memory.top_k > 0);
    assert!(!cfg.voice.language.is_empty());
}

#[test]
fn test_config_serialization_roundtrip() {
    let original = AriaConfig::default();
    let toml_str = toml::to_string(&original).expect("Serialização falhou");
    let parsed: AriaConfig = toml::from_str(&toml_str).expect("Parsing falhou");
    assert_eq!(parsed.inference.model, original.inference.model);
    assert_eq!(parsed.memory.collection_name, original.memory.collection_name);
    assert_eq!(parsed.voice.language, original.voice.language);
}

#[test]
fn test_blocked_commands_present() {
    let cfg = AriaConfig::default();
    assert!(!cfg.system.blocked_commands.is_empty());
    assert!(cfg.system.blocked_commands.iter().any(|c| c.contains("rm -rf /")));
}

// ─── Calculadora ──────────────────────────────────────────────────────────────

mod calculator_tests {
    use aria_lib::plugins::{AriaPlugin, CalculatorPlugin};

    fn calc() -> CalculatorPlugin { CalculatorPlugin::new() }

    #[tokio::test]
    async fn test_basic_arithmetic() {
        let c = calc();
        let r = c.execute("2 + 2").await.unwrap();
        assert!(r.contains('4'), "2+2 deve ser 4, obtido: {r}");
    }

    #[tokio::test]
    async fn test_power() {
        let c = calc();
        let r = c.execute("2^10").await.unwrap();
        assert!(r.contains("1024"), "2^10=1024, obtido: {r}");
    }

    #[tokio::test]
    async fn test_order_of_operations() {
        let c = calc();
        let r = c.execute("2 + 3 * 4").await.unwrap();
        assert!(r.contains("14"), "2+3*4=14, obtido: {r}");
    }

    #[tokio::test]
    async fn test_parentheses() {
        let c = calc();
        let r = c.execute("(2 + 3) * 4").await.unwrap();
        assert!(r.contains("20"), "(2+3)*4=20, obtido: {r}");
    }

    #[tokio::test]
    async fn test_pi_constant() {
        let c = calc();
        let r = c.execute("pi * 2").await.unwrap();
        assert!(r.contains("6.28"), "pi*2≈6.28, obtido: {r}");
    }

    #[tokio::test]
    async fn test_sqrt() {
        let c = calc();
        let r = c.execute("sqrt(16)").await.unwrap();
        assert!(r.contains('4'), "sqrt(16)=4, obtido: {r}");
    }

    #[tokio::test]
    async fn test_division_by_zero() {
        let c = calc();
        let r = c.execute("10 / 0").await.unwrap();
        assert!(r.to_lowercase().contains("zero") || r.contains("erro"),
                "Divisão por zero deve retornar erro, obtido: {r}");
    }

    #[tokio::test]
    async fn test_empty_args() {
        let c = calc();
        let r = c.execute("").await.unwrap();
        assert!(r.contains("Uso:"), "Args vazio deve mostrar uso, obtido: {r}");
    }
}

// ─── Plugin registry ─────────────────────────────────────────────────────────

mod plugin_tests {
    use aria_lib::plugins::PluginRegistry;

    #[tokio::test]
    async fn test_registry_dispatch_calc() {
        let reg = PluginRegistry::with_defaults();
        let r = reg.dispatch("/calc 3 + 3").await.unwrap();
        assert!(r.is_some(), "Plugin calc deve ser ativado");
        let resp = r.unwrap();
        assert!(resp.contains('6'), "3+3=6, obtido: {resp}");
    }

    #[tokio::test]
    async fn test_registry_no_match() {
        let reg = PluginRegistry::with_defaults();
        let r = reg.dispatch("olá, tudo bem?").await.unwrap();
        // Pode ser Some (se linguagem natural capturar) ou None — só não deve paniq
        let _ = r;
    }

    #[tokio::test]
    async fn test_registry_reminder_trigger() {
        let reg = PluginRegistry::with_defaults();
        let r = reg.dispatch("/lembrar xpto").await.unwrap();
        assert!(r.is_some());
        let resp = r.unwrap();
        // Deve retornar erro de formato, não paniq
        assert!(!resp.is_empty(), "Deve retornar alguma resposta");
    }

    #[test]
    fn test_registry_list() {
        let reg = PluginRegistry::with_defaults();
        let list = reg.list();
        assert!(list.len() >= 3, "Deve ter pelo menos 3 plugins built-in");
        let names: Vec<&str> = list.iter().map(|(n, _, _)| *n).collect();
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"weather"));
        assert!(names.contains(&"reminder"));
    }
}

// ─── Sistema de segurança ─────────────────────────────────────────────────────

mod security_tests {
    use aria_lib::config::AriaConfig;

    #[test]
    fn test_blocked_commands_list_not_empty() {
        let cfg = AriaConfig::default();
        assert!(!cfg.system.blocked_commands.is_empty());
    }

    #[test]
    fn test_fork_bomb_is_blocked() {
        let cfg = AriaConfig::default();
        let fork_bomb = ":(){:|:&};:";
        let blocked = cfg.system.blocked_commands.iter()
            .any(|b| fork_bomb.contains(b.as_str()) || b.contains(fork_bomb));
        assert!(blocked, "Fork bomb deve estar na blocklist");
    }
}