//! Plugin de calculadora — avalia expressões matemáticas simples sem dependências externas.

use anyhow::Result;
use async_trait::async_trait;
use super::AriaPlugin;

pub struct CalculatorPlugin;

impl CalculatorPlugin {
    pub fn new() -> Self { Self }

    fn eval(&self, expr: &str) -> Result<f64> {
        // Parser recursivo simples: suporta +, -, *, /, ^, parênteses
        let expr = expr.replace(',', ".").replace(" ", "");
        parse_expr(&expr, &mut 0)
    }
}

#[async_trait]
impl AriaPlugin for CalculatorPlugin {
    fn name(&self)        -> &str { "calculator" }
    fn description(&self) -> &str { "Avalia expressões matemáticas: /calc 2^10 + sqrt(16)" }
    fn triggers(&self)    -> Vec<&str> { vec!["/calc", "/calcular", "/math"] }

    async fn execute(&self, args: &str) -> Result<String> {
        if args.is_empty() {
            return Ok("Uso: /calc <expressão>  ex: /calc 2^10 + 3*4".into());
        }
        match self.eval(args) {
            Ok(result) => Ok(format!("{args} = {result}")),
            Err(e)     => Ok(format!("Erro ao avaliar '{args}': {e}")),
        }
    }

    fn handles_natural_language(&self) -> bool { true }

    async fn handle_natural(&self, input: &str) -> Result<Option<String>> {
        // Interceptar perguntas simples tipo "quanto é 2 + 2?"
        let lower = input.to_lowercase();
        if lower.starts_with("quanto é") || lower.starts_with("calcule") || lower.starts_with("quanto vale") {
            let expr = input
                .splitn(2, ' ')
                .nth(1)
                .unwrap_or("")
                .replace("?", "")
                .trim()
                .to_string();
            if !expr.is_empty() {
                return Ok(Some(self.execute(&expr).await?));
            }
        }
        Ok(None)
    }
}

// ─── Parser aritmético recursivo ─────────────────────────────────────────────

fn parse_expr(s: &str, pos: &mut usize) -> Result<f64> {
    let mut left = parse_term(s, pos)?;
    while *pos < s.len() {
        match s.as_bytes().get(*pos) {
            Some(b'+') => { *pos += 1; left += parse_term(s, pos)?; }
            Some(b'-') => { *pos += 1; left -= parse_term(s, pos)?; }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_term(s: &str, pos: &mut usize) -> Result<f64> {
    let mut left = parse_power(s, pos)?;
    while *pos < s.len() {
        match s.as_bytes().get(*pos) {
            Some(b'*') => { *pos += 1; left *= parse_power(s, pos)?; }
            Some(b'/') => {
                *pos += 1;
                let r = parse_power(s, pos)?;
                if r == 0.0 { return Err(anyhow::anyhow!("Divisão por zero")); }
                left /= r;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_power(s: &str, pos: &mut usize) -> Result<f64> {
    let base = parse_unary(s, pos)?;
    if s.as_bytes().get(*pos) == Some(&b'^') {
        *pos += 1;
        Ok(base.powf(parse_unary(s, pos)?))
    } else {
        Ok(base)
    }
}

fn parse_unary(s: &str, pos: &mut usize) -> Result<f64> {
    if s.as_bytes().get(*pos) == Some(&b'-') {
        *pos += 1;
        return Ok(-parse_atom(s, pos)?);
    }
    parse_atom(s, pos)
}

fn parse_atom(s: &str, pos: &mut usize) -> Result<f64> {
    if *pos >= s.len() {
        return Err(anyhow::anyhow!("Expressão incompleta"));
    }
    // Parênteses
    if s.as_bytes()[*pos] == b'(' {
        *pos += 1;
        let val = parse_expr(s, pos)?;
        if s.as_bytes().get(*pos) == Some(&b')') { *pos += 1; }
        return Ok(val);
    }
    // Funções (sqrt, sin, cos, etc.)
    for (name, func) in &[
        ("sqrt", f64::sqrt as fn(f64) -> f64),
        ("sin",  f64::sin),
        ("cos",  f64::cos),
        ("tan",  f64::tan),
        ("log",  f64::ln),
        ("abs",  f64::abs),
    ] {
        if s[*pos..].starts_with(name) {
            *pos += name.len();
            if s.as_bytes().get(*pos) == Some(&b'(') {
                *pos += 1;
                let arg = parse_expr(s, pos)?;
                if s.as_bytes().get(*pos) == Some(&b')') { *pos += 1; }
                return Ok(func(arg));
            }
        }
    }
    // Constantes
    if s[*pos..].starts_with("pi") { *pos += 2; return Ok(std::f64::consts::PI); }
    if s[*pos..].starts_with('e')  { *pos += 1; return Ok(std::f64::consts::E); }
    // Número
    let start = *pos;
    while *pos < s.len() && (s.as_bytes()[*pos].is_ascii_digit() || s.as_bytes()[*pos] == b'.') {
        *pos += 1;
    }
    if start == *pos {
        return Err(anyhow::anyhow!("Caractere inesperado: '{}'", s.chars().nth(start).unwrap_or('?')));
    }
    Ok(s[start..*pos].parse::<f64>()?)
}