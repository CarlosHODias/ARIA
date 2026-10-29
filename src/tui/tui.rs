//! Interface TUI com ratatui — modo visual alternativo ao CLI simples.
//! Ativado com: aria run --tui

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;

// ─── Estado da TUI ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MessageKind { User, Aria, System, Error }

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub kind:    MessageKind,
    pub content: String,
}

pub struct TuiApp {
    pub messages:    Vec<ChatMessage>,
    pub input_buf:   String,
    pub scroll:      u16,
    pub cursor_pos:  usize,
    pub status_line: String,
}

impl TuiApp {
    pub fn new() -> Self {
        Self {
            messages: vec![ChatMessage {
                kind:    MessageKind::System,
                content: "ARIA iniciada. Digite sua mensagem e pressione Enter.".into(),
            }],
            input_buf:   String::new(),
            scroll:      0,
            cursor_pos:  0,
            status_line: "Pronta".into(),
        }
    }

    pub fn push_user(&mut self, msg: &str) {
        self.messages.push(ChatMessage {
            kind:    MessageKind::User,
            content: msg.to_string(),
        });
        self.scroll_to_bottom();
    }

    pub fn push_aria(&mut self, msg: &str) {
        self.messages.push(ChatMessage {
            kind:    MessageKind::Aria,
            content: msg.to_string(),
        });
        self.scroll_to_bottom();
    }

    pub fn push_system(&mut self, msg: &str) {
        self.messages.push(ChatMessage {
            kind:    MessageKind::System,
            content: msg.to_string(),
        });
    }

    pub fn push_error(&mut self, msg: &str) {
        self.messages.push(ChatMessage {
            kind:    MessageKind::Error,
            content: msg.to_string(),
        });
    }

    pub fn set_status(&mut self, s: &str) {
        self.status_line = s.to_string();
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll = self.messages.len().saturating_sub(1) as u16;
    }

    pub fn take_input(&mut self) -> String {
        let out = self.input_buf.clone();
        self.input_buf.clear();
        self.cursor_pos = 0;
        out
    }

    pub fn handle_key_input(&mut self, key: KeyCode) -> TuiAction {
        match key {
            KeyCode::Enter => {
                if !self.input_buf.trim().is_empty() {
                    return TuiAction::Submit;
                }
            }
            KeyCode::Char(c) => {
                self.input_buf.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.input_buf.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input_buf.len() {
                    self.input_buf.remove(self.cursor_pos);
                }
            }
            KeyCode::Left  => { if self.cursor_pos > 0 { self.cursor_pos -= 1; } }
            KeyCode::Right => { if self.cursor_pos < self.input_buf.len() { self.cursor_pos += 1; } }
            KeyCode::Up    => { self.scroll = self.scroll.saturating_sub(1); }
            KeyCode::Down  => {
                let max = self.messages.len().saturating_sub(1) as u16;
                self.scroll = (self.scroll + 1).min(max);
            }
            KeyCode::PageUp   => { self.scroll = self.scroll.saturating_sub(10); }
            KeyCode::PageDown => {
                let max = self.messages.len().saturating_sub(1) as u16;
                self.scroll = (self.scroll + 10).min(max);
            }
            KeyCode::Esc => return TuiAction::Quit,
            _ => {}
        }
        TuiAction::None
    }
}

#[derive(Debug, PartialEq)]
pub enum TuiAction { None, Submit, Quit }

// ─── Renderização ─────────────────────────────────────────────────────────────

pub fn render(f: &mut Frame, app: &TuiApp) {
    let size = f.size();

    // Layout: [header 3] [chat restante] [status 1] [input 3]
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(4),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(size);

    // ── Header ────────────────────────────────────────────────────────────────
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" ARIA ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("— Autonomous Rust Intelligence Assistant ", Style::default().fg(Color::DarkGray)),
    ]))
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)))
        .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);

    // ── Chat ──────────────────────────────────────────────────────────────────
    let visible_start = app.scroll as usize;
    let items: Vec<ListItem> = app.messages
        .iter()
        .skip(visible_start)
        .map(|m| {
            let (prefix, style) = match m.kind {
                MessageKind::User   => ("Você   ", Style::default().fg(Color::Yellow)),
                MessageKind::Aria   => ("ARIA   ", Style::default().fg(Color::Cyan)),
                MessageKind::System => ("Sistema", Style::default().fg(Color::DarkGray)),
                MessageKind::Error  => ("ERRO   ", Style::default().fg(Color::Red)),
            };
            let line = Line::from(vec![
                Span::styled(format!("[{prefix}] "), style.add_modifier(Modifier::BOLD)),
                Span::styled(&m.content, Style::default()),
            ]);
            ListItem::new(line)
        })
        .collect();

    let chat = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Conversa ")
            .border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(chat, chunks[1]);

    // ── Status ────────────────────────────────────────────────────────────────
    let status = Paragraph::new(Span::styled(
        format!(" {} ", app.status_line),
        Style::default().fg(Color::DarkGray),
    ));
    f.render_widget(status, chunks[2]);

    // ── Input ─────────────────────────────────────────────────────────────────
    let input_display = format!("{}_", app.input_buf);
    let input = Paragraph::new(input_display.as_str())
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Mensagem (Enter=enviar, Esc=sair, ↑↓=scroll) ")
            .border_style(Style::default().fg(Color::Yellow)))
        .wrap(Wrap { trim: false });
    f.render_widget(input, chunks[3]);
}

// ─── Runner principal da TUI ──────────────────────────────────────────────────

/// Inicia o terminal TUI e retorna um canal bidirecional com o orquestrador.
/// O caller deve processar TuiEvent::Input e devolver TuiResponse.
pub async fn run_tui<F, Fut>(mut handler: F) -> Result<()>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend  = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend)?;

    let mut app = TuiApp::new();

    loop {
        term.draw(|f| render(f, &app))?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Ctrl+C / Ctrl+Q para sair
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('q'))
                {
                    break;
                }

                match app.handle_key_input(key.code) {
                    TuiAction::Quit   => break,
                    TuiAction::Submit => {
                        let input = app.take_input();
                        app.push_user(&input);
                        app.set_status("Pensando...");
                        term.draw(|f| render(f, &app))?;

                        match handler(input).await {
                            Ok(resp)  => { app.push_aria(&resp); app.set_status("Pronta"); }
                            Err(e)    => { app.push_error(&e.to_string()); app.set_status("Erro"); }
                        }
                    }
                    TuiAction::None => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    term.show_cursor()?;
    Ok(())
}