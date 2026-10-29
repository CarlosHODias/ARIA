//! Detecção de hot-word ("Ei, ARIA") via análise de energia + Whisper leve.
//! Roda em loop contínuo numa thread dedicada e sinaliza o orquestrador.

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Evento emitido quando a hot-word é detectada
#[derive(Debug)]
pub struct HotWordEvent {
    pub timestamp: std::time::SystemTime,
}

pub struct HotWordDetector {
    /// Palavras-chave a detectar (lowercase)
    keywords:     Vec<String>,
    /// Limiar de energia RMS para considerar que há fala
    energy_threshold: f32,
    /// Pausa mínima entre detecções (evita ativações repetidas)
    cooldown:     Duration,
}

impl HotWordDetector {
    pub fn new(keywords: Vec<String>) -> Self {
        Self {
            keywords,
            energy_threshold: 0.015, // calibrar conforme o microfone
            cooldown:         Duration::from_secs(2),
        }
    }

    /// Inicia o loop de detecção em background.
    /// Retorna um Receiver que emite HotWordEvent quando ativado.
    pub fn start(self, whisper_ctx: Arc<whisper_rs::WhisperContext>) -> mpsc::Receiver<HotWordEvent> {
        let (tx, rx) = mpsc::channel::<HotWordEvent>(4);

        std::thread::spawn(move || {
            if let Err(e) = self.run_loop(whisper_ctx, tx) {
                tracing::error!("HotWord loop encerrou com erro: {e}");
            }
        });

        rx
    }

    fn run_loop(
        &self,
        whisper_ctx: Arc<whisper_rs::WhisperContext>,
        tx: mpsc::Sender<HotWordEvent>,
    ) -> Result<()> {
        let host   = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("Sem dispositivo de entrada"))?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;

        info!("HotWord detector iniciado — ouvindo por: {:?}", self.keywords);

        let buffer: Arc<std::sync::Mutex<Vec<f32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let buf_clone = Arc::clone(&buffer);
        let running   = Arc::new(AtomicBool::new(true));

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                let mut buf = buf_clone.lock().unwrap();
                buf.extend_from_slice(data);
                // Manter janela de 2s máximo para não crescer indefinidamente
                let max = (sample_rate as usize) * 2;
                if buf.len() > max {
                    buf.drain(0..buf.len() - max);
                }
            },
            |e| tracing::error!("Erro stream hotword: {e}"),
            None,
        )?;
        stream.play()?;

        let mut last_trigger = Instant::now() - self.cooldown; // permite trigger imediato
        let window_size      = sample_rate as usize; // 1s de áudio

        loop {
            std::thread::sleep(Duration::from_millis(300));

            if !running.load(Ordering::Relaxed) { break; }

            // Pegar snapshot do buffer
            let samples: Vec<f32> = {
                let buf = buffer.lock().unwrap();
                if buf.len() < window_size { continue; }
                buf[buf.len() - window_size..].to_vec()
            };

            // 1. Filtro de energia (evita rodar Whisper em silêncio)
            let rms = rms_energy(&samples);
            if rms < self.energy_threshold {
                debug!("HotWord: silêncio (RMS={rms:.4})");
                continue;
            }

            // 2. Cooldown
            if last_trigger.elapsed() < self.cooldown { continue; }

            // 3. Transcrição rápida com Whisper
            let transcript = quick_transcribe(&whisper_ctx, &samples)?;
            let lower = transcript.to_lowercase();
            debug!("HotWord transcrição: '{lower}'");

            // 4. Verificar se contém alguma das palavras-chave
            let triggered = self.keywords.iter().any(|kw| lower.contains(kw.as_str()));
            if triggered {
                info!("🔔 Hot-word detectada: '{lower}'");
                last_trigger = Instant::now();
                // Limpar buffer para não re-transcrever o mesmo áudio
                buffer.lock().unwrap().clear();

                let event = HotWordEvent { timestamp: std::time::SystemTime::now() };
                if tx.blocking_send(event).is_err() {
                    break; // receiver caiu, encerrar
                }
            }
        }

        Ok(())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn rms_energy(samples: &[f32]) -> f32 {
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn quick_transcribe(ctx: &whisper_rs::WhisperContext, samples: &[f32]) -> Result<String> {
    use whisper_rs::{FullParams, SamplingStrategy};

    let mut state = ctx.create_state()
        .map_err(|e| anyhow::anyhow!("Whisper state: {e}"))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("pt"));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_no_context(true); // mais rápido, sem contexto anterior
    params.set_max_tokens(32);   // só precisamos de poucas palavras

    state.full(params, samples)
        .map_err(|e| anyhow::anyhow!("Whisper full: {e}"))?;

    let n = state.full_n_segments().unwrap_or(0);
    let mut out = String::new();
    for i in 0..n {
        if let Ok(seg) = state.full_get_segment_text(i) {
            out.push_str(seg.trim());
            out.push(' ');
        }
    }
    Ok(out.trim().to_string())
}