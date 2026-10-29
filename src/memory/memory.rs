use anyhow::{anyhow, Result};
use chrono::Utc;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::MemoryConfig;

// ─── Tipos públicos ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Episode {
    pub id:         String,
    pub input:      String,
    pub response:   String,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct SemanticHit {
    pub text:  String,
    pub score: f32,
}

// ─── Payloads Qdrant (HTTP direto, sem SDK pesado) ────────────────────────────

#[derive(Serialize)]
struct QdrantUpsertRequest {
    points: Vec<QdrantPoint>,
}

#[derive(Serialize)]
struct QdrantPoint {
    id:      String,
    vector:  Vec<f32>,
    payload: QdrantPayload,
}

#[derive(Serialize, Deserialize)]
struct QdrantPayload {
    text:       String,
    created_at: String,
}

#[derive(Serialize)]
struct QdrantSearchRequest {
    vector:       Vec<f32>,
    limit:        usize,
    with_payload: bool,
}

#[derive(Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantSearchHit>,
}

#[derive(Deserialize)]
struct QdrantSearchHit {
    score:   f32,
    payload: Option<QdrantPayload>,
}

// ─── Store principal ──────────────────────────────────────────────────────────

pub struct MemoryStore {
    config:         MemoryConfig,
    db_path:        String,
    http:           reqwest::Client,
    qdrant_ready:   bool,
    collection_url: String,
}

impl MemoryStore {
    pub async fn new(config: &MemoryConfig) -> Result<Self> {
        let db_path = shellexpand::tilde(&config.sqlite_path).to_string();
        Self::init_sqlite(&db_path)?;

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        let collection_url = format!(
            "{}/collections/{}",
            config.qdrant_url.trim_end_matches('/'),
            config.collection_name
        );

        let qdrant_ready =
            Self::ensure_collection(&http, &collection_url, config.embedding_dim).await;

        if qdrant_ready {
            info!("Qdrant conectado — coleção '{}' pronta", config.collection_name);
        } else {
            warn!(
                "Qdrant indisponível em {}. Memória semântica desativada.\n  \
                 Inicie com: docker run -d -p 6333:6333 qdrant/qdrant",
                config.qdrant_url
            );
        }

        Ok(Self { config: config.clone(), db_path, http, qdrant_ready, collection_url })
    }

    // ─── SQLite — memória episódica ───────────────────────────────────────────

    fn init_sqlite(path: &str) -> Result<()> {
        use std::path::Path;
        if let Some(dir) = Path::new(path).parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous  = NORMAL;
             CREATE TABLE IF NOT EXISTS episodes (
                 id          TEXT PRIMARY KEY,
                 input       TEXT NOT NULL,
                 response    TEXT NOT NULL,
                 created_at  TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at DESC);
             CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                 USING fts5(id UNINDEXED, input, response);",
        )?;
        debug!("SQLite inicializado: {path}");
        Ok(())
    }

    pub async fn save_episode(&self, input: &str, response: &str) -> Result<()> {
        let db   = self.db_path.clone();
        let id   = Uuid::new_v4().to_string();
        let now  = Utc::now().to_rfc3339();
        let inp  = input.to_string();
        let resp = response.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = Connection::open(&db)?;
            conn.execute(
                "INSERT INTO episodes (id, input, response, created_at) VALUES (?1,?2,?3,?4)",
                params![id, inp, resp, now],
            )?;
            conn.execute(
                "INSERT INTO episodes_fts (id, input, response) VALUES (?1,?2,?3)",
                params![id, inp, resp],
            )?;
            Ok(())
        }).await??;
        debug!("Episódio salvo");
        Ok(())
    }

    /// Busca por texto usando FTS5 (sem precisar de vetor)
    pub async fn search_episodes_fts(&self, query: &str, limit: usize) -> Result<Vec<Episode>> {
        let db    = self.db_path.clone();
        let query = query.replace('\'', "''");
        tokio::task::spawn_blocking(move || -> Result<Vec<Episode>> {
            let conn = Connection::open(&db)?;
            let sql = format!(
                "SELECT e.id, e.input, e.response, e.created_at
                 FROM episodes e JOIN episodes_fts f ON e.id = f.id
                 WHERE episodes_fts MATCH '{query}' ORDER BY rank LIMIT {limit}"
            );
            let mut stmt = conn.prepare(&sql)?;
            Ok(stmt.query_map([], |row| Ok(Episode {
                id:         row.get(0)?,
                input:      row.get(1)?,
                response:   row.get(2)?,
                created_at: row.get(3)?,
            }))?
                .filter_map(|r| r.ok())
                .collect())
        }).await?
    }

    pub async fn recent_episodes(&self, limit: usize) -> Result<Vec<Episode>> {
        let db = self.db_path.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Episode>> {
            let conn = Connection::open(&db)?;
            let mut stmt = conn.prepare(
                "SELECT id, input, response, created_at FROM episodes
                 ORDER BY created_at DESC LIMIT ?1",
            )?;
            Ok(stmt.query_map(params![limit as i64], |row| Ok(Episode {
                id:         row.get(0)?,
                input:      row.get(1)?,
                response:   row.get(2)?,
                created_at: row.get(3)?,
            }))?
                .filter_map(|r| r.ok())
                .collect())
        }).await?
    }

    pub async fn episode_count(&self) -> Result<i64> {
        let db = self.db_path.clone();
        tokio::task::spawn_blocking(move || -> Result<i64> {
            let conn = Connection::open(&db)?;
            Ok(conn.query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))?)
        }).await?
    }

    // ─── Qdrant — memória semântica ───────────────────────────────────────────

    /// Salva texto + vetor. O embedding é gerado externamente (InferenceEngine).
    pub async fn save_semantic(&self, text: &str, embedding: Vec<f32>) -> Result<()> {
        if !self.qdrant_ready { return Ok(()); }
        let body = QdrantUpsertRequest {
            points: vec![QdrantPoint {
                id:     Uuid::new_v4().to_string(),
                vector: embedding,
                payload: QdrantPayload {
                    text:       text.to_string(),
                    created_at: Utc::now().to_rfc3339(),
                },
            }],
        };
        let url  = format!("{}/points", self.collection_url);
        let resp = self.http.put(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            return Err(anyhow!("Qdrant upsert: {}", resp.text().await.unwrap_or_default()));
        }
        debug!("Vetor salvo no Qdrant");
        Ok(())
    }

    pub async fn search_semantic(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<SemanticHit>> {
        if !self.qdrant_ready { return Ok(vec![]); }
        let url  = format!("{}/points/search", self.collection_url);
        let req  = QdrantSearchRequest { vector: query_embedding, limit: top_k, with_payload: true };
        let resp = self.http.post(&url).json(&req).send().await?;
        if !resp.status().is_success() {
            return Err(anyhow!("Qdrant search: {}", resp.text().await.unwrap_or_default()));
        }
        let search: QdrantSearchResponse = resp.json().await?;
        Ok(search.result.into_iter().filter_map(|h| {
            h.payload.map(|p| SemanticHit { text: p.text, score: h.score })
        }).collect())
    }

    pub async fn clear_semantic(&self) -> Result<()> {
        if !self.qdrant_ready { return Ok(()); }
        let url  = format!("{}/points/delete", self.collection_url);
        let body = serde_json::json!({"filter": {"must": []}});
        self.http.post(&url).json(&body).send().await?;
        info!("Memória semântica limpa");
        Ok(())
    }

    pub fn qdrant_online(&self) -> bool { self.qdrant_ready }

    // ─── Gestão de coleção Qdrant ─────────────────────────────────────────────

    async fn ensure_collection(http: &reqwest::Client, url: &str, dim: usize) -> bool {
        match http.get(url).send().await {
            Ok(r) if r.status().is_success() => return true,
            Err(_) => return false,
            Ok(_)  => {}
        }
        #[derive(Serialize)]
        struct CC { vectors: VC }
        #[derive(Serialize)]
        struct VC { size: usize, distance: &'static str }
        let body = CC { vectors: VC { size: dim, distance: "Cosine" } };
        match http.put(url).json(&body).send().await {
            Ok(r) if r.status().is_success() => { info!("Coleção Qdrant criada (dim={dim})"); true }
            Ok(r) => { warn!("Falha ao criar coleção: {}", r.status()); false }
            Err(e) => { warn!("Qdrant offline: {e}"); false }
        }
    }
}