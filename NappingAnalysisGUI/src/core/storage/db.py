# src/core/storage/db.py
import os
import sqlite3
from src.core.utils.paths import data_path

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS protocols (
  id TEXT PRIMARY KEY,                 -- UUID
  name TEXT NOT NULL UNIQUE,
  goal TEXT,
  hypotheses TEXT,
  instruction_type TEXT NOT NULL,      -- audio|image|video
  modules_enabled TEXT NOT NULL,        -- JSON array string
  data_to_export TEXT NOT NULL,         -- JSON array string
  locked INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS instruction_assets (
  id TEXT PRIMARY KEY,                 -- UUID
  protocol_id TEXT NOT NULL,
  asset_type TEXT NOT NULL,            -- audio|image|video
  path TEXT NOT NULL,
  meta TEXT NOT NULL,                  -- JSON string
  created_at TEXT NOT NULL,
  FOREIGN KEY(protocol_id) REFERENCES protocols(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS timeline_steps (
  id TEXT PRIMARY KEY,                 -- UUID
  protocol_id TEXT NOT NULL,
  order_index INTEGER NOT NULL,
  asset_ref TEXT,                      -- instruction_assets.id ou NULL si pause/trigger
  duration_s REAL NOT NULL,
  label TEXT NOT NULL,
  repeat INTEGER,
  pause INTEGER,
  trigger TEXT,
  FOREIGN KEY(protocol_id) REFERENCES protocols(id) ON DELETE CASCADE,
  FOREIGN KEY(asset_ref) REFERENCES instruction_assets(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS protocol_participants (
  protocol_id TEXT NOT NULL,
  participant_id TEXT NOT NULL,
  PRIMARY KEY(protocol_id, participant_id),
  FOREIGN KEY(protocol_id) REFERENCES protocols(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,                 -- UUID
  protocol_id TEXT NOT NULL,
  participant_id TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  output_dir TEXT NOT NULL,
  FOREIGN KEY(protocol_id) REFERENCES protocols(id)
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  t TEXT NOT NULL,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,               -- JSON string
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_protocol ON sessions(protocol_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
"""

def get_db_path() -> str:
    os.makedirs(data_path(), exist_ok=True)
    return data_path("app.db")

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db() -> None:
    conn = connect()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1');")
        conn.commit()
    finally:
        conn.close()
