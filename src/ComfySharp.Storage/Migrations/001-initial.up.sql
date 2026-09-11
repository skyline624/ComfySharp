CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE assets (
  id TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  relative_path TEXT NOT NULL UNIQUE,
  media_type TEXT NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  missing INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX assets_hash ON assets(sha256);
CREATE TABLE asset_tags (
  asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  tag TEXT NOT NULL COLLATE BINARY,
  PRIMARY KEY(asset_id, tag)
);
