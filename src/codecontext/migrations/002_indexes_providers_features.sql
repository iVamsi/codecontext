PRAGMA foreign_keys = ON;

-- 1. First-class GitLab & workspace base metadata
ALTER TABLE store_meta ADD COLUMN gitlab_project_id INTEGER;
ALTER TABLE store_meta ADD COLUMN gitlab_name TEXT;
ALTER TABLE store_meta ADD COLUMN gitlab_base_url TEXT;
ALTER TABLE store_meta ADD COLUMN workspace_base_revision TEXT;

-- 2. Missing high-impact indexes identified via EXPLAIN QUERY PLAN
CREATE INDEX IF NOT EXISTS syntax_occ_ext_kind ON syntax_occurrences(extraction_id, kind);
CREATE INDEX IF NOT EXISTS syntax_occ_enclosing ON syntax_occurrences(enclosing_symbol_id);
CREATE INDEX IF NOT EXISTS snapshot_files_ext ON snapshot_files(extraction_id);
CREATE INDEX IF NOT EXISTS snapshot_files_blob ON snapshot_files(blob_oid);
CREATE INDEX IF NOT EXISTS symbols_qualified_name ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS symbols_parent ON symbols(parent_symbol_id);
CREATE INDEX IF NOT EXISTS relations_method_state ON relations(method, state);
CREATE INDEX IF NOT EXISTS evidence_git_oid_kind ON evidence(git_oid, kind);
CREATE INDEX IF NOT EXISTS sources_parent ON sources(parent_source_id);
CREATE INDEX IF NOT EXISTS episodes_commit ON episodes(commit_oid);
CREATE INDEX IF NOT EXISTS episodes_primary_source ON episodes(primary_source_id);
CREATE INDEX IF NOT EXISTS commits_diff_state ON commits(diff_state, first_seen_at_us);
CREATE INDEX IF NOT EXISTS commit_parents_parent ON commit_parents(parent_oid);
CREATE INDEX IF NOT EXISTS search_code_path ON search_documents(
  record_kind, json_extract(metadata_json, '$.path')
) WHERE record_kind = 'code';

-- 3. Multi-branch / worktree snapshot cache (reusing content-addressed extractions)
CREATE TABLE IF NOT EXISTS revision_snapshots (
  revision TEXT NOT NULL,
  path BLOB NOT NULL,
  blob_oid TEXT NOT NULL,
  mode INTEGER NOT NULL,
  extraction_id TEXT REFERENCES extractions(id),
  content_state TEXT NOT NULL,
  prepared_at_us INTEGER NOT NULL,
  PRIMARY KEY (revision, path)
);
CREATE INDEX IF NOT EXISTS revision_snapshots_ext ON revision_snapshots(revision, extraction_id);

CREATE TABLE IF NOT EXISTS revision_ancestors (
  revision TEXT NOT NULL,
  commit_oid TEXT NOT NULL,
  PRIMARY KEY (revision, commit_oid)
);

-- 4. Optional semantic embeddings for hybrid RRF retrieval
CREATE TABLE IF NOT EXISTS document_embeddings (
  record_kind TEXT NOT NULL,
  record_id TEXT NOT NULL,
  chunk_ordinal INTEGER NOT NULL,
  model_fingerprint TEXT NOT NULL,
  dimensions INTEGER NOT NULL CHECK (dimensions > 0),
  vector_f32 BLOB NOT NULL,
  updated_at_us INTEGER NOT NULL,
  PRIMARY KEY (record_kind, record_id, chunk_ordinal, model_fingerprint),
  FOREIGN KEY (record_kind, record_id, chunk_ordinal)
    REFERENCES search_documents(record_kind, record_id, chunk_ordinal) ON DELETE CASCADE
);
