PRAGMA foreign_keys = ON;

CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at_us INTEGER NOT NULL,
  migration_sha256 TEXT NOT NULL
);

CREATE TABLE store_meta (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  repository_id TEXT NOT NULL UNIQUE,
  schema_version INTEGER NOT NULL,
  object_format TEXT NOT NULL,
  local_repository_path TEXT,
  common_dir_realpath TEXT,
  common_dir_device INTEGER,
  common_dir_inode INTEGER,
  github_repository_id INTEGER,
  github_name TEXT,
  principal_id TEXT,
  active_revision TEXT,
  generation INTEGER NOT NULL DEFAULT 0,
  access_epoch INTEGER NOT NULL DEFAULT 0,
  access_state TEXT NOT NULL DEFAULT 'allowed',
  created_at_us INTEGER NOT NULL
);

CREATE TABLE refs (
  name TEXT PRIMARY KEY,
  commit_oid TEXT NOT NULL,
  source TEXT NOT NULL,
  last_seen_at_us INTEGER NOT NULL,
  active INTEGER NOT NULL CHECK (active IN (0,1))
);

CREATE TABLE commits (
  oid TEXT PRIMARY KEY,
  tree_oid TEXT NOT NULL,
  authored_at_us INTEGER,
  committed_at_us INTEGER,
  author_name TEXT,
  author_email TEXT,
  message TEXT NOT NULL,
  first_seen_at_us INTEGER NOT NULL,
  diff_state TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE commit_parents (
  commit_oid TEXT NOT NULL REFERENCES commits(oid) ON DELETE CASCADE,
  parent_index INTEGER NOT NULL CHECK (parent_index >= 0),
  parent_oid TEXT NOT NULL,
  PRIMARY KEY (commit_oid, parent_index)
);

CREATE TABLE changes (
  id TEXT PRIMARY KEY,
  commit_oid TEXT NOT NULL REFERENCES commits(oid),
  parent_index INTEGER NOT NULL,
  parent_oid TEXT,
  old_path BLOB,
  new_path BLOB,
  old_blob_oid TEXT,
  new_blob_oid TEXT,
  old_mode INTEGER,
  new_mode INTEGER,
  change_kind TEXT NOT NULL,
  rename_score INTEGER,
  hunks_state TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE hunks (
  id TEXT PRIMARY KEY,
  change_id TEXT NOT NULL REFERENCES changes(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  old_start INTEGER NOT NULL,
  old_count INTEGER NOT NULL,
  new_start INTEGER NOT NULL,
  new_count INTEGER NOT NULL,
  patch_text TEXT,
  UNIQUE (change_id, ordinal)
);

CREATE TABLE extractions (
  id TEXT PRIMARY KEY,
  blob_oid TEXT NOT NULL,
  language TEXT NOT NULL,
  extractor_fingerprint TEXT NOT NULL,
  state TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL DEFAULT '[]',
  UNIQUE (blob_oid, language, extractor_fingerprint)
);

CREATE TABLE symbols (
  id TEXT PRIMARY KEY,
  extraction_id TEXT NOT NULL REFERENCES extractions(id) ON DELETE CASCADE,
  parent_symbol_id TEXT REFERENCES symbols(id),
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  qualified_name TEXT NOT NULL,
  byte_start INTEGER NOT NULL,
  byte_end INTEGER NOT NULL,
  body_byte_start INTEGER,
  body_byte_end INTEGER,
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  signature TEXT NOT NULL,
  CHECK (byte_start >= 0 AND byte_end >= byte_start),
  CHECK ((body_byte_start IS NULL AND body_byte_end IS NULL) OR
    (body_byte_start IS NOT NULL AND body_byte_end IS NOT NULL AND
     body_byte_start >= byte_start AND body_byte_end >= body_byte_start AND body_byte_end <= byte_end)),
  CHECK (line_start >= 1 AND line_end >= line_start)
);

CREATE TABLE syntax_occurrences (
  id TEXT PRIMARY KEY,
  extraction_id TEXT NOT NULL REFERENCES extractions(id) ON DELETE CASCADE,
  enclosing_symbol_id TEXT REFERENCES symbols(id),
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  byte_start INTEGER NOT NULL,
  byte_end INTEGER NOT NULL,
  details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE snapshot_files (
  path BLOB PRIMARY KEY,
  blob_oid TEXT NOT NULL,
  mode INTEGER NOT NULL,
  extraction_id TEXT REFERENCES extractions(id),
  content_state TEXT NOT NULL
);

CREATE TABLE snapshot_commits (
  commit_oid TEXT PRIMARY KEY
);

CREATE TABLE sources (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  repository_provider_id TEXT NOT NULL,
  provider_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  parent_source_id TEXT REFERENCES sources(id),
  current_version_id TEXT,
  availability TEXT NOT NULL DEFAULT 'available',
  last_seen_at_us INTEGER NOT NULL,
  last_reconciled_at_us INTEGER,
  UNIQUE (provider, repository_provider_id, kind, provider_id)
);

CREATE TABLE source_versions (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  content_sha256 TEXT NOT NULL,
  title TEXT,
  body TEXT,
  raw_payload_json TEXT NOT NULL,
  author_json TEXT,
  source_url TEXT,
  authored_at_us INTEGER,
  updated_at_us INTEGER,
  first_observed_at_us INTEGER NOT NULL,
  last_observed_at_us INTEGER NOT NULL,
  UNIQUE (source_id, content_sha256)
);

CREATE TABLE evidence (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  source_version_id TEXT REFERENCES source_versions(id) ON DELETE CASCADE,
  git_oid TEXT,
  anchor_json TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'available'
);

CREATE TABLE relations (
  id TEXT PRIMARY KEY,
  from_kind TEXT NOT NULL,
  from_id TEXT NOT NULL,
  to_kind TEXT NOT NULL,
  to_id TEXT NOT NULL,
  relation_kind TEXT NOT NULL,
  method TEXT NOT NULL,
  mapping TEXT NOT NULL,
  evidence_id TEXT REFERENCES evidence(id),
  details_json TEXT NOT NULL DEFAULT '{}',
  state TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE episodes (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  primary_source_id TEXT REFERENCES sources(id),
  commit_oid TEXT REFERENCES commits(oid),
  input_fingerprint TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending',
  inspected_at_us INTEGER
);

CREATE TABLE model_runs (
  id TEXT PRIMARY KEY,
  model_fingerprint TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  state TEXT NOT NULL,
  reserved_input_tokens INTEGER NOT NULL,
  reserved_output_tokens INTEGER NOT NULL,
  actual_input_tokens INTEGER,
  actual_output_tokens INTEGER,
  cost_decimal TEXT,
  error_code TEXT,
  created_at_us INTEGER NOT NULL,
  finished_at_us INTEGER
);

CREATE TABLE claims (
  id TEXT PRIMARY KEY,
  episode_id TEXT NOT NULL REFERENCES episodes(id),
  model_run_id TEXT NOT NULL REFERENCES model_runs(id),
  statement TEXT NOT NULL,
  rationale TEXT,
  alternatives_json TEXT NOT NULL DEFAULT '[]',
  constraints_json TEXT NOT NULL DEFAULT '[]',
  support TEXT NOT NULL,
  lifecycle TEXT NOT NULL,
  attribution_json TEXT NOT NULL DEFAULT '{}',
  state TEXT NOT NULL DEFAULT 'ready',
  created_at_us INTEGER NOT NULL
);

CREATE TABLE episode_segments (
  episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  input_fingerprint TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  required_slots_json TEXT NOT NULL,
  inspected_slots_json TEXT NOT NULL DEFAULT '[]',
  unresolved_slots_json TEXT NOT NULL DEFAULT '[]',
  state TEXT NOT NULL DEFAULT 'pending',
  model_run_id TEXT REFERENCES model_runs(id),
  PRIMARY KEY (episode_id, input_fingerprint, segment_id),
  UNIQUE (episode_id, input_fingerprint, ordinal)
);

CREATE TABLE claim_evidence (
  claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
  evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  details_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (claim_id, evidence_id, role)
);

CREATE TABLE summaries (
  id TEXT PRIMARY KEY,
  scope_kind TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  revision TEXT NOT NULL,
  model_run_id TEXT REFERENCES model_runs(id),
  text TEXT NOT NULL,
  statements_json TEXT NOT NULL,
  coverage_json TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'ready',
  input_fingerprint TEXT NOT NULL
);

CREATE TABLE dependencies (
  dependent_kind TEXT NOT NULL,
  dependent_id TEXT NOT NULL,
  input_kind TEXT NOT NULL,
  input_id TEXT NOT NULL,
  PRIMARY KEY (dependent_kind, dependent_id, input_kind, input_id)
);

CREATE TABLE jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  priority INTEGER NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending',
  cursor_json TEXT NOT NULL DEFAULT '{}',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error_code TEXT,
  created_at_us INTEGER NOT NULL,
  updated_at_us INTEGER NOT NULL,
  UNIQUE (kind, owner_id)
);

CREATE TABLE checkpoints (
  stream_key TEXT PRIMARY KEY,
  cursor_json TEXT NOT NULL,
  etag TEXT,
  last_modified TEXT,
  high_watermark_us INTEGER,
  last_success_at_us INTEGER,
  last_reconciled_at_us INTEGER
);

CREATE TABLE coverage_gaps (
  id TEXT PRIMARY KEY,
  code TEXT NOT NULL,
  scope TEXT NOT NULL,
  record_id TEXT,
  message TEXT NOT NULL,
  retryable INTEGER NOT NULL CHECK (retryable IN (0,1)),
  active INTEGER NOT NULL CHECK (active IN (0,1)),
  first_seen_at_us INTEGER NOT NULL,
  last_seen_at_us INTEGER NOT NULL
);

CREATE TABLE reconciliation_seen (
  stream_key TEXT NOT NULL REFERENCES checkpoints(stream_key) ON DELETE CASCADE,
  pass_id TEXT NOT NULL,
  source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  PRIMARY KEY (stream_key, pass_id, source_id)
);

CREATE TABLE search_documents (
  rowid INTEGER PRIMARY KEY,
  record_kind TEXT NOT NULL,
  record_id TEXT NOT NULL,
  chunk_ordinal INTEGER NOT NULL,
  title TEXT NOT NULL,
  identifiers TEXT NOT NULL,
  body TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  UNIQUE (record_kind, record_id, chunk_ordinal)
);

CREATE VIRTUAL TABLE search_fts USING fts5(
  title, identifiers, body,
  content='search_documents', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER search_documents_ai AFTER INSERT ON search_documents BEGIN
  INSERT INTO search_fts(rowid,title,identifiers,body)
    VALUES(new.rowid,new.title,new.identifiers,new.body);
END;
CREATE TRIGGER search_documents_ad AFTER DELETE ON search_documents BEGIN
  INSERT INTO search_fts(search_fts,rowid,title,identifiers,body)
    VALUES('delete',old.rowid,old.title,old.identifiers,old.body);
END;
CREATE TRIGGER search_documents_au AFTER UPDATE ON search_documents BEGIN
  INSERT INTO search_fts(search_fts,rowid,title,identifiers,body)
    VALUES('delete',old.rowid,old.title,old.identifiers,old.body);
  INSERT INTO search_fts(rowid,title,identifiers,body)
    VALUES(new.rowid,new.title,new.identifiers,new.body);
END;

CREATE INDEX changes_commit ON changes(commit_oid,parent_index);
CREATE INDEX changes_old_path ON changes(old_path,commit_oid);
CREATE INDEX changes_new_path ON changes(new_path,commit_oid);
CREATE INDEX changes_old_blob ON changes(old_blob_oid);
CREATE INDEX changes_new_blob ON changes(new_blob_oid);
CREATE INDEX symbols_name ON symbols(name,qualified_name);
CREATE INDEX symbols_extraction ON symbols(extraction_id,byte_start,byte_end);
CREATE INDEX versions_source_time ON source_versions(source_id,updated_at_us);
CREATE INDEX evidence_version ON evidence(source_version_id);
CREATE INDEX relations_from ON relations(from_kind,from_id,relation_kind,state);
CREATE INDEX relations_to ON relations(to_kind,to_id,relation_kind,state);
CREATE INDEX claims_episode ON claims(episode_id,state);
CREATE INDEX claim_evidence_reverse ON claim_evidence(evidence_id,claim_id);
CREATE INDEX dependencies_reverse ON dependencies(input_kind,input_id);
CREATE INDEX jobs_queue ON jobs(state,priority,updated_at_us);
CREATE INDEX search_record ON search_documents(record_kind,record_id);
