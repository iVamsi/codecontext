# Storage, identity, transactions, and retention

Read [shared contracts](00-contracts.md) first. This chapter specifies the initial relational design. The SQL is a planned migration, not an installed database. Implement the tables when their task is reached; the final schema must have these semantics.

## 1. Storage rules

1. The bare Git directory owns code bytes and tree/commit objects. SQLite owns normalized metadata, discussion content, extraction results, and derived knowledge. Never copy whole historical source trees into SQLite.
2. A `symbols` row belongs to a unique blob extraction. The active snapshot maps a path to a blob; it does not duplicate that blob's symbols. Historical occurrences are constructed from Git tree membership and anchors when needed.
3. A discussion source has a stable identity and multiple immutable content versions. Its current pointer changes atomically with invalidation. Historical versions are retained unless explicitly purged/redacted.
4. Git content addressing is used only for Git objects. Project-generated IDs use the canonical SHA-256 identity algorithm from the contracts.
5. Source records and generated claims are separate tables. Generated text never replaces original text.
6. One SQLite writer is permitted at a time. Readers use independent read transactions. No full-database copy is required for each published generation.

## 2. Planned schema

All `*_us` columns hold integer UTC microseconds. `*_json` columns contain validated canonical JSON. JSON shape is validated in Pydantic before insertion. `NULL` means unknown or inapplicable, never an empty string sentinel. All ordinary IDs are TEXT; Git paths are BLOB. Boolean columns are INTEGER with values 0 or 1. Persist complete Git object IDs.

Use this as the v1 logical migration. Add the explicit indexes below. Application validation enforces enum values and polymorphic references, while foreign keys enforce direct ownership. Do not add a generic ORM or mirror the entire GitHub API schema.

```sql
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
```

The schema deliberately avoids a global transitive-closure table, a row for every symbol at every commit, and an embedding column before semantic retrieval is justified. Additive indexes are permitted after measuring actual query plans. Table renames or changed public semantics require updating this specification and migration tests together.

## 3. Additional table semantics

- `commits.diff_state`: `pending`, `complete`, or `partial`. Roots have changes with `parent_index=-1` and `parent_oid=NULL`; they have no `commit_parents` row. Missing parent objects are retained as textual parent IDs with a gap, not dropped by a foreign-key constraint.
- `changes.change_kind`: `added`, `modified`, `deleted`, `renamed`, `copied`, `type_changed`, `unmerged`, or `unknown`. No all-zero pseudo-object ID is stored; absence is NULL. Normalize Git modes to integers without losing executable/symlink/submodule distinctions.
- `extractions.state`: `complete`, `partial`, `unsupported`, `oversized`, `binary`, or `error`. Partial parse results may expose valid definitions with diagnostics; parse errors never mean zero symbols with complete coverage.
- `snapshot_files.content_state`: `indexed`, `text_only`, `metadata_only`, or `pending`. Replace only this one active map, preferably by staging a temporary table and swapping its content inside one transaction.
- `sources.kind`: `issue`, `pull_request`, `issue_comment`, `review`, `review_comment`, or `timeline_event`. PRs also appear in issue endpoints: normalize to one PR identity and fetch conversation comments through the issues endpoint; never double-count one PR as two independent records.
- `sources.availability`: `available`, `unverified`, `inaccessible`, `redacted`, or `deleted`. A single 404 cannot decide between deletion and inaccessible content. Purge and confirmed access changes have separate control operations.
- `sources.current_version_id` is application-validated to point to a version owned by that source; the circular ownership is intentionally not an immediate SQL foreign key.
- `evidence.kind`: `discussion`, `code`, `commit`, or `hunk`. Discussion evidence has a source-version foreign key; Git evidence has an owned object reference plus anchor. The anchor stores path bytes as base64, revision, original review commit where relevant, byte/line spans, and the underlying record ID. The referenced bytes must hash to `content_sha256`.
- `relations.mapping`: exactly `exact`, `candidate`, or `unmapped`. Relation kinds used in v1: `contains`, `imports_candidate`, `calls_candidate`, `changes`, `associated_pr`, `references_issue`, `review_anchor`, `landed_mapping`, `targets`, `supports`, `contradicts`, `supersedes`, and `corroborates`. A `targets` edge connects a claim to code-anchor evidence containing revision/path/blob/range. Relation certainty does not determine claim support or applicability.
- Episodes are `pr` or `commit`; a PR can own multiple commits via relations. Unmerged/rejected PRs are retained as historical alternatives, but are not implicitly ancestors of current code.
- `episode_segments` records the complete deterministic segmentation for one input fingerprint before calls start. Each segment's slot arrays are bounded by the per-call input cap. Only validated acknowledged slots become inspected; missing/unresolved slots leave the segment partial. An episode is inspected only when every required segment for the current fingerprint is complete. Source edits create a new fingerprint and leave old inspection results ineligible. Segment dependencies include every supplied source version, extraction, and evidence record.
- `symbols.byte_start/end` describe the full declaration/display interval, including decorators where applicable; nullable `body_byte_start/end` describe its body for insertion-boundary mapping. Abstract declarations have no body. Docstrings are `syntax_occurrences(kind='docstring')` with exact spans and their enclosing symbol.
- `summaries.statements_json` stores validated bounded statements and per-statement evidence refs defined in the enrichment chapter; `text` is a deterministic joined rendering of those statements. It is never an uncited free-form replacement. `coverage_json` contains `file_count`, `group_count`, `included_file_count`, `included_group_count`, `included_claim_count`, `omitted_claim_count`, and `partial`, as declared by `Overview` in the public models.
- `checkpoints.cursor_json` stores active reconciliation pass ID, start time, and state. `reconciliation_seen` stores membership page by page in the same transaction as its cursor. Compare absence only after a successful full pass; delete obsolete membership after committing reconciliation. Never infer absence from an interrupted pass or store an unbounded ID list in JSON.
- `claims.state` and `summaries.state`: `ready`, `stale`, or `retracted`. `claim_evidence.role`: `support`, `counterevidence`, `lifecycle`, or `applicability`. Claim target associations use typed relations and anchors rather than a single path column.
- Dependency input kinds include `source_version`, `evidence`, `extraction`, `claim`, `summary`, and `snapshot`. Dependencies include all model inputs, not only the few sources quoted in output. A discarded input may still have influenced generation.
- Jobs are `pending`, `running`, `complete`, `retryable_error`, or `blocked`. A crash-recovery pass resets abandoned `running` jobs to `pending` before starting new work. Model reservations need separate recovery semantics; do not auto-repeat a possibly charged request.
- Search metadata includes source/commit associations, language, byte span, eligibility category, and summary/claim state. It is a cache of filter fields; canonical tables remain authoritative for authorization and historical eligibility.

## 4. Transaction recipes

At writable startup configure WAL, `foreign_keys=ON`, `synchronous=FULL`, and a 5-second busy timeout. Do not change these on a read-only connection. Resolve and validate the schema version before loading business records. Read-only connections use SQLite URI read-only mode; they never attempt a migration.

Use short transactions. HTTP requests, Git fetches, parsing, and model calls happen outside write transactions. Write their bounded validated results with `BEGIN IMMEDIATE`; commit the new rows, dependency invalidation, FTS updates, gap/checkpoint changes, and incremented `generation` together. SQLite busy expiration becomes `STORE_BUSY` with retryable=true. Do not spin indefinitely.

### Source refresh

```text
fetch outside transaction
normalize and hash source payload
BEGIN IMMEDIATE
  upsert source identity
  if current hash is unchanged:
    update last-observed/checkpoint timestamps
  else:
    insert or reuse immutable source version
    set current_version_id
    register exact evidence locators
    recursively mark dependent claims/summaries stale
    remove stale generated search documents
    update current-source search documents
    enqueue the affected enrichment episode
  update coverage and checkpoint
  increment generation
COMMIT
```

Recursive invalidation uses an iterative queue and visited set over the reverse-dependency index. A derivation cycle is an implementation error; guard it and add a regression fixture. Stale derivations remain inspectable through explicit `get` with a status warning but are not ranked as usable current claims. A source version's historical body does not change when the current pointer moves.

### Snapshot preparation

Resolve the requested source revision once at run start. Ensure its objects are owned locally. Build file/tree membership and reachable commit IDs into staging tables; parse necessary unique active blobs outside the write transaction. Then atomically replace `snapshot_files` and `snapshot_commits`, set `active_revision`, update active code search documents, stale old revision-specific summaries, and increment generation. A crash before publication leaves the prior prepared revision usable. An initial partial preparation may publish files with `pending` extraction and explicit coverage; it must not pretend absent symbols were parsed.

### Claim publication

Reserve model budget and record request fingerprint before the external call. Validate model output and every evidence span after the call. In the publication transaction re-read all input fingerprints. If any source changed, store the run's usage, discard the would-be active claims, and leave the episode pending. Otherwise insert validated claims, evidence links, full dependencies, and search records atomically. Duplicate successful request fingerprints reuse results; concurrent completions must not create duplicate claims.

### Read snapshot

Start a read transaction; capture `generation`, `access_epoch`, active revision, and access state. Resolve all references and run retrieval inside that snapshot. Release it before optional workspace filesystem inspection. If workspace data is included, identify it as a separate observation with its own timestamp/hash; it is not part of SQLite's atomic snapshot. Immediately before emission check `access_epoch` through a fresh lightweight connection; if it changed, discard the result and return `CURSOR_STALE` or `ACCESS_BLOCKED` as appropriate. This validation is the authorization point: purge cannot recall results already authorized or delivered. Generation changes alone do not invalidate an already consistent completed read.

## 5. FTS maintenance and eligibility

Use one external-content FTS5 table with trigger-maintained index consistency. Use parameterized MATCH expressions generated by the retrieval module; never pass user query text as raw FTS syntax. In the early slice index current source versions, active code, historical commit/change text, ready claims, and ready summaries. Earlier source versions remain available by ID. The final release retains all eligible observed version documents for cutoff-aware search as described below.

Cutoff searches must not silently use a newer version's indexed text to claim an older version matched. If a matching current record is ineligible, the historical search path may search eligible stored versions in a bounded source-candidate set and must report incomplete historical-search coverage when earlier versions were never indexed. This limit is explicit in the release documentation; current-knowledge retrieval remains the primary v1 workflow.

Task T14 retains version documents in FTS with `current=false` metadata and filters each query to the eligible version of each source. Current-only documents are permitted only during the early slice, with an explicit historical-search gap. Limit duplicates by collapsing ranked hits to source/episode after version eligibility.

Run FTS integrity checks in tests and maintenance commands, not on every query. SQLite documents the need to maintain external-content indexes consistently; the trigger approach follows that mechanism. See [FTS5 external-content tables](https://www.sqlite.org/fts5.html#external_content_tables).

## 6. Content retention and access

Initialization binds private source data to one configured principal after authentication succeeds. Credentials are read from `GITHUB_TOKEN` at collection time. Store the provider's principal identifier, not the credential. A different principal cannot silently reuse private evidence; require a separate store. Public/local-only evidence may use a null principal.

Offline queries read the last authorized local archive. They do not prove current remote authorization. Authentication failures or ambiguous 404s mark synchronization unavailable/unverified; they do not automatically erase local history. A confirmed repository-access removal sets access_state=blocked and increments access_epoch before cleanup. `ACCESS_BLOCKED` applies to all read paths, including direct IDs and exported-cache regeneration.

Deletion/redaction of a discussion is handled by a purge transaction. First compute the full reverse dependency closure, including claims and summaries influenced by the source even if they cite other sources too. Remove generated bodies in that closure, associated FTS documents, caches, segment outputs, and source-dependent relations. Delete relation rows referencing affected evidence before deleting evidence because that foreign key does not cascade. Remove affected claim-evidence rows, dependency edges, source versions/raw bodies, and obsolete episode links in a foreign-key-safe order. Retain only a non-content source tombstone. Previously known purged version/evidence/claim/summary IDs return `RECORD_NOT_FOUND`; marking them stale is insufficient. Re-enrichment from surviving sources creates new records after recomputing inputs. Do not retain content in error details, model request logs, or cached JSON. Increment generation and access epoch in the purge transaction. Physical SSD erasure is not promised. After successful logical purge and when no reader is active, checkpoint WAL and compact the database; failed compaction leaves logical access blocked and reports cleanup pending.

Entire-store purge requires obtaining the store operation lock, stopping writes, blocking new reads, incrementing access_epoch, closing owned connections, and removing the owned database/mirror/temp files. Validate the store marker before deletion and refuse to delete an arbitrary directory. The resulting `Memory` instance is closed; `close()` remains idempotent. Do not delete the developer's source repository or exports outside the store.

For individual Git blobs or secret-bearing commit history, v1 requires rebuilding/purging the entire store. Rewriting an arbitrary subset of Git history is outside scope. An exported copy already moved elsewhere cannot be recalled; export metadata states this limitation.

## 7. Owned Git object survival

Use an owned bare repository without alternates or hard-link dependence on the source. Disable automatic garbage collection. Before replacing a fetched ref tip, retain a pin for the old tip when needed by stored history/evidence. Pin fetched PR/review commit objects used by evidence. Never assume an unreachable blob will survive a future source-repository GC.

The metadata-to-object invariant is: every available Git evidence locator resolves in `repository.git`, or has a recorded `MISSING_GIT_OBJECT` gap. Export validates this invariant. No current source checkout is needed to expand exported evidence.

## 8. Export and recovery

Take an explicit offline snapshot with no writer active. Use SQLite's backup API, not copying a live database file without its WAL. Copy the owned bare object store without alternates, or produce a Git bundle containing all pinned refs. V1 chooses a copied bare store for simpler deterministic restoration. Do not include runtime credentials, machine-specific source bindings, temp responses, or raw model prompts in the export. Clear the four nullable local-path/common-directory columns in the destination database, omit them from `store.json`, and replace copied Git config with minimal local bare configuration before computing hashes. The exported `store.json` marks the archive read-only. Private repository content, author data, and secrets already committed to history remain source evidence; v1 does not promise automatic secret detection or redaction of full Git exports. Export is an explicit local operation, not an upload.

Scrub machine bindings with `secure_delete=ON` on the destination, then vacuum/checkpoint and close it before hashing so the export does not contain old local binding strings in free pages or WAL files. Copy only required Git object/ref files and minimal sanitized config; exclude logs, hooks, `FETCH_HEAD`, temp files, and any alternates. The object ownership/pinning invariant makes those operational files unnecessary. A source file can itself contain an absolute path: preserve source bytes, and distinguish them from the library's machine-binding metadata when auditing an export.

Reject export destinations within the owned store or indexed source checkout, including realpath/symlink aliases. Create a new temporary sibling of the destination on the same filesystem. Generate `manifest.json` with schema version, repository identity, active revision, generation, fingerprints, counts, gap summaries, and SHA-256 hashes of exported files. The manifest does not hash itself; `ExportReport.manifest_sha256` hashes its finished bytes. Write `OVERVIEW.md` from the published overview; if enrichment is missing, render a deterministic structural overview and gaps. Atomically publish to the requested non-existing destination while refusing replacement if another process created it. Never merge into an existing export.

On import/open, validate manifest hashes and schema compatibility before serving data. Do not execute anything from the archive. An export without Git objects supports metadata reads and returns an explicit gap for code expansion.

Recovery checks: incomplete temporary exports are ignored; a committed database transaction is authoritative; orphaned temp files can be removed after validating they are owned by this store; partially fetched Git objects do not imply completed SQL checkpoints. Run a lightweight consistency check at mutation startup, and expose a deliberate full `doctor` maintenance command later only if needed—do not add a second management interface in v1 merely for hypothetical problems.

## 9. Required storage tests

- Apply the migration to a new in-memory SQLite database; verify foreign keys and FTS insertion/update/deletion.
- Insert one blob reused at 100 paths and across 1,000 commits: one extraction/symbol set remains; snapshot path rows reflect only the active tree.
- Edit one source and verify dependent claims, parent summaries, and their FTS documents become stale/removed in the same generation.
- Interrupt before and after a write commit; a reader sees the previous or next generation, never a mixture.
- Source purges invalidate direct ID retrieval, FTS hits, cursors, derived summaries, and cached context responses.
- Export during a held writer is refused/retried predictably; opening the finished export does not require the original checkout.
- Reject modified migration checksums and unsupported future schema versions rather than silently repairing them.

Testing details and fixture IDs are in [validation](05-validation-and-benchmarks.md). The SQL above supplies a concrete starting point; its successful execution alone does not establish application correctness or performance.
