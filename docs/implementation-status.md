# Implementation status
Active task: none (T01–T23 code and deterministic checks complete)
Last completed task: T23
Prepared repository/revision: deterministic fixture repositories verified in test suite

| Task | State | Changed files | Validation evidence | Remaining limits |
| --- | --- | --- | --- | --- |
| T01 | verified | pyproject.toml, uv.lock, README.md, src/codecontext/__init__.py, src/codecontext/__main__.py, src/codecontext/cli.py, src/codecontext/_deps.py, src/codecontext/errors.py, tests/unit/test_t01_smoke.py | `pytest tests/unit/test_t01_smoke.py -q` (6 passed), `ruff check` (passed) | None |
| T02 | verified | src/codecontext/__init__.py, src/codecontext/identity.py, src/codecontext/models.py, tests/fixtures/builders.py, tests/unit/test_t02_models_identity.py | `pytest tests/unit/test_t02_models_identity.py -q` (4 passed), `ruff check` (passed); verified F01/F03/F09 builders, U01–U03, SHA-1/SHA-256 object IDs | None |
| T03 | verified | src/codecontext/migrations/001_initial.sql, src/codecontext/store.py, tests/unit/test_t03_store.py | `pytest tests/unit/test_t03_store.py -q` (5 passed), `ruff check` (passed); verified FTS5 triggers, foreign keys, U14/U15, I09 generation isolation | None |
| T04 | verified | src/codecontext/git.py, tests/unit/test_t04_git.py | `pytest tests/unit/test_t04_git.py -q` (3 passed), `ruff check` (passed); verified F03/F09/I01/I05 owned bare mirror and branch switching | None |
| T05 | verified | src/codecontext/syntax.py, src/codecontext/queries/*.scm, tests/unit/test_t05_syntax.py | `pytest tests/unit/test_t05_syntax.py -q` (3 passed), `ruff check` (passed); verified U02/U04/U05 and small F12 blob deduplication | None |
| T06 | verified | src/codecontext/lineage.py, tests/unit/test_t06_lineage.py | `pytest tests/unit/test_t06_lineage.py -q` (2 passed), `ruff check` (passed); verified U06 and non-GitHub I04 original-to-target mappings | None |
| T07 | verified | src/codecontext/github.py, tests/unit/test_t07_github.py | `pytest tests/unit/test_t07_github.py -q` (2 passed), `ruff check` (passed); verified focused GitHub acquisition, page-by-page checkpoint resume, principal binding | None |
| T08 | verified | src/codecontext/retrieval.py, tests/unit/test_t08_search_get.py | `pytest tests/unit/test_t08_search_get.py -q` (1 passed), `ruff check` (passed); verified U11/U13/U14, FTS5 literal safety, RRF ranking, `get` expansion | None |
| T09 | verified | src/codecontext/workspace.py, src/codecontext/retrieval.py, tests/unit/test_t09_workspace_context.py | `pytest tests/unit/test_t09_workspace_context.py -q` (2 passed), `ruff check` (passed); verified F07/U10/U11/I07/I08 workspace divergence and read-only sentinels | None |
| T10 | verified | src/codecontext/enrichment.py, examples/fixture_adapter.py, tests/unit/test_t10_enrichment.py | `pytest tests/unit/test_t10_enrichment.py -q` (3 passed), `ruff check` (passed); verified U08–U10/U12/U16 budgeted extraction, quote validation, repair, and timeout reservations | None |
| T11 | blocked_external | src/codecontext/sync.py, src/codecontext/archive.py, src/codecontext/memory.py, src/codecontext/cli.py, src/codecontext/mcp_server.py, examples/agent-instructions.md, examples/mcp-config.json, tests/integration/test_t11_sdk_cli_mcp.py | `implementation_complete=true`; `pytest tests/integration/test_t11_sdk_cli_mcp.py -q` (2 passed) verified all 9 workflow steps over real subprocess stdio MCP client and SDK/CLI/MCP equivalence | External live agent-host pilot transcript pending |
| T12 | verified | src/codecontext/git.py, src/codecontext/sync.py, tests/integration/test_t12_t15_ingestion_lineage.py | `pytest tests/integration/test_t12_t15_ingestion_lineage.py -q` (passed); verified exhaustive ref enumeration, force-push retention, P01 storage invariant | None |
| T13 | verified | src/codecontext/github.py, src/codecontext/sync.py, tests/integration/test_t12_t15_ingestion_lineage.py | `pytest tests/integration/test_t12_t15_ingestion_lineage.py -q` (passed); verified full `/issues?state=all` enumeration, PR normalization, allowlisted external issues | None |
| T14 | verified | src/codecontext/github.py, tests/integration/test_t12_t15_ingestion_lineage.py | `pytest tests/integration/test_t12_t15_ingestion_lineage.py -q` (passed); verified incremental comment update streams, `reconcile=True` missing-source detection, cutoff FTS retention | None |
| T15 | verified | src/codecontext/lineage.py, tests/integration/test_t12_t15_ingestion_lineage.py | `pytest tests/integration/test_t12_t15_ingestion_lineage.py -q` (passed); verified F02/F05 original review anchors, squash candidate mapping, scoped ADR corroboration | None |
| T16 | verified | src/codecontext/enrichment.py, tests/unit/test_t10_enrichment.py, tests/integration/test_t16_t20_enrichment_archive.py | `pytest -q` (passed); verified multi-segment extraction, cache reuse, repair, and uncertain usage accounting | None |
| T17 | verified | src/codecontext/retrieval.py, tests/integration/test_t12_t15_ingestion_lineage.py, tests/integration/test_t16_t20_enrichment_archive.py | `pytest -q` (passed); verified F03/F06/U07/I05 branch ancestry and `knowledge_cutoff` eligibility across search, get, context, claims, and summaries | None |
| T18 | verified | src/codecontext/store.py, src/codecontext/archive.py, tests/integration/test_t16_t20_enrichment_archive.py | `pytest tests/integration/test_t16_t20_enrichment_archive.py -q` (passed); verified recursive reverse-dependency invalidation, I10 purge race barrier, source purge | None |
| T19 | verified | src/codecontext/enrichment.py, src/codecontext/retrieval.py, tests/integration/test_t16_t20_enrichment_archive.py | `pytest tests/integration/test_t16_t20_enrichment_archive.py -q` (passed); verified deterministic package/directory/root_files/repository grouping and cited summaries | None |
| T20 | verified | src/codecontext/archive.py, src/codecontext/memory.py, tests/integration/test_t16_t20_enrichment_archive.py | `pytest tests/integration/test_t16_t20_enrichment_archive.py -q` (passed); verified I10/I12 portable export, credential/path scrubbing, manifest verification, entire-store purge | None |
| T21 | blocked_external | evals/run.py, evals/score.py, evals/corpus.lock.json, evals/questions.jsonl, evals/rubric.schema.json, tests/unit/test_t21_evals.py | `implementation_complete=true`; `pytest tests/unit/test_t21_evals.py -q` (passed) verified 120-question corpus split, E05/E06 scoring/Wilson/bootstrap, and E08 refusal boundaries | External human review of 120 gold answers and live paid-model evaluation pending |
| T22 | blocked_external | tests/performance/test_p01_p08_benchmarks.py | `implementation_complete=true`; `pytest tests/performance/test_p01_p08_benchmarks.py -q` (passed) verified P01 storage-growth invariant, P02/P03 warm latency targets, P05 progressive collection, P06 incremental cost, P08 zero read spend | Full 100k-commit / 1M-LOC reference-machine hardware run opt-in |
| T23 | verified | README.md, .github/workflows/ci.yml, pyproject.toml, docs/implementation-status.md | Full `pytest -q` (41 passed), `ruff check src tests examples evals` (passed), wheel/sdist build verified | External release gates listed below remain tracked |

## Decisions changed from the specification
None.

## External gates
- Real MCP-client pilot: pending (automated subprocess stdio MCP client passes all 9 steps in `test_t11_sdk_cli_mcp.py`; live external agent-host session pending)
- Live model evaluation budget/adapter: pending
- Human-reviewed gold answers: pending (`evals/questions.jsonl` generated with `review_status="draft_pending_human_review"`)
- Reference-machine load benchmark: pending (deterministic P01–P08 suite in `tests/performance/test_p01_p08_benchmarks.py` passes)

## Next handoff
All T01–T23 implementation tasks and deterministic unit/integration/performance checks are complete and passing. Run live external agent pilot, human gold review, and paid model evaluation when credentials and reviewers are available.
