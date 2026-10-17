from pathlib import Path

from codecontext.git import GitMirror
from codecontext.models import InitRequest
from codecontext.store import Store
from codecontext.syntax import (
    MAX_PARSE_BYTES,
    chunk_utf8_bytes,
    index_active_snapshot_files,
    index_blob_into_store,
    parse_blob,
)
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f09_samples,
    commit_fixture_spec,
    init_fixture_repo,
)


def test_u04_extract_all_supported_languages_and_fingerprint_caching(tmp_path: Path) -> None:
    """U04: Extract every supported F09 language twice, then alter extractor fingerprint.
    Verify expected syntax records exist, identical cache keys reuse results, and changed fingerprints invalidate reuse.
    """
    f09 = build_f09_samples()

    # 1. Python
    py_res = parse_blob(f09.python_source, "python")
    assert py_res.state == "complete"
    py_qnames = {s.qualified_name: s for s in py_res.symbols}
    assert {"DEFAULT_LIMIT", "helper", "Processor", "Processor.compute", "Processor.compute.inner"} <= set(
        py_qnames.keys()
    )
    # Decorator included in display range of Processor.compute while body range is inside
    comp_sym = py_qnames["Processor.compute"]
    assert f09.python_source[comp_sym.byte_start : comp_sym.byte_end].startswith(b"@staticmethod")
    assert comp_sym.body_byte_start is not None and comp_sym.body_byte_start > comp_sym.byte_start
    # Docstrings extracted as syntax_occurrences
    doc_occs = [o for o in py_res.occurrences if o.kind == "docstring"]
    assert len(doc_occs) == 3

    # 2. JavaScript
    js_res = parse_blob(f09.javascript_source, "javascript")
    assert js_res.state == "complete"
    js_qnames = {s.qualified_name for s in js_res.symbols}
    assert {"runTask", "arrowFn", "Service", "Service.execute"} <= js_qnames

    # 3. TypeScript
    ts_res = parse_blob(f09.typescript_source, "typescript")
    assert ts_res.state == "complete"
    ts_qnames = {s.qualified_name: s.kind for s in ts_res.symbols}
    assert ts_qnames.get("Runner") == "interface"
    assert ts_qnames.get("Mode") == "type_alias"
    assert ts_qnames.get("StatusCode") == "enum"
    assert ts_qnames.get("DefaultRunner") == "class"
    assert ts_qnames.get("DefaultRunner.run") == "method"

    # 4. JSX (using javascript grammar)
    jsx_res = parse_blob(f09.jsx_source, "javascript")
    assert jsx_res.state == "complete"
    assert "Card" in {s.qualified_name for s in jsx_res.symbols}
    jsx_comps = [o.details.get("component") for o in jsx_res.occurrences if o.kind == "jsx_component"]
    assert "Badge" in jsx_comps

    # 5. TSX
    tsx_res = parse_blob(f09.tsx_source, "tsx")
    assert tsx_res.state == "complete"
    assert {"Props", "ActionButton"} <= {s.qualified_name for s in tsx_res.symbols}

    # 6. Kotlin and Java
    kt_source = (
        b"package com.example\n\n"
        b"class CountryRepository(private val api: Api) {\n"
        b"    suspend fun loadCountries(force: Boolean): List<String> {\n"
        b"        return fetchRemote(force)\n"
        b"    }\n\n"
        b"    private fun fetchRemote(force: Boolean): List<String> = emptyList()\n"
        b"}\n"
    )
    kt_res = parse_blob(kt_source, "kotlin")
    assert kt_res.state == "complete"
    assert {"CountryRepository", "CountryRepository.loadCountries", "CountryRepository.fetchRemote"} <= {
        s.qualified_name for s in kt_res.symbols
    }

    java_source = (
        b"package com.example;\n\n"
        b"public class CountryService {\n"
        b"    public int countCountries() {\n"
        b"        return helperCount();\n"
        b"    }\n"
        b"    private int helperCount() { return 42; }\n"
        b"}\n"
    )
    java_res = parse_blob(java_source, "java")
    assert java_res.state == "complete"
    assert {"CountryService", "CountryService.countCountries", "CountryService.helperCount"} <= {
        s.qualified_name for s in java_res.symbols
    }

    # Test store-level caching and fingerprint invalidation
    repo_dir = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(message="Initial", files={"app.py": f09.python_source}),
    )
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo_dir))
    blob_oid = "a" * 40
    with store.write_transaction() as conn:
        r1, hit1 = index_blob_into_store(
            conn,
            repository_id=store.config.repository_id,
            blob_oid=blob_oid,
            blob_bytes=f09.python_source,
            language="python",
        )
        assert hit1 is False
        r2, hit2 = index_blob_into_store(
            conn,
            repository_id=store.config.repository_id,
            blob_oid=blob_oid,
            blob_bytes=f09.python_source,
            language="python",
        )
        assert hit2 is True
        assert [s.id for s in r1.symbols] == [s.id for s in r2.symbols]

    # Altering fingerprint creates distinct extraction & symbol IDs
    r_alt = parse_blob(f09.python_source, "python", extractor_fingerprint="custom_fp_v2", blob_oid=blob_oid)
    assert r_alt.extraction_id != r1.extraction_id
    assert {s.id for s in r_alt.symbols}.isdisjoint({s.id for s in r1.symbols})
    store.close()


def test_u05_malformed_unsupported_oversized_binary_and_lfs(tmp_path: Path) -> None:
    """U05: Parse malformed, unsupported, and oversized content.
    Usable supported fragments remain labeled; missing structure produces gaps, not fabricated graph edges.
    """
    f09 = build_f09_samples()

    # Malformed Python retains valid definitions before and after broken syntax
    mal_res = parse_blob(f09.malformed_python, "python")
    assert mal_res.state == "partial"
    assert len(mal_res.diagnostics) >= 1
    mal_names = {s.qualified_name for s in mal_res.symbols}
    assert "valid_before" in mal_names
    assert "valid_after" in mal_names
    assert "broken_func" not in mal_names

    # Oversized blob (> 2 MiB)
    over_res = parse_blob(b"x = 1\n" * (MAX_PARSE_BYTES // 6 + 10), "python")
    assert over_res.state == "oversized"
    assert over_res.symbols == []

    # Binary content
    bin_res = parse_blob(b"\x89PNG\r\n\x1a\n\x00\x00\x00", "python")
    assert bin_res.state == "binary"
    assert bin_res.symbols == []

    # Non-UTF-8 content
    bad_utf8 = parse_blob(b"def f():\n    return '\xff\xfe'\n", "python")
    assert bad_utf8.state == "error"
    assert bad_utf8.diagnostics[0]["code"] == "TEXT_DECODE_FAILED"

    # End-to-end snapshot indexing with binary, LFS pointer, non-UTF-8, malformed, and markdown doc
    repo_dir = init_fixture_repo(tmp_path / "u05_repo")
    lfs_bytes = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
        b"size 12345\n"
    )
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Mixed capability files",
            files={
                "src/broken.py": f09.malformed_python,
                "assets/image.bin": b"\x00\x01\x02\x03",
                "assets/model.weights": lfs_bytes,
                "legacy/latin1.py": b"# \xff\xfe invalid utf8\n",
                "docs/guide.md": b"# Guide\n\nDetailed documentation for retry behavior.\n",
            },
        ),
    )
    store = Store.initialize(tmp_path / "u05_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")
    parsed_cnt, hits_cnt = index_active_snapshot_files(store, mirror)
    assert parsed_cnt == 1 and hits_cnt == 0

    st = store.status()
    gap_codes = {g.code for g in st.gaps}
    assert {"PARSE_ERROR", "BINARY_CONTENT", "LFS_CONTENT_NOT_FETCHED", "TEXT_DECODE_FAILED"} <= gap_codes
    store.close()


def test_f12_small_blob_deduplication_across_commits_and_paths(tmp_path: Path) -> None:
    """Small F12 storage invariant: unchanged blobs across commits and shared paths parse once;
    symbol rows equal unique parsed blobs * symbols per blob, never symbols * commits.
    """
    repo_dir = init_fixture_repo(tmp_path / "f12_small")
    shared_mod = (
        b"def alpha(x: int) -> int:\n    return x + 1\n\n"
        b"def beta(y: int) -> int:\n    return y * 2\n"
    )
    files: dict[str | bytes, bytes | None] = {f"pkg/m{i}.py": shared_mod for i in range(10)}
    commit_fixture_spec(repo_dir, GitCommitSpec(message="Initial 10 identical modules", files=files))

    # Create 25 commits changing only m0.py while m1..m9 stay unchanged
    for step in range(1, 26):
        commit_fixture_spec(
            repo_dir,
            GitCommitSpec(
                message=f"Edit m0 step {step}",
                files={
                    "pkg/m0.py": (
                        f"def alpha(x: int) -> int:\n    return x + {step}\n\n"
                        f"def beta(y: int) -> int:\n    return y * 2\n"
                    ).encode()
                },
            ),
        )

    store = Store.initialize(tmp_path / "f12_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")
    parsed_cnt, cache_hits = index_active_snapshot_files(store, mirror)

    # In the active snapshot: m0 has 1 unique blob, m1..m9 all share 1 unique blob -> 2 unique blobs total!
    assert parsed_cnt == 2
    assert cache_hits == 8

    with store.connect_read() as conn:
        ext_count = int(conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0])
        sym_count = int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])
        snap_count = int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0])
        assert ext_count == 2
        assert sym_count == 4  # 2 blobs * 2 functions each
        assert snap_count == 10

    # Also test UTF-8 chunking with overlap on multi-byte characters
    multi_byte_doc = ("Café Ω 日本語 🚀 line\n" * 600).encode("utf-8")
    chunks = chunk_utf8_bytes(multi_byte_doc)
    assert len(chunks) >= 2
    for ch in chunks:
        assert multi_byte_doc[ch.byte_start : ch.byte_end].decode("utf-8") == ch.text

    store.close()
