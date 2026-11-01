import gzip
import random
import struct
import time
from pathlib import Path

import pytest

from codecontext import InitRequest, Memory, SyncRequest, scip_import, syntax
from codecontext.errors import CodeContextError
from codecontext.workspace import (
    MAX_GIT_INDEX_PATH_BYTES,
    parse_git_index,
    read_git_index_stat_cache,
)
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo

# --- SCIP wire helpers -------------------------------------------------------------------------


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(num: int, payload: bytes | int) -> bytes:
    if isinstance(payload, int):
        return _varint(num << 3) + _varint(payload)
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _occurrence(line: int, symbol: str, roles: int) -> bytes:
    packed = b"".join(_varint(v) for v in (line, 4, 10))
    return _field(1, packed) + _field(2, symbol.encode()) + _field(3, roles)


def _scip_index(path: str, occurrences: list[tuple[int, str, int]]) -> bytes:
    doc = _field(1, path.encode()) + b"".join(_field(2, _occurrence(*occ)) for occ in occurrences)
    return _field(3, doc)


def _sym(name: str) -> str:
    return f"scip-python python app 1.0 app/{name}()."


APP_SOURCE = b"def helper():\n    pass\n\n\ndef caller():\n    helper()\n\n\ndef other():\n    pass\n"
DEFINITIONS = [(0, _sym("helper"), 1), (4, _sym("caller"), 1), (8, _sym("other"), 1)]


def _scip_edges(mem: Memory) -> set[tuple[str, str]]:
    with mem._store.connect_read() as conn:
        rows = conn.execute(
            """
            SELECT fs.name AS src, ts.name AS dst FROM relations r
            JOIN symbols fs ON fs.id = r.from_id JOIN symbols ts ON ts.id = r.to_id
            WHERE r.method = 'scip'
            """
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM relations WHERE method = 'scip'").fetchone()[0]
    assert total == len(rows), "dangling scip relations left behind"
    return {(str(r["src"]), str(r["dst"])) for r in rows}


def test_scip_reimport_replaces_edges_and_sync_retires_dangling_ones(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="app", files={"app.py": APP_SOURCE}))
    scip_file = tmp_path / "index.scip"

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))

        scip_file.write_bytes(_scip_index("app.py", [*DEFINITIONS, (5, _sym("helper"), 0)]))
        assert mem.import_scip(scip_file).relations_added == 1
        assert _scip_edges(mem) == {("caller", "helper")}

        scip_file.write_bytes(gzip.compress(_scip_index("app.py", [*DEFINITIONS, (5, _sym("other"), 0)])))
        assert mem.import_scip(scip_file).relations_added == 1
        assert _scip_edges(mem) == {("caller", "other")}

        commit_fixture_spec(repo, GitCommitSpec(message="edit", files={"app.py": APP_SOURCE + b"# changed\n"}))
        mem.sync(SyncRequest(offline=True))
        assert _scip_edges(mem) == set()


def test_scip_rejects_oversized_file_and_gzip_bomb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scip_import, "MAX_SCIP_INDEX_BYTES", 1024)
    with pytest.raises(CodeContextError, match="decompressed payload exceeds"):
        scip_import.decode_scip_index(gzip.compress(b"\0" * 100_000))
    with pytest.raises(CodeContextError, match="file exceeds"):
        scip_import.decode_scip_index(b"\0" * 2048)
    scip_import.decode_scip_index(gzip.compress(_scip_index("a.py", [])))


def _old_find_enclosing(file_syms: list[dict], line_1: int) -> str | None:
    containing = [s for s in file_syms if s["line_start"] <= line_1 <= s["line_end"]]
    if not containing:
        return None
    containing.sort(key=lambda s: (s["line_end"] - s["line_start"], -s["line_start"]))
    return str(containing[0]["id"])


def test_enclosing_symbol_lookup_matches_linear_scan() -> None:
    rng = random.Random(7)
    for _ in range(200):
        file_syms = []
        for i in range(rng.randint(0, 25)):
            start = rng.randint(1, 60)
            file_syms.append({"id": f"s{i}", "line_start": start, "line_end": start + rng.randint(-1, 30)})
        lookup = scip_import._build_enclosing_symbol_lookup(file_syms)
        for line in range(0, 100):
            assert lookup(line) == _old_find_enclosing(file_syms, line)


# --- Cross-file call resolution ----------------------------------------------------------------


def test_module_keys_agree_with_module_spec_matching() -> None:
    specs = ["a", "pkg.a", "./service.js", "com.example.Repo", "pkg", "..x", "", "./", "a/b", "src.pkg.a"]
    paths = ["a.py", "src/pkg/a.py", "src/service.ts", "src/com/example/Repo.kt", "pkg/__init__.py",
             "web/pkg/index.ts", "x.py", "src/a/b.go", "index/index.js", "b.py"]
    for spec in specs:
        for path in paths:
            key = syntax._module_spec_key(spec)
            via_index = bool(key) and key in syntax._path_module_keys(path)
            assert via_index == syntax._module_spec_matches_path(spec, path), (spec, path)


def _call_edges(mem: Memory) -> set[tuple[str, str]]:
    with mem._store.connect_read() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT fs.qualified_name AS src, ff.path AS src_path, ts.qualified_name AS dst, tf.path AS dst_path
            FROM relations r
            JOIN symbols fs ON fs.id = r.from_id JOIN snapshot_files ff ON ff.extraction_id = fs.extraction_id
            JOIN symbols ts ON ts.id = r.to_id JOIN snapshot_files tf ON tf.extraction_id = ts.extraction_id
            WHERE r.method = 'lexical_snapshot' AND r.relation_kind = 'calls_candidate'
            """
        ).fetchall()
    return {(f"{bytes(r['src_path']).decode()}:{r['src']}", f"{bytes(r['dst_path']).decode()}:{r['dst']}")
            for r in rows}


def test_cross_file_resolution_edges_per_strategy(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="calls",
            files={
                "pkg/a.py": b"def save():\n    pass\n\n\nclass Repo:\n    def save(self):\n        pass\n\n"
                            b"    def flush(self):\n        self.save()\n",
                "pkg/b.py": b"def save():\n    pass\n",
                "pkg/c.py": b"from pkg.a import save\n\n\ndef use_import():\n    save()\n",
                "pkg/d.py": b"from pkg.b import *\n\n\ndef use_wild():\n    save()\n",
                "pkg/e.py": b"import pkg.a as mod\n\n\ndef use_alias():\n    mod.save()\n",
                "pkg/f.py": b"def use_typed(repo: Repo):\n    repo.save()\n",
                "tools/h.py": b"def tool():\n    pass\n",
                "lib/h.py": b"def tool():\n    return 1\n",
                "tools/g.py": b"def use_dir():\n    tool()\n\n\ndef use_unique():\n    lonely()\n",
                "misc/l.py": b"def lonely():\n    pass\n",
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        edges = _call_edges(mem)

    cross_file = {(src, dst) for src, dst in edges if src.split(":")[0] != dst.split(":")[0]}
    assert cross_file == {
        # Module matching links every `save` defined in the imported file.
        ("pkg/c.py:use_import", "pkg/a.py:save"),
        ("pkg/c.py:use_import", "pkg/a.py:Repo.save"),
        ("pkg/d.py:use_wild", "pkg/b.py:save"),
        ("pkg/e.py:use_alias", "pkg/a.py:save"),
        ("pkg/e.py:use_alias", "pkg/a.py:Repo.save"),
        ("pkg/f.py:use_typed", "pkg/a.py:Repo.save"),
        ("tools/g.py:use_dir", "tools/h.py:tool"),
        ("tools/g.py:use_unique", "misc/l.py:lonely"),
    }
    assert ("pkg/a.py:Repo.flush", "pkg/a.py:Repo.save") in edges


def test_cross_file_resolution_scales_with_common_names(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    files: dict[str | bytes, bytes | None] = {
        # The comment keeps each blob distinct; identical blobs would share one extraction.
        f"src/m{i}.py": (f"# {i}\ndef run():\n    run()\n" + "".join(f"\n\ndef f{k}():\n    run()\n" for k in range(5))).encode()
        for i in range(2000)
    }
    commit_fixture_spec(repo, GitCommitSpec(message="many runs", files=files))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        with mem._store.write_transaction(bump_generation=False) as conn:
            started = time.perf_counter()
            syntax._resolve_cross_file_calls_and_imports(conn)
            elapsed = time.perf_counter() - started
        edges = _call_edges(mem)

    # Scanning every same-named definition per call took about 2s here; the indexed lookup takes about 0.3s.
    assert elapsed < 1.0
    assert all(src.split(":")[0] == dst.split(":")[0] for src, dst in edges)


# --- Docs ingest -------------------------------------------------------------------------------


def test_doc_restored_unchanged_after_delete_is_searchable_again(tmp_path: Path) -> None:
    doc = b"# Use SQLite\n\nWe store everything in SQLite with WAL mode.\n"
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="adr", files={"docs/adr/0001.md": doc, "a.py": b"x = 1\n"}))

    def history_rows(mem: Memory) -> int:
        with mem._store.connect_read() as conn:
            return int(conn.execute(
                """
                SELECT COUNT(*) FROM search_documents
                WHERE record_kind = 'history' AND record_id IN (SELECT current_version_id FROM sources WHERE kind = 'adr')
                """
            ).fetchone()[0])

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True, ingest_docs=True))
        assert history_rows(mem) > 0
        commit_fixture_spec(repo, GitCommitSpec(message="drop adr", files={"docs/adr/0001.md": None}))
        mem.sync(SyncRequest(offline=True, ingest_docs=True))
        assert history_rows(mem) == 0
        commit_fixture_spec(repo, GitCommitSpec(message="restore adr", files={"docs/adr/0001.md": doc}))
        mem.sync(SyncRequest(offline=True, ingest_docs=True))
        assert history_rows(mem) > 0


# --- Git index v4 ------------------------------------------------------------------------------


def _v4_index(paths_suffixes: list[bytes]) -> bytes:
    entries = b"".join(
        struct.pack(">10I", 0, 0, 0, 0, 0, 0, 0o100644, 0, 0, 0) + b"\x11" * 20 + struct.pack(">H", 0)
        + b"\x00" + suffix + b"\0"
        for suffix in paths_suffixes
    )
    return b"DIRC" + struct.pack(">II", 4, len(paths_suffixes)) + entries + b"\0" * 20


def test_git_index_v4_path_growth_is_capped(tmp_path: Path) -> None:
    ok = _v4_index([b"a" * MAX_GIT_INDEX_PATH_BYTES])
    assert list(parse_git_index(ok, oid_bytes=20)) == [b"a" * MAX_GIT_INDEX_PATH_BYTES]

    # Strip 0 appends to the previous path, so each entry is one byte longer than the last.
    bomb = tmp_path / "index"
    bomb.write_bytes(_v4_index([b"a"] * (MAX_GIT_INDEX_PATH_BYTES + 1)))
    with pytest.raises(ValueError, match="path too long"):
        parse_git_index(bomb.read_bytes(), oid_bytes=20)
    assert read_git_index_stat_cache(bomb, oid_bytes=20) is None
