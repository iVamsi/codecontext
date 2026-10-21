import json
from pathlib import Path

import pytest

from codecontext import InitRequest, Memory, SyncRequest, git
from codecontext.deadline import SyncDeadlineExceeded
from codecontext.identity import make_ev_id
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_unchanged_sync_does_not_repin_shared_tips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo, GitCommitSpec(message="Base", files={"main.py": b"def current(): pass\n"})
    )
    for i in range(5):
        run_git_fixture(repo, ["tag", f"tag{i}"])
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        original = git.run_safe_git
        writes = []

        def record_writes(args, **kwargs):
            if args[0] == "update-ref":
                writes.append((args, kwargs.get("stdin_bytes")))
            return original(args, **kwargs)

        monkeypatch.setattr(git, "run_safe_git", record_writes)
        for _ in range(2):
            mem.sync(SyncRequest(offline=True))
            assert not writes


@pytest.mark.parametrize("suffix", ["py", "txt"])
def test_shared_chunk_rename_repeat_and_delete(tmp_path: Path, suffix: str) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    content = b"# uniquechunkmark2026\n"
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Shared content",
            files={
                f"a.{suffix}": content,
                f"b.{suffix}": content,
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        run_git_fixture(repo, ["mv", f"a.{suffix}", f"c.{suffix}"])
        for expected in [
            {f"b.{suffix}", f"c.{suffix}"},
            {f"b.{suffix}", f"c.{suffix}"},
            {f"b.{suffix}"},
        ]:
            if len(expected) == 1:
                (repo / f"c.{suffix}").unlink()
            mem.sync(SyncRequest(offline=True))
            hits = mem.search({"query": "uniquechunkmark2026", "kind": "code"}).hits
            assert sorted(h.path.path for h in hits) == sorted(expected)
            with mem._store.connect_read() as conn:
                assert conn.execute(
                    "SELECT COUNT(*) FROM search_documents WHERE record_kind='code'"
                ).fetchone()[0] == len(expected)


def test_retention_resumes_missing_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Many objects",
            files={f"file{i}.txt": f"content {i}\n".encode() for i in range(130)},
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        mirror = mem._mirror
        with mem._store.connect_read() as conn:
            oids = {r[0] for r in conn.execute("SELECT blob_oid FROM snapshot_files")}
        original = git.run_safe_git
        completed = []

        def interrupt_after_batch(args, **kwargs):
            result = original(args, **kwargs)
            if args[0] == "update-ref":
                completed.extend(
                    line.split()[2] for line in kwargs["stdin_bytes"].decode().splitlines()
                )
                raise SyncDeadlineExceeded
            return result

        monkeypatch.setattr(git, "run_safe_git", interrupt_after_batch)
        with pytest.raises(SyncDeadlineExceeded):
            mirror.pin_retained_oids(oids)
        assert len(completed) == 128
        resumed = []

        def record_resume(args, **kwargs):
            if args[0] == "update-ref":
                resumed.extend(
                    line.split()[2] for line in kwargs["stdin_bytes"].decode().splitlines()
                )
            return original(args, **kwargs)

        monkeypatch.setattr(git, "run_safe_git", record_resume)
        mirror.pin_retained_oids(oids)
        assert set(resumed) == oids - set(completed)
        assert len(resumed) == 2
        resumed.clear()
        mirror.pin_retained_oids(oids)
        assert not resumed

        # A retained ref with the wrong target still needs correction.
        oid, wrong_oid = sorted(oids)[:2]
        original(
            ["update-ref", f"refs/codecontext/retained/{oid}", wrong_oid], git_dir=mirror.git_dir
        )
        mirror.pin_retained_oid(oid)
        assert resumed == [oid]
        target = original(["rev-parse", f"refs/codecontext/retained/{oid}"], git_dir=mirror.git_dir)
        assert target.stdout.decode().strip() == oid


def test_many_unchanged_refs_allow_source_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    oid = commit_fixture_spec(
        repo, GitCommitSpec(message="Base", files={"main.py": b"def old(): pass\n"})
    )
    git.run_safe_git(
        ["update-ref", "--stdin"],
        cwd=repo,
        stdin_bytes="".join(f"update refs/tags/tag{i} {oid}\n" for i in range(3161)).encode(),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        initial = mem.sync(SyncRequest(offline=True, max_seconds=120))
        assert initial.coverage.structure.state == "complete", initial
        writes = []
        original = git.run_safe_git

        def record_writes(args, **kwargs):
            if args[0] == "update-ref":
                writes.append(kwargs.get("stdin_bytes"))
            return original(args, **kwargs)

        monkeypatch.setattr(git, "run_safe_git", record_writes)
        (repo / "main.py").write_text("def updated(): pass\n")
        refreshed = mem.sync(SyncRequest(offline=True, max_seconds=120))
        assert refreshed.coverage.structure.state == "complete", refreshed
        assert mem.search_graph({"name_pattern": "^updated$"}).nodes
        assert len(writes) == 1  # Only the new workspace snapshot needs retaining.
        writes.clear()
        unchanged = mem.sync(SyncRequest(offline=True, max_seconds=120))
        assert unchanged.coverage.structure.state == "complete", unchanged
        assert not writes


def test_sync_repairs_existing_duplicate_chunks(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Shared",
            files={
                "a.py": b"# uniquechunkmark2026\n",
                "b.py": b"# uniquechunkmark2026\n",
                "symbols.py": b"def kept(): pass\n",
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        run_git_fixture(repo, ["mv", "a.py", "c.py"])
        mem.sync(SyncRequest(offline=True))
        with mem._store.write_transaction() as conn:
            old = conn.execute(
                "SELECT * FROM search_documents WHERE record_kind = 'code' AND json_extract(metadata_json, '$.path') = 'b.py'"
            ).fetchone()
            evidence = conn.execute(
                "SELECT * FROM evidence WHERE id = ?", (old["record_id"],)
            ).fetchone()
            meta = conn.execute("SELECT repository_id, active_revision FROM store_meta").fetchone()
            anchor = json.loads(evidence["anchor_json"])
            anchor["revision"] = meta["active_revision"]
            new_id = make_ev_id(
                repository_id=meta["repository_id"],
                kind="code",
                source_version_id=None,
                git_oid=evidence["git_oid"],
                anchor=anchor,
                content_sha256=evidence["content_sha256"],
            )
            metadata = json.loads(old["metadata_json"])
            metadata.update(evidence_id=new_id, revision=meta["active_revision"])
            conn.execute(
                """INSERT INTO evidence (id, kind, git_oid, anchor_json, content_sha256, state)
                            VALUES (?, 'code', ?, ?, ?, 'available')""",
                (new_id, evidence["git_oid"], json.dumps(anchor), evidence["content_sha256"]),
            )
            conn.execute(
                """INSERT INTO search_documents (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json)
                            VALUES ('code', ?, ?, ?, ?, ?, ?)""",
                (
                    new_id,
                    old["chunk_ordinal"],
                    old["title"],
                    old["identifiers"],
                    old["body"],
                    json.dumps(metadata),
                ),
            )
            conn.execute("DELETE FROM checkpoints WHERE stream_key = 'syntax:chunk_dedup:v1'")
        assert len(mem.search({"query": "uniquechunkmark2026", "kind": "code"}).hits) == 3
        mem.sync(SyncRequest(offline=True))
        hits = mem.search({"query": "uniquechunkmark2026", "kind": "code"}).hits
        assert sorted(h.path.path for h in hits) == ["b.py", "c.py"]
        assert mem.search({"query": "kept", "kind": "code"}).hits
        with mem._store.connect_read() as conn:
            assert (
                conn.execute(
                    "SELECT record_id FROM search_documents WHERE record_kind = 'code' AND json_extract(metadata_json, '$.path') = 'b.py'"
                ).fetchone()[0]
                == new_id
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM evidence WHERE id IN (?, ?)", (old["record_id"], new_id)
                ).fetchone()[0]
                == 2
            )


def test_shared_symbol_paths_still_refresh(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Shared symbol",
            files={
                "a.py": b"def shared(): pass\n",
                "b.py": b"def shared(): pass\n",
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        run_git_fixture(repo, ["mv", "a.py", "c.py"])
        for expected in [["b.py", "c.py"], ["b.py"]]:
            if len(expected) == 1:
                (repo / "c.py").unlink()
            mem.sync(SyncRequest(offline=True))
            with mem._store.connect_read() as conn:
                rows = conn.execute(
                    "SELECT metadata_json FROM search_documents WHERE record_kind = 'code'"
                ).fetchall()
                assert len(rows) == 1
                assert sorted(json.loads(rows[0][0])["paths"]) == expected
