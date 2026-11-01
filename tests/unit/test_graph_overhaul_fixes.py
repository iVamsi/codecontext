from pathlib import Path

import pytest

from codecontext import InitRequest, Memory, SyncRequest
from codecontext.graph import DetectChangesRequest, GetArchitectureRequest, is_test_path
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo

MOD_V0 = (
    b"def keep(a):\n"
    b"    return a\n\n"
    b"def sig(a):\n"
    b"    return a\n\n"
    b"def body(a):\n"
    b"    return a\n\n"
    b"def gone(a):\n"
    b"    return a\n"
)
MOD_V1 = (
    b"# shifts every line down without touching keep\n"
    b"def keep(a):\n"
    b"    return a\n\n"
    b"def sig(a, b):\n"
    b"    return a\n\n"
    b"def body(a):\n"
    b"    return a + 1\n\n"
    b"def fresh():\n"
    b"    return 0\n"
)


def _build_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = init_fixture_repo(tmp_path / "repo")
    c0 = commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="C0",
            files={
                "src/mod.py": MOD_V0,
                "src/old.py": b"def legacy():\n    return 1\n",
                "src/attestation.py": b"from src.mod import body\n\ndef attest():\n    return body(1)\n",
                "tests/test_mod.py": b"from src.mod import body\n\ndef test_body():\n    assert body(1)\n",
            },
        ),
    )
    commit_fixture_spec(
        repo,
        GitCommitSpec(message="C1", files={"src/mod.py": MOD_V1, "src/old.py": None}),
    )
    return repo, c0


def test_detect_changes_classifies_hunks_and_finds_deleted_symbols(tmp_path: Path) -> None:
    repo, c0 = _build_repo(tmp_path)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        res = mem.detect_changes(DetectChangesRequest(since=c0, depth=2))

    assert res.changed_files == ["src/mod.py", "src/old.py"]
    assert res.classifications == {
        "sig": "signature_modified",
        "body": "body_modified",
        "fresh": "added",
        "gone": "deleted",
        "legacy": "deleted",
    }
    assert res.deleted_symbols == ["gone", "legacy"]
    assert [s.name for s in res.directly_modified_symbols] == ["sig", "body", "fresh"]
    assert {s.name for s in res.affected_callers} == {"attest", "test_body"}
    assert res.affected_tests == ["test_body"]


def test_detect_changes_worktree_deletion(tmp_path: Path) -> None:
    repo, _ = _build_repo(tmp_path)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        (repo / "src" / "mod.py").write_bytes(MOD_V1.replace(b"\ndef fresh():\n    return 0\n", b""))
        mem.configure_workspace(repo)
        res = mem.detect_changes(DetectChangesRequest())

    assert res.changed_files == ["src/mod.py"]
    assert res.classifications == {"fresh": "deleted"}
    assert res.deleted_symbols == ["fresh"]
    assert res.directly_modified_symbols == []


def test_get_architecture_uses_louvain_communities_deterministically(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="C0",
            files={
                "app/api.py": b"from core.db import store_row\n\ndef serve():\n    return store_row(1)\n",
                "core/db.py": b"def store_row(x):\n    return x\n",
                "ui/view.py": b"from ui.widgets import draw_box\n\ndef render():\n    return draw_box()\n",
                "ui/widgets.py": b"def draw_box():\n    return 1\n",
                "tools/util.py": b"def lonely():\n    return 0\n",
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        first = mem.get_architecture(GetArchitectureRequest())
        second = mem.get_architecture(GetArchitectureRequest())

    assert first == second
    by_id = {c.id: c for c in first.clusters}
    assert set(by_id) == {"community:.", "community:ui", "tools"}
    assert by_id["community:."].file_count == 2
    assert by_id["community:."].scope_kind == "community"
    assert by_id["community:."].internal_calls == 1
    assert by_id["community:ui"].path == "ui"
    assert by_id["tools"].scope_kind == "directory"
    assert first.cycles == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/attestation.py", False),
        ("src/contest/latest.py", False),
        ("testing_utils.py", False),
        ("tests/helpers.py", True),
        ("pkg/test/conf.py", True),
        ("test_mod.py", True),
        ("pkg/mod_test.py", True),
        ("cmd/server_test.go", True),
        ("web/app.test.ts", True),
        ("web/app.spec.ts", True),
        ("web/__tests__/app.js", True),
        ("src/main/FooTest.java", True),
        ("src/main/FooTest.kt", True),
    ],
)
def test_is_test_path_matches_segments_not_substrings(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected
