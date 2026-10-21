import json
import time
from pathlib import Path

from codecontext import InitRequest, Memory, SyncRequest
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_populated_sync_refresh(tmp_path: Path) -> None:
    """Print local timings; assert reused parses rather than machine-dependent speed."""
    root = tmp_path
    repo = init_fixture_repo(root / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Base",
            files={f"f{i}.py": f"def call{i}(): pass\n".encode() for i in range(120)},
        ),
    )
    for i in range(12):
        commit_fixture_spec(
            repo,
            GitCommitSpec(
                message=f"History {i}",
                files={"history.py": f"def history(): return {i}\n".encode()},
            ),
        )
    with Memory.initialize(root / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True, max_seconds=120))
        results = {}
        for case in ["unchanged", "edit", "rename", "delete"]:
            if case == "edit":
                (repo / "f0.py").write_text("def edited(): call1()\n")
            if case == "rename":
                run_git_fixture(repo, ["mv", "f0.py", "moved.py"])
            if case == "delete":
                (repo / "moved.py").unlink()
            start = time.monotonic()
            report = mem.sync(SyncRequest(offline=True, max_seconds=120))
            results[case] = {
                "seconds": round(time.monotonic() - start, 3),
                "blobs_parsed": report.blobs_parsed,
                "commits_added": report.commits_added,
            }
        print(json.dumps(results, indent=2))
        assert results["unchanged"]["blobs_parsed"] == 0
        assert results["unchanged"]["commits_added"] == 0
        assert results["edit"]["blobs_parsed"] == 1
        assert results["rename"]["blobs_parsed"] == results["delete"]["blobs_parsed"] == 0
        assert mem.search_graph({"name_pattern": "^call1$"}).nodes
        assert not mem.search_graph({"name_pattern": "^edited$"}).nodes
