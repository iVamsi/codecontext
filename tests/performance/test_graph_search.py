import statistics
import time
from pathlib import Path

from codecontext import InitRequest, Memory, SyncRequest
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo


def test_kotlin_graph_search_latency(tmp_path: Path) -> None:
    """Measure a 20,200-symbol Kotlin graph without a machine-dependent time assertion."""
    repo = init_fixture_repo(tmp_path / "repo")
    files = {
        f"src/Module{i}.kt": (
            f"class Module{i} {{\n"
            + "".join(f" fun method{j}() {{}}\n" for j in range(100))
            + "}\n"
        ).encode()
        for i in range(200)
    }
    commit_fixture_spec(repo, GitCommitSpec(message="Kotlin graph benchmark", files=files))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        for query, expected in [
            ({"name_pattern": "^method50$", "limit": 20}, 200),
            ({"name_pattern": "^absent$"}, 0),
            ({"max_degree": 0, "limit": 20}, 20200),
        ]:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                result = mem.search_graph(query)
                times.append((time.perf_counter() - start) * 1000)
                assert result.total_matched == expected
                assert len(result.nodes) == min(expected, 20)
            print(f"{query}: median={statistics.median(times):.2f} ms, max={max(times):.2f} ms")
