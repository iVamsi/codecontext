import subprocess
import sys
from pathlib import Path

import pytest

CHILD = r"""
import resource
import sys
from pathlib import Path

from codecontext import InitRequest, Memory, SyncRequest, syntax
from codecontext.deadline import SyncDeadlineExceeded
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
target, read_number, mode, root = sys.argv[1:]
read_number = int(read_number)
source = (
    b"class Client\nclass Example { val client: Client? = null }\n"
    if target == "property_declaration"
    else b"class Client { fun send() {} }\n"
         b"fun run() { val client: Client? = null; client?.send() }\n"
)
control = syntax.parse_blob(source, "kotlin")
assert control.state == "complete", control.diagnostics
if target == "property_declaration":
    assert any(o.kind == "property_type" and o.details == {"name": "client", "type": "Client"}
               for o in control.occurrences)
elif target == "call_expression":
    assert any(o.kind == "call" and o.details.get("receiver_type") == "Client"
               for o in control.occurrences)

original = syntax.check_deadline
text_reads = 0

def expire_during_extraction():
    global text_reads
    caller = sys._getframe(1)
    if target == "read_source" and caller.f_code.co_name == "read_source":
        raise SyncDeadlineExceeded
    frame = caller
    while frame is not None:
        if frame.f_code.co_name == "walk_kotlin_java":
            if frame.f_locals["node"].type == target:
                if caller.f_code.co_name == "read_source":
                    text_reads += 1
                    if text_reads < read_number:
                        return
                raise SyncDeadlineExceeded
            break
        frame = frame.f_back
    original()

if mode == "parse":
    syntax.check_deadline = expire_during_extraction
    try:
        syntax.parse_blob(source, "kotlin")
    except SyncDeadlineExceeded:
        print("cancelled")
    else:
        raise AssertionError("Traversal must still check the deadline")
else:
    root = Path(root)
    repo = init_fixture_repo(root / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Kotlin", files={"Example.kt": source}))
    with Memory.initialize(root / "store", InitRequest(repo_path=repo)) as mem:
        syntax.check_deadline = expire_during_extraction
        partial = mem.sync(SyncRequest(offline=True, max_seconds=120))
        assert partial.stop_reason == "time_budget"
        assert partial.state == "partial"
        assert partial.coverage.structure.state == "partial"
        assert any(g.code == "STRUCTURE_PENDING" for g in partial.gaps)
        syntax.check_deadline = original
        resumed = mem.sync(SyncRequest(offline=True, max_seconds=120))
        assert resumed.coverage.structure.state == "complete"
        assert not any(g.code == "STRUCTURE_PENDING" for g in resumed.gaps)
        assert mem.search_graph({"name_pattern": "^Client$"}).nodes
        print("resumed")
"""


@pytest.mark.parametrize(
    "target,read_number,mode",
    [
        ("property_declaration", 1, "parse"),
        ("property_declaration", 2, "parse"),
        ("call_expression", 1, "parse"),
        ("call_expression", 2, "parse"),
        ("read_source", 1, "parse"),
        ("property_declaration", 1, "sync"),
        ("call_expression", 1, "sync"),
    ],
)
def test_kotlin_deadline_is_safe(tmp_path: Path, target: str, read_number: int, mode: str) -> None:
    # Isolate native crashes so a regression fails this test instead of killing pytest.
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            CHILD,
            target,
            str(read_number),
            mode,
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ("cancelled" if mode == "parse" else "resumed")
