from pathlib import Path

import pytest

from codecontext import InitRequest, Memory, SyncRequest, deadline, syntax
from codecontext.deadline import SyncDeadlineExceeded, sync_deadline
from codecontext.syntax import parse_blob
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo

EDGE_KINDS = {"imports", "inherits", "implements", "decorates", "references_type"}


def _edges(source: bytes, language: str) -> set[tuple[str | None, str, str]]:
    res = parse_blob(source, language)
    assert res.state == "complete", res.diagnostics
    names = {s.id: s.name for s in res.symbols}
    return {
        (
            names.get(o.enclosing_symbol_id),
            o.kind,
            str(
                o.details.get("base_class")
                or o.details.get("decorator")
                or o.details.get("type_name")
                or o.details.get("imported")
            ),
        )
        for o in res.occurrences
        if o.kind in EDGE_KINDS
    }


@pytest.mark.parametrize(
    "language,source",
    [
        ("python", b"x = " + b"(" * 5000 + b"1" + b")" * 5000 + b"\n"),
        ("javascript", b"const x = " + b"[" * 5000 + b"]" * 5000 + b";\n"),
        ("java", b"class A { int f() { return " + b"(" * 5000 + b"1" + b")" * 5000 + b"; } }\n"),
        ("kotlin", b"fun f() = " + b"(" * 5000 + b"1" + b")" * 5000 + b"\n"),
        ("c", b"int f() { " + b"{" * 5000 + b"}" * 5000 + b" }\n"),
    ],
)
def test_parse_blob_deeply_nested_input_does_not_hit_recursion_limit(
    language: str, source: bytes
) -> None:
    res = parse_blob(source, language)

    assert res.state in ("complete", "partial")
    assert res.symbols


def test_parse_blob_python_emits_import_inheritance_decorator_and_type_edges() -> None:
    source = (
        b"from pkg.models import Base, Repo as R\n"
        b"@register\n"
        b"class Service(Base, mixins.Logged):\n"
        b"    repo: R\n"
        b"    @cached\n"
        b"    def run(self, user: User) -> Result: pass\n"
    )

    assert _edges(source, "python") == {
        (None, "imports", "Base"),
        (None, "imports", "Repo"),
        ("Service", "decorates", "register"),
        ("Service", "inherits", "Base"),
        ("Service", "inherits", "mixins.Logged"),
        ("Service", "references_type", "R"),
        ("run", "decorates", "cached"),
        ("run", "references_type", "User"),
        ("run", "references_type", "Result"),
    }


def test_parse_blob_typescript_separates_extends_from_implements() -> None:
    source = (
        b'import { Base, Runner } from "./base";\n'
        b"@Component({})\n"
        b"export class Job extends Base<T> implements Runner, ns.Named {\n"
        b"  @Get() run(input: Input): Output { return input; }\n"
        b"}\n"
    )

    assert _edges(source, "typescript") == {
        (None, "imports", "Base"),
        (None, "imports", "Runner"),
        ("Job", "decorates", "Component"),
        ("Job", "inherits", "Base"),
        ("Job", "implements", "Runner"),
        ("Job", "implements", "ns.Named"),
        ("run", "decorates", "Get"),
        ("run", "references_type", "Input"),
        ("run", "references_type", "Output"),
    }


def test_parse_blob_java_and_kotlin_emit_annotations_and_supertypes() -> None:
    java = (
        b"import a.b.Base;\nimport x.*;\n"
        b"@Entity public class Job extends Base implements Runner {\n"
        b"  @Override public Output run(Input in) { return null; }\n"
        b"}\n"
    )
    kotlin = (
        b"import a.b.Base\n"
        b"@Entity class Job(val input: Input) : Base(), Runner {\n"
        b"  @Throws override fun run(): Output = TODO()\n"
        b"}\n"
        b"interface Runner : Named\n"
    )

    assert _edges(java, "java") == {
        (None, "imports", "Base"),
        ("Job", "decorates", "Entity"),
        ("Job", "inherits", "Base"),
        ("Job", "implements", "Runner"),
        ("run", "decorates", "Override"),
        ("run", "references_type", "Output"),
        ("run", "references_type", "Input"),
    }
    assert _edges(kotlin, "kotlin") == {
        (None, "imports", "Base"),
        ("Job", "decorates", "Entity"),
        ("Job", "inherits", "Base"),
        ("Job", "implements", "Runner"),
        ("Job", "references_type", "Input"),
        ("run", "decorates", "Throws"),
        ("run", "references_type", "Output"),
        ("Runner", "inherits", "Named"),
    }


def test_parse_blob_keeps_calls_inside_python_annotations() -> None:
    source = b"class M:\n    x: Annotated[Count, Field(gt=0)] = 1\n"

    res = parse_blob(source, "python")

    assert any(o.kind == "call" and o.details.get("callee") == "Field" for o in res.occurrences)
    assert {o.details["type_name"] for o in res.occurrences if o.kind == "references_type"} == {
        "Annotated",
        "Count",
    }


def _relations(mem: Memory) -> set[tuple[str, str, str]]:
    with mem._store.connect_read() as conn:
        rows = conn.execute(
            """
            SELECT f.name AS src, r.relation_kind AS kind, t.name AS dst
            FROM relations r
            JOIN symbols f ON f.id = r.from_id
            JOIN symbols t ON t.id = r.to_id
            WHERE r.relation_kind IN ('imports', 'inherits', 'implements', 'decorates', 'references_type', 'overrides')
            """
        ).fetchall()
    return {(str(r["src"]), str(r["kind"]), str(r["dst"])) for r in rows}


def test_sync_resolves_cross_file_relations_and_search_graph_filters_them(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Add services",
            files={
                "pkg/base.py": b"def register(cls):\n    return cls\n\nclass Base:\n    def run(self): pass\n\nclass User: pass\n",
                "pkg/service.py": (
                    b"from pkg.base import Base, User, register\n\n"
                    b"@register\n"
                    b"class Service(Base):\n"
                    b"    def run(self, user: User): pass\n"
                ),
                "other/base.py": b"class Base:\n    def run(self): pass\n",
            },
        ),
    )

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        relations = _relations(mem)
        inherits = mem.search_graph(
            {"relationship": "INHERITS", "name_pattern": "^Service$", "min_degree": 1}
        )

    assert {
        ("Service", "inherits", "Base"),
        ("Service", "decorates", "register"),
        ("run", "references_type", "User"),
        ("run", "overrides", "run"),
        ("Service", "imports", "Base"),
        ("Service", "imports", "register"),
        ("run", "imports", "User"),
    } <= relations
    assert [n.name for n in inherits.nodes] == ["Service"]


def test_sync_parallel_parse_matches_sequential_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files: dict[str | bytes, bytes | None] = {
        f"mod_{i}.py": f"from mod_{i - 1} import Item{i - 1}\n\nclass Item{i}(Item{i - 1}):\n    pass\n".encode()
        for i in range(1, 12)
    }
    files["mod_0.py"] = b"class Item0:\n    pass\n"

    def snapshot(store_dir: Path) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
        repo = init_fixture_repo(store_dir / "repo")
        commit_fixture_spec(repo, GitCommitSpec(message="Items", files=files))
        with Memory.initialize(store_dir / "store", InitRequest(repo_path=repo)) as mem:
            mem.sync(SyncRequest(offline=True))
            with mem._store.connect_read() as conn:
                symbols = [
                    tuple(map(str, r))
                    for r in conn.execute("SELECT id, qualified_name FROM symbols ORDER BY id")
                ]
                relations = [
                    tuple(map(str, r))
                    for r in conn.execute(
                        "SELECT from_id, to_id, relation_kind, method FROM relations WHERE from_kind = 'symbol' ORDER BY id"
                    )
                ]
        return symbols, relations

    parallel = snapshot(tmp_path / "parallel")
    monkeypatch.setattr(syntax, "PARALLEL_PARSE_MIN_BLOBS", 10_000)
    sequential = snapshot(tmp_path / "sequential")

    assert parallel == sequential
    assert len(parallel[0]) == 12


def test_parse_pool_workers_see_the_sync_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="init", files={"a.py": b"x = 1\n"}))
    rows = [{"path": f"m{i}.py".encode(), "blob_oid": f"oid{i}"} for i in range(8)]
    payloads: dict[str, tuple[int, bytes] | None] = {f"oid{i}": (6, b"x = 1\n") for i in range(8)}
    original = syntax.parse_blob
    seen_deadlines: list[float | None] = []

    def recording_parse(*args: object, **kwargs: object) -> syntax.ParseResult:
        seen_deadlines.append(deadline._deadline.get())
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(syntax, "parse_blob", recording_parse)
    with (
        Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem,
        sync_deadline(60),
    ):
        parsed, unparsed = syntax._parse_pending_blobs(mem._store, rows, payloads)  # type: ignore[arg-type]

    assert list(parsed) == [f"oid{i}:python" for i in range(8)]
    assert not unparsed
    assert len(seen_deadlines) == 8
    assert None not in seen_deadlines


def test_sync_checkpoints_pooled_parses_finished_before_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    files: dict[str | bytes, bytes | None] = {
        f"a{i}.py": f"def f{i}(): pass\n".encode() for i in range(7)
    }
    files["z_last.py"] = b"def stuck(): pass\n"
    commit_fixture_spec(repo, GitCommitSpec(message="Files", files=files))
    original = syntax.parse_blob

    def parse_until_stuck(blob_bytes: bytes, *args: object, **kwargs: object) -> syntax.ParseResult:
        if b"stuck" in blob_bytes:
            raise SyncDeadlineExceeded
        return original(blob_bytes, *args, **kwargs)  # type: ignore[arg-type]

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        monkeypatch.setattr(syntax, "parse_blob", parse_until_stuck)
        partial = mem.sync(SyncRequest(offline=True))
        with mem._store.connect_read() as conn:
            states = {
                bytes(r[0]).decode(): r[1]
                for r in conn.execute("SELECT path, content_state FROM snapshot_files")
            }
        monkeypatch.setattr(syntax, "parse_blob", original)
        resumed = mem.sync(SyncRequest(offline=True))

    assert partial.stop_reason == "time_budget"
    assert states.pop("z_last.py") == "pending"
    assert set(states.values()) == {"indexed"}
    assert resumed.coverage.structure.state == "complete"
