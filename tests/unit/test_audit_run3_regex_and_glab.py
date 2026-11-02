import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from codecontext.adapters import strip_markdown_json_fences
from codecontext.docs_ingest import _extract_markdown_title
from codecontext.github import extract_issue_references
from codecontext.gitlab import resolve_gitlab_token
from codecontext.syntax import parse_blob

TIME_LIMIT_SECONDS = 1.0


def _elapsed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_strip_markdown_json_fences_keeps_fenced_and_inline_json() -> None:
    assert strip_markdown_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_markdown_json_fences('```\n{"a": 1}  \n  ```') == '{"a": 1}'
    assert strip_markdown_json_fences('```json\n{"a": 1}') == '{"a": 1}'
    assert strip_markdown_json_fences('Here you go:\n```json\n{"a": 1}\n```\nDone.') == '{"a": 1}'
    assert strip_markdown_json_fences('text ```{"a": 1}``` more ```x```') == '{"a": 1}'
    assert strip_markdown_json_fences("note ```json``` end") == "note ```json``` end"
    assert strip_markdown_json_fences('{"a": "```"}') == '{"a": "```"}'


def test_strip_markdown_json_fences_is_linear_on_whitespace_runs() -> None:
    assert _elapsed(lambda: strip_markdown_json_fences("```\n" + "\t" * 60_000 + "x")) < TIME_LIMIT_SECONDS
    assert _elapsed(lambda: strip_markdown_json_fences("a```" + "\t" * 2_000 + "x")) < TIME_LIMIT_SECONDS


def test_markdown_title_reads_first_h1() -> None:
    assert _extract_markdown_title("\n\n  # Use SQLite  \nbody\n# Later", "docs/adr/0001.md") == "Use SQLite"
    assert _extract_markdown_title("## Not h1\n# Real Title\n", "docs/adr/0002.md") == "Real Title"
    assert _extract_markdown_title("no heading here", "docs/adr/0003-pick-db.md") == "0003-pick-db"


def test_markdown_title_is_linear_on_newline_runs() -> None:
    body = "\n" * 60_000 + "x"
    assert _elapsed(lambda: _extract_markdown_title(body, "docs/adr/0001.md")) < TIME_LIMIT_SECONDS
    line = "# x" + " " * 60_000 + "y"
    assert _elapsed(lambda: _extract_markdown_title(line, "docs/adr/0001.md")) < TIME_LIMIT_SECONDS


def test_issue_references_match_normal_text() -> None:
    assert extract_issue_references("fixes #12", "o/r") == [("o/r", 12)]
    assert extract_issue_references("see owner/repo#3.", "o/r") == [("owner/repo", 3)]
    assert extract_issue_references("owner/repo#12 and #12 and #12", "o/r") == [("owner/repo", 12), ("o/r", 12)]
    assert extract_issue_references("my.org/my-repo_1#7", "o/r") == [("my.org/my-repo_1", 7)]


def test_issue_references_are_linear_on_class_runs() -> None:
    text = "a" * 65_536
    assert _elapsed(lambda: extract_issue_references(text, "o/r")) < TIME_LIMIT_SECONDS
    assert extract_issue_references(text + " #1", "o/r") == [("o/r", 1)]


STRUCTURAL_CASES = [
    ("ruby", b"\n" * 40_000 + b"x"),
    ("bash", b"\n" * 16_000 + b"x"),
    ("c_sharp", b"class A :" + b" " * 2_000),
    ("swift", b"class A :" + b" " * 2_000),
    ("scala", b"class A with " + b" " * 2_000),
    ("php", b"class A implements " + b" " * 2_000),
    ("php", b"function a(" * 20_000),
    ("swift", b"func a(" * 30_000),
    ("c_sharp", b"public int A(" * 16_000),
    ("scala", b"def a(" * 35_000),
    ("protobuf", b"rpc a(" * 35_000),
    ("c_sharp", b"public " * 6_000),
    ("php", b"abstract " * 10_000),
    ("swift", b"public " * 8_000),
    ("ruby", b"class A\n" * 8_000),
    ("c_sharp", b"static<" * 16_000),
    ("c_sharp", b"public?" * 16_000),
]


@pytest.mark.parametrize(("language", "blob"), STRUCTURAL_CASES, ids=[f"{lang}-{i}" for i, (lang, _) in enumerate(STRUCTURAL_CASES)])
def test_structural_fallback_is_linear_on_crafted_blobs(language: str, blob: bytes) -> None:
    assert _elapsed(lambda: parse_blob(blob, language)) < TIME_LIMIT_SECONDS


def _names(language: str, blob: bytes) -> list[tuple[str, str]]:
    return [(s.kind, s.name) for s in parse_blob(blob, language).symbols]


def test_structural_fallback_keeps_multiline_symbols() -> None:
    cs = (
        b"namespace App {\n    public class Greeter : Base, IFoo<int>\n    {\n"
        b"        public static async Task<int> SayHello(string a, int b)\n        {\n        }\n    }\n"
        b"    class Empty : {\n    }\n}\n"
    )
    cs_result = parse_blob(cs, "c_sharp")
    assert [(s.kind, s.name) for s in cs_result.symbols] == [
        ("class", "Greeter"),
        ("class", "Empty"),
        ("method", "SayHello"),
    ]
    assert {o.details["base_class"] for o in cs_result.occurrences if o.kind == "inherits"} == {"Base", "IFoo"}
    assert _names("ruby", b"\n\nclass Worker < Base::Job\n  def self.build?\n  end\nend\n") == [
        ("class", "Worker"),
        ("method", "build?"),
    ]
    assert parse_blob(b"\n\nclass Worker\nend\n", "ruby").symbols[0].line_start == 3
    assert _names("bash", b"\nbuild ()\n{\n  make\n}\nfunction cleanup {\n}\n") == [
        ("function", "build"),
        ("function", "cleanup"),
    ]
    assert _names("php", b"<?php\nabstract class Router extends Base implements A, B\n{\n    private static function make(array $a = [])\n    {\n    }\n}\n") == [
        ("class", "Router"),
        ("function", "make"),
    ]
    assert _names("swift", b"struct S<T>: P\n{\n}\nextension Service: P { }\n") == [("class", "S"), ("class", "Service")]
    assert _names("scala", b"abstract class Calc extends Base with Mixin with Other {\n  def gen[T](t: T): T = t\n}\n") == [
        ("class", "Calc"),
        ("function", "gen"),
    ]


def _fake_glab(bin_dir: Path, script: str) -> None:
    glab = bin_dir / "glab"
    glab.write_text("#!/bin/sh\n" + script + "\n", encoding="utf-8")
    glab.chmod(0o755)


@pytest.mark.parametrize("env_name", ["GITLAB_TOKEN", "GITLAB_ACCESS_TOKEN", "OAUTH_TOKEN", "GLAB_TOKEN"])
def test_glab_fallback_cannot_return_env_token_for_other_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env_name: str
) -> None:
    _fake_glab(tmp_path, f'printf "%s" "${env_name}"')
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr("codecontext.gitlab.Path.home", lambda: tmp_path)
    monkeypatch.delenv("GITLAB_HOST", raising=False)
    monkeypatch.setenv(env_name, "dummy-token")
    assert resolve_gitlab_token("https://gitlab.attacker.example") is None


def test_glab_fallback_returns_token_stored_for_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_glab(tmp_path, '[ -n "$GITLAB_HOST$GITLAB_TOKEN" ] && exit 1; [ "$5" = "gitlab.example.internal" ] && printf stored-token')
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr("codecontext.gitlab.Path.home", lambda: tmp_path)
    monkeypatch.setenv("GITLAB_HOST", "gitlab.attacker.example")
    monkeypatch.setenv("GITLAB_TOKEN", "dummy-token")
    assert resolve_gitlab_token("https://gitlab.example.internal") == "stored-token"
