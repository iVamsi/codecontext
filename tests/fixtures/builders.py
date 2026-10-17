import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GitCommitSpec:
    message: str
    files: dict[str | bytes, bytes | None]
    author_name: str = "Fixture Author"
    author_email: str = "author@example.test"
    authored_at: str = "2026-01-01T00:00:00Z"
    committer_name: str = "Fixture Committer"
    committer_email: str = "committer@example.test"
    committed_at: str = "2026-01-01T00:00:00Z"


def run_git_fixture(repo_dir: Path, args: list[str], env_extra: dict[str, str] | None = None) -> bytes:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        "HOME": str(repo_dir.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def init_fixture_repo(repo_dir: Path, *, object_format: str = "sha1", default_branch: str = "main") -> Path:
    repo_dir.mkdir(parents=True, exist_ok=True)
    empty_template = repo_dir.parent / "_empty_git_template"
    empty_template.mkdir(parents=True, exist_ok=True)
    init_args = ["init", f"--template={empty_template}", f"--initial-branch={default_branch}"]
    if object_format == "sha256":
        init_args.append("--object-format=sha256")
    run_git_fixture(repo_dir, init_args)
    run_git_fixture(repo_dir, ["config", "user.name", "Fixture Author"])
    run_git_fixture(repo_dir, ["config", "user.email", "author@example.test"])
    run_git_fixture(repo_dir, ["config", "gc.auto", "0"])
    return repo_dir


def commit_fixture_spec(repo_dir: Path, spec: GitCommitSpec) -> str:
    for rel_path, content in spec.files.items():
        if isinstance(rel_path, bytes):
            target = Path(os.fsdecode(bytes(repo_dir) + b"/" + rel_path))
        else:
            target = repo_dir / rel_path
        if content is None:
            if target.exists() or target.is_symlink():
                target.unlink()
            run_git_fixture(
                repo_dir,
                ["rm", "-f", "--ignore-unmatch", "--", os.fsdecode(rel_path) if isinstance(rel_path, bytes) else rel_path],
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            run_git_fixture(
                repo_dir,
                ["add", "--", os.fsdecode(rel_path) if isinstance(rel_path, bytes) else rel_path],
            )

    env = {
        "GIT_AUTHOR_NAME": spec.author_name,
        "GIT_AUTHOR_EMAIL": spec.author_email,
        "GIT_AUTHOR_DATE": spec.authored_at,
        "GIT_COMMITTER_NAME": spec.committer_name,
        "GIT_COMMITTER_EMAIL": spec.committer_email,
        "GIT_COMMITTER_DATE": spec.committed_at,
    }
    run_git_fixture(repo_dir, ["commit", "--allow-empty", "-m", spec.message], env_extra=env)
    return run_git_fixture(repo_dir, ["rev-parse", "HEAD"]).decode("ascii").strip()


@dataclass
class F01Fixture:
    repo_dir: Path
    c0_oid: str
    c1_oid: str
    c2_oid: str
    c0_code: bytes
    c1_code: bytes
    c2_readme: bytes
    pr_payload: dict[str, Any]
    comments_payload: list[dict[str, Any]]
    reviews_payload: list[dict[str, Any]]
    gold_spans: dict[str, tuple[int, int]] = field(default_factory=dict)


def build_f01_fixture(tmp_path: Path, *, object_format: str = "sha1") -> F01Fixture:
    """F01: documented retry rationale.

    - transport.py with `send` at C0
    - C1 adds a three-attempt retry limit
    - Mock PR contains explicit rationale + rejected unlimited retries
    - C2 changes unrelated documentation
    """
    repo_dir = init_fixture_repo(tmp_path / "f01_repo", object_format=object_format)
    c0_code = (
        b"def send(payload: bytes) -> bool:\n"
        b'    """Send payload over transport."""\n'
        b"    return _dispatch(payload)\n\n\n"
        b"def _dispatch(payload: bytes) -> bool:\n"
        b"    return len(payload) > 0\n"
    )
    c1_code = (
        b"MAX_RETRIES = 3\n\n\n"
        b"def send(payload: bytes) -> bool:\n"
        b'    """Send payload over transport with bounded retries."""\n'
        b"    for attempt in range(MAX_RETRIES):\n"
        b"        if _dispatch(payload):\n"
        b"            return True\n"
        b"    return False\n\n\n"
        b"def _dispatch(payload: bytes) -> bool:\n"
        b"    return len(payload) > 0\n\n\n"
        b"def undocumented_helper(x: int) -> int:\n"
        b"    return x + 1\n"
    )
    c2_readme = b"# Transport Service\n\nUpdated docs.\n"

    c0 = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Initial transport send implementation",
            files={"src/transport.py": c0_code, "README.md": b"# Transport\n"},
            authored_at="2026-01-01T10:00:00Z",
            committed_at="2026-01-01T10:00:00Z",
        ),
    )
    c1 = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Bound send retries to 3 attempts (#10)",
            files={"src/transport.py": c1_code},
            authored_at="2026-01-02T10:00:00Z",
            committed_at="2026-01-02T10:00:00Z",
        ),
    )
    c2 = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Update README documentation",
            files={"README.md": c2_readme},
            authored_at="2026-01-03T10:00:00Z",
            committed_at="2026-01-03T10:00:00Z",
        ),
    )

    pr_body = (
        "Limit retries to three because each retry holds a worker and the legacy proxy "
        "duplicates requests after its timeout."
    )
    proposal_body = "Why not allow unlimited retries with exponential backoff until the proxy recovers?"
    rejection_body = (
        "Rejected unlimited retries: holding workers indefinitely exhausts the worker pool "
        "under proxy brownouts."
    )

    pr_payload = {
        "id": 10010,
        "number": 10,
        "state": "closed",
        "title": "Bound send retries to 3 attempts",
        "body": pr_body,
        "user": {"id": 501, "login": "alice"},
        "created_at": "2026-01-02T09:00:00Z",
        "updated_at": "2026-01-02T10:05:00Z",
        "merged_at": "2026-01-02T10:00:00Z",
        "merge_commit_sha": c1,
        "head": {"sha": c1, "ref": "fix/retry-limit"},
        "base": {"sha": c0, "ref": "main"},
        "html_url": "https://github.com/example/transport/pull/10",
    }
    comments_payload = [
        {
            "id": 20001,
            "body": proposal_body,
            "user": {"id": 502, "login": "bob"},
            "created_at": "2026-01-02T09:15:00Z",
            "updated_at": "2026-01-02T09:15:00Z",
            "html_url": "https://github.com/example/transport/pull/10#issuecomment-20001",
        },
        {
            "id": 20002,
            "body": rejection_body,
            "user": {"id": 501, "login": "alice"},
            "created_at": "2026-01-02T09:30:00Z",
            "updated_at": "2026-01-02T09:30:00Z",
            "html_url": "https://github.com/example/transport/pull/10#issuecomment-20002",
        },
    ]
    reviews_payload = [
        {
            "id": 30001,
            "user": {"id": 503, "login": "carol"},
            "body": "Approved: three attempts matches our worker timeout budget.",
            "state": "APPROVED",
            "commit_id": c1,
            "submitted_at": "2026-01-02T09:45:00Z",
            "html_url": "https://github.com/example/transport/pull/10#pullrequestreview-30001",
        }
    ]

    pr_bytes = pr_body.encode("utf-8")
    c1_quote = b"each retry holds a worker"
    c2_quote = b"the legacy proxy duplicates requests after its timeout"
    gold_spans = {
        "full_rationale": (0, len(pr_bytes)),
        "constraint_worker": (pr_bytes.index(c1_quote), pr_bytes.index(c1_quote) + len(c1_quote)),
        "constraint_proxy": (pr_bytes.index(c2_quote), pr_bytes.index(c2_quote) + len(c2_quote)),
        "rejected_alternative": (0, len(rejection_body.encode("utf-8"))),
    }

    return F01Fixture(
        repo_dir=repo_dir,
        c0_oid=c0,
        c1_oid=c1,
        c2_oid=c2,
        c0_code=c0_code,
        c1_code=c1_code,
        c2_readme=c2_readme,
        pr_payload=pr_payload,
        comments_payload=comments_payload,
        reviews_payload=reviews_payload,
        gold_spans=gold_spans,
    )


@dataclass
class F03Fixture:
    repo_dir: Path
    root_oid: str
    branch_a_oid: str
    branch_b_oid: str
    merge_oid: str
    backdated_oid: str
    divergent_oid: str


def build_f03_fixture(tmp_path: Path) -> F03Fixture:
    """F03: branching and integration history.

    Includes root commit, two-parent merge with distinct changes against each parent,
    divergent branch with equal-looking patch, and a commit whose timestamp predates its parent.
    """
    repo_dir = init_fixture_repo(tmp_path / "f03_repo")
    root_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Root commit",
            files={"a.py": b"def a():\n    return 1\n", "b.py": b"def b():\n    return 1\n"},
            authored_at="2026-01-01T12:00:00Z",
            committed_at="2026-01-01T12:00:00Z",
        ),
    )

    run_git_fixture(repo_dir, ["checkout", "-b", "branch-a", root_oid])
    branch_a_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Modify a.py on branch-a",
            files={"a.py": b"def a():\n    return 2\n"},
            authored_at="2026-01-02T12:00:00Z",
            committed_at="2026-01-02T12:00:00Z",
        ),
    )

    run_git_fixture(repo_dir, ["checkout", "-b", "branch-b", root_oid])
    branch_b_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Modify b.py on branch-b",
            files={"b.py": b"def b():\n    return 3\n"},
            authored_at="2026-01-02T13:00:00Z",
            committed_at="2026-01-02T13:00:00Z",
        ),
    )

    run_git_fixture(repo_dir, ["checkout", "main"])
    run_git_fixture(repo_dir, ["reset", "--hard", branch_a_oid])
    env = {
        "GIT_AUTHOR_NAME": "Fixture Author",
        "GIT_AUTHOR_EMAIL": "author@example.test",
        "GIT_AUTHOR_DATE": "2026-01-03T12:00:00Z",
        "GIT_COMMITTER_NAME": "Fixture Committer",
        "GIT_COMMITTER_EMAIL": "committer@example.test",
        "GIT_COMMITTER_DATE": "2026-01-03T12:00:00Z",
    }
    run_git_fixture(repo_dir, ["merge", "--no-ff", branch_b_oid, "-m", "Merge branch-b into main"], env_extra=env)
    merge_oid = run_git_fixture(repo_dir, ["rev-parse", "HEAD"]).decode("ascii").strip()

    # Commit whose timestamp predates its parent
    backdated_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Backdated commit after merge",
            files={"c.py": b"def c():\n    return 4\n"},
            authored_at="2025-12-31T00:00:00Z",
            committed_at="2025-12-31T00:00:00Z",
        ),
    )

    # Divergent branch with equal-looking change to c.py
    run_git_fixture(repo_dir, ["checkout", "-b", "divergent", root_oid])
    divergent_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Equal patch on divergent branch",
            files={"c.py": b"def c():\n    return 4\n"},
            authored_at="2026-01-04T00:00:00Z",
            committed_at="2026-01-04T00:00:00Z",
        ),
    )
    run_git_fixture(repo_dir, ["checkout", "main"])

    return F03Fixture(
        repo_dir=repo_dir,
        root_oid=root_oid,
        branch_a_oid=branch_a_oid,
        branch_b_oid=branch_b_oid,
        merge_oid=merge_oid,
        backdated_oid=backdated_oid,
        divergent_oid=divergent_oid,
    )


@dataclass
class F09Samples:
    python_source: bytes
    javascript_source: bytes
    typescript_source: bytes
    jsx_source: bytes
    tsx_source: bytes
    malformed_python: bytes
    unicode_crlf_source: bytes
    unsupported_text: bytes


def build_f09_samples() -> F09Samples:
    """F09: syntax, paths, and size limits samples."""
    python_source = (
        b"import os\n"
        b"from pathlib import Path\n\n\n"
        b"DEFAULT_LIMIT = 10\n\n\n"
        b"def helper(val: int) -> int:\n"
        b'    """Helper docstring."""\n'
        b"    return val * 2\n\n\n"
        b"class Processor:\n"
        b'    """Processor class docstring."""\n\n'
        b"    @staticmethod\n"
        b"    def compute(x: int) -> int:\n"
        b'        """Compute docstring."""\n'
        b"        def inner(y: int) -> int:\n"
        b"            return helper(y)\n"
        b"        return inner(x)\n"
    )
    javascript_source = (
        b"import { util } from './util.js';\n\n"
        b"export function runTask(input) {\n"
        b"  return util(input);\n"
        b"}\n\n"
        b"export const arrowFn = (a, b) => a + b;\n\n"
        b"export class Service {\n"
        b"  execute(item) {\n"
        b"    return runTask(item);\n"
        b"  }\n"
        b"}\n"
    )
    typescript_source = (
        b"import type { Config } from './config.js';\n\n"
        b"export interface Runner {\n"
        b"  run(cfg: Config): boolean;\n"
        b"}\n\n"
        b"export type Mode = 'fast' | 'safe';\n\n"
        b"export enum StatusCode {\n"
        b"  Ok = 0,\n"
        b"  Failed = 1,\n"
        b"}\n\n"
        b"export class DefaultRunner implements Runner {\n"
        b"  run(cfg: Config): boolean {\n"
        b"    return Boolean(cfg);\n"
        b"  }\n"
        b"}\n"
    )
    jsx_source = (
        b"import React from 'react';\n"
        b"import { Badge } from './Badge.jsx';\n\n"
        b"export function Card({ title }) {\n"
        b"  return <div className='card'><Badge label={title} /></div>;\n"
        b"}\n"
    )
    tsx_source = (
        b"import React from 'react';\n\n"
        b"export interface Props {\n"
        b"  label: string;\n"
        b"}\n\n"
        b"export function ActionButton(props: Props) {\n"
        b"  return <button>{props.label}</button>;\n"
        b"}\n"
    )
    malformed_python = (
        b"def valid_before(x: int) -> int:\n"
        b"    return x + 1\n\n"
        b"def broken_func(:\n"
        b"    pass\n\n"
        b"def valid_after(y: int) -> int:\n"
        b"    return y * 2\n"
    )
    unicode_crlf_source = (
        "# Café résumé — naïve 🚀\r\n"
        "def greet_unicode(name: str) -> str:\r\n"
        '    """Return greeting with Ω and 日本語."""\r\n'
        '    return f"Bonjour {name} — Ω"\r\n'
    ).encode()
    unsupported_text = b"# Configuration Notes\n\nPlain documentation text.\n"
    return F09Samples(
        python_source=python_source,
        javascript_source=javascript_source,
        typescript_source=typescript_source,
        jsx_source=jsx_source,
        tsx_source=tsx_source,
        malformed_python=malformed_python,
        unicode_crlf_source=unicode_crlf_source,
        unsupported_text=unsupported_text,
    )
