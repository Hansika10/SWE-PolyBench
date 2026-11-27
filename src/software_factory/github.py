import json
import re
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from software_factory.models import FactoryResult, Issue

Runner = Callable[[List[str], Optional[str]], str]

_REF_RE = re.compile(
    r"^(?:https?://github\.com/)?(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)(?:#|/issues/)(?P<number>\d+)/?$"
)


def default_runner(args: List[str], cwd: Optional[str] = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def parse_issue_ref(ref: str):
    match = _REF_RE.match(ref.strip())
    if not match:
        raise ValueError(f"Expected owner/repo#123 or a GitHub issue URL, got: {ref}")
    return match["owner"], match["repo"], int(match["number"])


def issue_from_github(ref: str, runner: Runner = default_runner) -> Issue:
    owner, repo, number = parse_issue_ref(ref)
    raw = runner(
        ["gh", "issue", "view", str(number), "-R", f"{owner}/{repo}", "--json", "title,body,url"],
        None,
    )
    data = json.loads(raw)
    return Issue(
        instance_id=f"{owner}__{repo}-{number}",
        repo=f"{owner}/{repo}",
        problem_statement=f"{data['title']}\n\n{data.get('body') or ''}".strip(),
        issue_url=data["url"],
    )


def clone_repo(repo: str, dest: str, runner: Runner = default_runner) -> str:
    if not (Path(dest) / ".git").exists():
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        runner(["gh", "repo", "clone", repo, dest], None)
    return dest


def open_pull_request(
    repo_dir: str,
    result: FactoryResult,
    patch_file: str,
    base_branch: Optional[str] = None,
    draft: bool = True,
    runner: Runner = default_runner,
) -> str:
    """Commit only the final patch on a new branch, push it and open a PR. Returns the PR URL.

    Leftover agent files (scratch scripts, tests filtered from the patch) are stashed, not lost.
    """
    branch = f"software-factory/{result.instance_id.lower()}"
    body_file = str(Path(patch_file).with_name("pr_body.md"))
    Path(body_file).write_text(result.pr_body)
    patch_file = str(Path(patch_file).resolve())

    runner(["git", "stash", "push", "--include-untracked", "-m", f"factory {branch}"], repo_dir)
    runner(["git", "checkout", "-b", branch], repo_dir)
    runner(["git", "apply", "--index", patch_file], repo_dir)
    runner(["git", "commit", "-m", result.pr_title], repo_dir)
    runner(["git", "push", "-u", "origin", branch], repo_dir)
    args = ["gh", "pr", "create", "--title", result.pr_title, "--body-file", body_file, "--head", branch]
    if base_branch:
        args += ["--base", base_branch]
    if draft:
        args.append("--draft")
    return runner(args, repo_dir).strip()
