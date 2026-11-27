import subprocess
from pathlib import Path

import pytest

BUGGY_SOURCE = '''def add(a, b):
    return a - b
'''


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "calc").mkdir(parents=True)
    (repo / "calc" / "__init__.py").write_text("")
    (repo / "calc" / "ops.py").write_text(BUGGY_SOURCE)
    (repo / "README.md").write_text("calc\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo
