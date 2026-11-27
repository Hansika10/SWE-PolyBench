from unittest.mock import Mock

from software_factory.models import Issue
from software_factory.tools import ToolExecutor, truncate
from software_factory.workspace import (
    DockerWorkspace,
    LocalWorkspace,
    filter_patch,
    is_test_path,
    patch_files,
)


def test_baseline_diff_only_contains_agent_edits(git_repo):
    (git_repo / "preexisting_untracked.txt").write_text("already here\n")
    ws = LocalWorkspace(str(git_repo))
    ws.setup_baseline()
    assert ws.diff() == ""

    (git_repo / "calc" / "ops.py").write_text("def add(a, b):\n    return a + b\n")
    (git_repo / "calc" / "new.py").write_text("X = 1\n")
    patch = ws.diff()

    assert patch_files(patch) == ["calc/new.py", "calc/ops.py"]
    assert "+    return a + b" in patch
    # No commits were created and nothing is left staged.
    assert ws.run("git rev-list --count HEAD").output.strip() == "1"
    assert ws.run("git diff --cached --name-only").output.strip() == ""


def test_filter_patch_drops_scratch_and_tests():
    patch = "".join(
        f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n"
        for p in ["src/app.py", ".factory_scratch/repro.py", "tests/test_app.py", "src/app.test.ts"]
    )
    assert patch_files(filter_patch(patch)) == ["src/app.py"]
    assert patch_files(filter_patch(patch, keep_test_changes=True)) == [
        "src/app.py",
        "tests/test_app.py",
        "src/app.test.ts",
    ]


def test_is_test_path():
    for path in [
        "tests/test_x.py",
        "pkg/test_x.py",
        "src/test/java/FooTest.java",
        "lib/foo.spec.ts",
        "a/__tests__/b.js",
        "conftest.py",
    ]:
        assert is_test_path(path), path
    for path in ["src/main/java/Foo.java", "lib/foo.ts", "pkg/contest.py", "src/attest.py"]:
        assert not is_test_path(path), path


def test_tools_read_search_edit(git_repo):
    ws = LocalWorkspace(str(git_repo))
    tools = ToolExecutor(ws)

    out, ok = tools.execute("list_files", {})
    assert ok and "calc/ops.py" in out
    out, ok = tools.execute("search", {"pattern": "return a - b"})
    assert ok and "calc/ops.py:2:" in out
    out, ok = tools.execute("view_file", {"path": "calc/ops.py", "start_line": 2})
    assert "     2\t    return a - b" in out and "[lines 2-2 of 2]" in out

    out, ok = tools.execute("str_replace", {"path": "calc/ops.py", "old_str": "a - b", "new_str": "a + b"})
    assert ok
    assert "a + b" in (git_repo / "calc" / "ops.py").read_text()
    out, ok = tools.execute("str_replace", {"path": "calc/ops.py", "old_str": "nope", "new_str": "x"})
    assert not ok and "found 0" in out

    out, ok = tools.execute("write_file", {"path": "deep/dir/f.txt", "content": "hi\n"})
    assert ok and (git_repo / "deep" / "dir" / "f.txt").read_text() == "hi\n"

    out, ok = tools.execute("bash", {"command": "exit 3"})
    assert not ok and "[exit code: 3]" in out


def test_tool_errors_are_returned_not_raised(git_repo):
    tools = ToolExecutor(LocalWorkspace(str(git_repo)), tools=["view_file"])
    out, ok = tools.execute("bash", {"command": "ls"})
    assert not ok and "Unknown or unavailable" in out
    out, ok = tools.execute("view_file", {"path": "missing.py"})
    assert not ok and "FileNotFoundError" in out
    out, ok = tools.execute("view_file", {"bogus": 1})
    assert not ok and "Invalid arguments" in out


def test_local_timeout(git_repo):
    result = LocalWorkspace(str(git_repo)).run("sleep 5", timeout=1)
    assert result.timed_out and result.exit_code == 124


def test_truncate():
    text = "x" * 100
    assert truncate(text, 200) == text
    short = truncate(text, 30)
    assert "truncated" in short and short.startswith("x" * 20)


def test_docker_workspace_run_wraps_timeout():
    container = Mock()
    container.exec_run.return_value = Mock(exit_code=0, output=b"hello")
    issue = Issue(instance_id="o__r-1", repo="o/r", problem_statement="p", language="Python")
    ws = DockerWorkspace(issue, client=Mock())
    ws.container, ws.root = container, "/testbed"

    result = ws.run("echo 'hi'", timeout=12)

    args, kwargs = container.exec_run.call_args
    assert args[0][:2] == ["bash", "-c"]
    assert args[0][2].startswith("timeout -s KILL 12 bash -c ")
    assert kwargs["workdir"] == "/testbed"
    assert result.output == "hello" and result.exit_code == 0
    assert ws.image_id == "polybench_python_o__r-1"
    assert ws.resolve("a/b.py") == "/testbed/a/b.py"
