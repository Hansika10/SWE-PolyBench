import json

from software_factory.llm import MockLLM, tool_call
from software_factory.models import Issue, LLMResponse
from software_factory.pipeline import FactoryConfig, SoftwareFactory, draft_pr
from software_factory.workspace import LocalWorkspace

ISSUE = Issue(
    instance_id="acme__calc-1",
    repo="acme/calc",
    problem_statement="# add() subtracts\n\n`add(2, 3)` returns -1 instead of 5.",
    language="Python",
)

TEST_FILE = "tests/test_ops.py"
TEST_CONTENT = "from calc.ops import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def scripted_responder(verify_results):
    """Plans, fixes ops.py, then verifies by writing and running a pytest file."""
    verify_results = list(verify_results)

    def respond(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"]
        started = any("toolResult" in block for block in last)
        if "submit_plan" in names:
            if not started:
                return tool_call("search", pattern="def add")
            return tool_call(
                "submit_plan", summary="add() uses '-'", files=["calc/ops.py"], steps=["use '+'"]
            )
        if "submit_changes" in names:
            if not started:
                return tool_call("str_replace", path="calc/ops.py", old_str="a - b", new_str="a + b")
            return tool_call("submit_changes", summary="Fixed operator")
        if "submit_verification" in names:
            if not started:
                return tool_call("write_file", path=TEST_FILE, content=TEST_CONTENT)
            if len(messages) == 3:
                return tool_call("bash", command="python -m pytest -q tests/test_ops.py")
            passed, summary = verify_results.pop(0)
            return tool_call(
                "submit_verification", passed=passed, summary=summary, commands=["pytest"]
            )
        return LLMResponse(text="")

    return respond


def test_end_to_end_issue_to_patch(git_repo, tmp_path):
    llm = MockLLM(scripted_responder([(True, "test_add passes")]))
    factory = SoftwareFactory(
        llm, lambda issue: LocalWorkspace(str(git_repo)), artifacts_dir=str(tmp_path / "art")
    )

    result = factory.run(ISSUE)

    assert result.status == "verified", result.error
    assert result.iterations == 1
    assert "-    return a - b\n+    return a + b" in result.model_patch
    assert TEST_FILE not in result.model_patch
    assert result.plan.files == ["calc/ops.py"]
    assert result.pr_title == "Fix: add() subtracts"
    assert "`calc/ops.py`" in result.pr_body and "Verification passed" in result.pr_body

    # The agent really ran pytest in the workspace and saw it pass.
    verify = json.loads((tmp_path / "art" / ISSUE.instance_id / "transcript_verify_1.json").read_text())
    pytest_output = verify["messages"][4]["content"][0]["toolResult"]["content"][0]["text"]
    assert "1 passed" in pytest_output

    art = tmp_path / "art" / ISSUE.instance_id
    for name in ["patch.diff", "pr.md", "result.json", "transcript_plan.json", "transcript_code_1.json"]:
        assert (art / name).exists(), name
    assert (art / "patch.diff").read_text() == result.model_patch


def test_failed_verification_triggers_another_iteration(git_repo):
    llm = MockLLM(scripted_responder([(False, "edge case broken"), (False, "still broken")]))
    factory = SoftwareFactory(
        llm, lambda issue: LocalWorkspace(str(git_repo)), FactoryConfig(max_iterations=2)
    )

    result = factory.run(ISSUE)

    assert result.iterations == 2
    assert result.status == "unverified"
    assert result.model_patch
    second_code_task = [
        c for c in llm.calls if any(t["name"] == "submit_changes" for t in c["tools"])
    ][2]["messages"][0]["content"][0]["text"]
    assert "edge case broken" in second_code_task
    assert "<current_diff>" in second_code_task


def test_no_changes_yields_no_patch(git_repo):
    factory = SoftwareFactory(MockLLM(), lambda issue: LocalWorkspace(str(git_repo)))
    result = factory.run(ISSUE)
    assert result.status == "no_patch"
    assert result.model_patch == ""
    assert result.verification.summary == "No source changes were made."


def test_workspace_failure_is_reported(tmp_path):
    class Broken(LocalWorkspace):
        def start(self):
            raise RuntimeError("image pull failed")

    factory = SoftwareFactory(MockLLM(), lambda issue: Broken(str(tmp_path)))
    result = factory.run(ISSUE)
    assert result.status == "error"
    assert "image pull failed" in result.error


def test_agent_hits_step_limit(git_repo):
    llm = MockLLM(lambda s, m, t: LLMResponse(text="thinking..."))
    factory = SoftwareFactory(
        llm,
        lambda issue: LocalWorkspace(str(git_repo)),
        FactoryConfig(max_plan_steps=2, max_code_steps=2, max_iterations=1),
    )
    result = factory.run(ISSUE)
    assert result.status == "no_patch"
    assert len(llm.calls) == 4


def test_run_batch_writes_predictions_and_skips_existing(git_repo, tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    factory = SoftwareFactory(
        MockLLM(scripted_responder([(True, "ok")])), lambda issue: LocalWorkspace(str(git_repo))
    )
    factory.run_batch([ISSUE], str(predictions))
    rows = [json.loads(l) for l in predictions.read_text().splitlines()]
    assert rows[0]["instance_id"] == ISSUE.instance_id
    assert rows[0]["model_name_or_path"] == "mock"
    assert "a + b" in rows[0]["model_patch"]

    assert factory.run_batch([ISSUE], str(predictions), skip_existing=True) == []
    assert len(predictions.read_text().splitlines()) == 1


def test_draft_pr_without_plan():
    title, body = draft_pr(Issue("i", "o/r", ""), None, None, "")
    assert title == "Fix i"
    assert "Not run." in body
