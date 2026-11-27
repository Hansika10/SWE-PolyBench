import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from software_factory import cli
from software_factory.github import issue_from_github, open_pull_request, parse_issue_ref
from software_factory.intake import issue_from_row, load_issues, select_instances
from software_factory.models import FactoryResult
from software_factory.scoring import score_predictions, summarize


def _dataset(tmp_path: Path) -> str:
    rows = [
        {
            "instance_id": f"o__r-{i}",
            "repo": "o/r",
            "base_commit": "abc",
            "language": lang,
            "problem_statement": f"issue {i}",
            "hints_text": "hint",
            "Dockerfile": "FROM x",
            "patch": "GOLD",
            "test_patch": "TESTS",
            "F2P": "['t1']",
            "P2P": "[]",
            "test_command": "pytest",
            "task_category": "Bug Fix",
            "modified_nodes": "[]",
        }
        for i, lang in enumerate(["Python", "Java", "Python", "TypeScript"])
    ]
    path = tmp_path / "ds.jsonl"
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    return str(path)


def test_issue_never_exposes_grading_data(tmp_path):
    issue = load_issues(_dataset(tmp_path), instance_ids=["o__r-0"])[0]
    values = json.dumps(issue.__dict__)
    for secret in ["GOLD", "TESTS", "t1", "pytest"]:
        assert secret not in values
    assert issue.hints_text == ""
    assert issue_from_row(pd.Series({"instance_id": "a", "repo": "b", "hints_text": "h"}), True).hints_text == "h"


def test_select_instances(tmp_path):
    df = pd.read_json(_dataset(tmp_path), lines=True)
    assert len(select_instances(df, languages=["python"])) == 2
    assert select_instances(df, per_language=1)["instance_id"].tolist() == ["o__r-0", "o__r-1", "o__r-3"]
    assert len(select_instances(df, limit=1)) == 1
    with pytest.raises(ValueError):
        select_instances(df, instance_ids=["nope"])


def test_score_predictions_subsets_dataset_and_summarizes(tmp_path):
    dataset = _dataset(tmp_path)
    predictions = tmp_path / "preds.jsonl"
    predictions.write_text(json.dumps({"instance_id": "o__r-1", "model_patch": "diff"}) + "\n")
    seen = {}

    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        seen["subset"] = pd.read_csv(kwargs["dataset_path"])["instance_id"].tolist()
        Path("result.json").write_text(
            json.dumps({"total_instances": 1, "total_resolved": 1, "resolved": ["o__r-1"], "patch_applied": ["o__r-1"]})
        )

    with patch("poly_bench_evaluation.run_evaluation.evaluate_predictions", fake_evaluate):
        summary = score_predictions(dataset, str(predictions), str(tmp_path / "eval"))

    assert seen["subset"] == ["o__r-1"]
    assert seen["evaluate_gold"] is False
    assert summary["resolve_rate"] == 1.0 and summary["resolved_ids"] == ["o__r-1"]
    assert (tmp_path / "eval" / "summary.json").exists()


def test_summarize_empty(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    assert summarize(tmp_path / "result.json")["resolve_rate"] == 0.0


def test_parse_issue_ref():
    assert parse_issue_ref("octo/cat#12") == ("octo", "cat", 12)
    assert parse_issue_ref("https://github.com/octo/cat.js/issues/7") == ("octo", "cat.js", 7)
    with pytest.raises(ValueError):
        parse_issue_ref("octo/cat")


def test_issue_from_github():
    calls = []

    def runner(args, cwd):
        calls.append(args)
        return json.dumps({"title": "Crash", "body": "Steps", "url": "https://github.com/o/r/issues/3"})

    issue = issue_from_github("o/r#3", runner)
    assert calls[0][:4] == ["gh", "issue", "view", "3"]
    assert issue.instance_id == "o__r-3"
    assert issue.problem_statement == "Crash\n\nSteps"
    assert issue.issue_url.endswith("/3")


def test_open_pull_request_commands(tmp_path):
    calls = []

    def runner(args, cwd):
        calls.append(args)
        return "https://github.com/o/r/pull/9\n" if args[:3] == ["gh", "pr", "create"] else ""

    patch_file = tmp_path / "patch.diff"
    patch_file.write_text("diff")
    result = FactoryResult("o__r-3", "verified", "diff", pr_title="Fix: crash", pr_body="body")

    url = open_pull_request("/repo", result, str(patch_file), base_branch="main", runner=runner)

    assert url == "https://github.com/o/r/pull/9"
    assert [c[:2] for c in calls] == [
        ["git", "stash"], ["git", "checkout"], ["git", "apply"], ["git", "commit"], ["git", "push"], ["gh", "pr"]
    ]
    assert calls[1][-1] == "software-factory/o__r-3"
    assert calls[-1][-3:] == ["--base", "main", "--draft"]
    assert (tmp_path / "pr_body.md").read_text() == "body"


def test_cli_run_with_mock_llm(tmp_path, git_repo):
    dataset = _dataset(tmp_path)
    from software_factory.workspace import LocalWorkspace

    with patch.object(cli, "DockerWorkspace", lambda issue, **kw: LocalWorkspace(str(git_repo))):
        code = cli.main(
            ["run", "--llm", "mock", "--dataset", dataset, "--limit", "2", "--output-dir", str(tmp_path / "out")]
        )
    assert code == 0
    rows = (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()
    assert [json.loads(r)["instance_id"] for r in rows] == ["o__r-0", "o__r-1"]
    assert (tmp_path / "out" / "artifacts" / "o__r-0" / "result.json").exists()
