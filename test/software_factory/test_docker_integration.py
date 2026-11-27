"""Opt-in harness check against a real SWE-PolyBench container and the official evaluator.

Uses an "oracle" mock LLM that applies the gold patch through the agent tools, so a correct
pipeline must produce a resolved instance. Run with: FACTORY_DOCKER_TESTS=1 pytest -k docker
"""

import os

import pytest

from software_factory.intake import load_dataset_frame, issue_from_row
from software_factory.llm import MockLLM, tool_call
from software_factory.models import LLMResponse
from software_factory.pipeline import SoftwareFactory
from software_factory.scoring import score_predictions
from software_factory.workspace import DockerWorkspace

INSTANCE_ID = os.environ.get("FACTORY_DOCKER_INSTANCE", "yt-dlp__yt-dlp-4841")
DATASET = os.environ.get("FACTORY_DOCKER_DATASET", "AmazonScience/SWE-PolyBench_Verified")

pytestmark = pytest.mark.skipif(
    os.environ.get("FACTORY_DOCKER_TESTS") != "1", reason="set FACTORY_DOCKER_TESTS=1 to run"
)


def oracle(gold_patch: str):
    def respond(system, messages, tools):
        names = {t["name"] for t in tools}
        started = any("toolResult" in b for b in messages[-1]["content"])
        if "submit_plan" in names:
            return tool_call("submit_plan", summary="oracle", files=[], steps=[])
        if "submit_changes" in names:
            if not started:
                return tool_call("write_file", path=".factory_scratch/gold.diff", content=gold_patch)
            if len(messages) == 3:
                return tool_call("bash", command="git apply .factory_scratch/gold.diff")
            return tool_call("submit_changes", summary="applied")
        if "submit_verification" in names:
            return tool_call("submit_verification", passed=True, summary="oracle", commands=[])
        return LLMResponse(text="")

    return respond


def test_docker_pipeline_resolves_with_oracle(tmp_path):
    df = load_dataset_frame(DATASET)
    row = df[df["instance_id"] == INSTANCE_ID].iloc[0]
    factory = SoftwareFactory(
        MockLLM(oracle(row["patch"]), model_name="oracle"),
        lambda issue: DockerWorkspace(issue),
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    predictions = tmp_path / "predictions.jsonl"

    results = factory.run_batch([issue_from_row(row)], str(predictions))

    assert results[0].status == "verified", results[0].error
    assert ".factory_scratch" not in results[0].model_patch
    summary = score_predictions(DATASET, str(predictions), str(tmp_path / "eval"))
    assert summary["resolved_ids"] == [INSTANCE_ID]
