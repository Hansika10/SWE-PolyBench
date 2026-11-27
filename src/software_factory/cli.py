import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from software_factory.intake import DEFAULT_DATASET, load_issues
from software_factory.llm import BedrockClient, MockLLM
from software_factory.pipeline import FactoryConfig, SoftwareFactory
from software_factory.workspace import DockerWorkspace, LocalWorkspace

DEFAULT_REPAIR_PACKAGES = ["hnswlib", "chroma-hnswlib"]


def _build_llm(args):
    if args.llm == "mock":
        return MockLLM()
    return BedrockClient(model_id=args.model_id, region=args.region, max_tokens=args.max_tokens)


def _config(args) -> FactoryConfig:
    return FactoryConfig(
        max_plan_steps=args.max_plan_steps,
        max_code_steps=args.max_code_steps,
        max_verify_steps=args.max_verify_steps,
        max_iterations=args.max_iterations,
        keep_test_changes=args.keep_test_changes,
    )


def _repair(args) -> Optional[List[str]]:
    if args.repair_native_imports is None:
        return None
    return args.repair_native_imports or DEFAULT_REPAIR_PACKAGES


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", choices=["bedrock", "mock"], default="bedrock")
    p.add_argument("--model-id", help="Bedrock model or inference profile id (env: FACTORY_BEDROCK_MODEL_ID)")
    p.add_argument("--region", help="AWS region (env: AWS_REGION)")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-iterations", type=int, default=2, help="code/verify rounds")
    p.add_argument("--max-plan-steps", type=int, default=25)
    p.add_argument("--max-code-steps", type=int, default=40)
    p.add_argument("--max-verify-steps", type=int, default=30)
    p.add_argument("--keep-test-changes", action="store_true", help="keep agent-written tests in the patch")
    p.add_argument("--output-dir", help="default: factory_runs/<timestamp>")


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-threads", type=int, default=1)
    p.add_argument("--repo-path", default="~/polybench_repos")
    p.add_argument("--delete-image", action="store_true")
    p.add_argument("--repair-native-imports", nargs="*", metavar="PACKAGE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="software-factory", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Resolve SWE-PolyBench instances and write predictions.jsonl")
    run.add_argument("--dataset", default=DEFAULT_DATASET)
    run.add_argument("--instance-ids", nargs="+")
    run.add_argument("--languages", nargs="+")
    run.add_argument("--per-language", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--include-hints", action="store_true")
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--score", action="store_true", help="score predictions with the official evaluator")
    _add_agent_args(run)
    _add_eval_args(run)

    score = sub.add_parser("score", help="Score a predictions.jsonl with the official evaluator")
    score.add_argument("--dataset", default=DEFAULT_DATASET)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output-dir", required=True)
    _add_eval_args(score)

    issue = sub.add_parser("issue", help="Resolve a live GitHub issue (owner/repo#123) and open a PR")
    issue.add_argument("ref")
    issue.add_argument("--repo-dir", help="existing checkout; cloned with gh if omitted")
    issue.add_argument("--open-pr", action="store_true")
    issue.add_argument("--base-branch")
    issue.add_argument("--ready", action="store_true", help="open a ready-for-review PR instead of a draft")
    _add_agent_args(issue)
    return parser


def _output_dir(args) -> Path:
    path = Path(args.output_dir or f"factory_runs/{datetime.now():%Y%m%d-%H%M%S}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_run(args) -> int:
    from software_factory.scoring import score_predictions

    out = _output_dir(args)
    issues = load_issues(
        args.dataset, args.instance_ids, args.languages, args.per_language, args.limit, args.include_hints
    )
    logger.info(f"Running {len(issues)} instance(s); outputs in {out}")
    repair = _repair(args)
    factory = SoftwareFactory(
        llm=_build_llm(args),
        workspace_factory=lambda issue: DockerWorkspace(
            issue,
            delete_image=args.delete_image,
            repo_path=args.repo_path,
            repair_native_packages=repair,
        ),
        config=_config(args),
        artifacts_dir=str(out / "artifacts"),
    )
    predictions = out / "predictions.jsonl"
    results = factory.run_batch(issues, str(predictions), skip_existing=args.skip_existing)
    statuses = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    print(json.dumps({"predictions": str(predictions), "statuses": statuses}, indent=2))

    if args.score:
        summary = score_predictions(
            args.dataset,
            str(predictions),
            str(out / "eval"),
            num_threads=args.eval_threads,
            repo_path=args.repo_path,
            delete_image=args.delete_image,
            repair_native_packages=repair,
        )
        print(json.dumps(summary, indent=2))
    return 0


def cmd_score(args) -> int:
    from software_factory.scoring import score_predictions

    summary = score_predictions(
        args.dataset,
        args.predictions,
        args.output_dir,
        num_threads=args.eval_threads,
        repo_path=args.repo_path,
        delete_image=args.delete_image,
        repair_native_packages=_repair(args),
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_issue(args) -> int:
    from software_factory.github import clone_repo, issue_from_github, open_pull_request

    out = _output_dir(args)
    issue = issue_from_github(args.ref)
    repo_dir = args.repo_dir or clone_repo(issue.repo, str(out / "repo"))
    factory = SoftwareFactory(
        llm=_build_llm(args),
        workspace_factory=lambda _: LocalWorkspace(repo_dir),
        config=_config(args),
        artifacts_dir=str(out / "artifacts"),
    )
    result = factory.run(issue)
    report = {"status": result.status, "artifacts": str(out / "artifacts" / issue.instance_id)}
    if args.open_pr:
        if not result.model_patch.strip():
            logger.error("No patch produced; not opening a PR.")
            print(json.dumps(report, indent=2))
            return 1
        patch_file = out / "artifacts" / issue.instance_id / "patch.diff"
        report["pr_url"] = open_pull_request(
            repo_dir, result, str(patch_file), args.base_branch, draft=not args.ready
        )
    print(json.dumps(report, indent=2))
    return 0 if result.status != "error" else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"run": cmd_run, "score": cmd_score, "issue": cmd_issue}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
