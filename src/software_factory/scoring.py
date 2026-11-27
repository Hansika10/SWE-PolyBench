import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from software_factory.intake import load_dataset_frame


@contextmanager
def _chdir(path: Path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def score_predictions(
    dataset: str,
    predictions_path: str,
    output_dir: str,
    instance_ids: Optional[Iterable[str]] = None,
    num_threads: int = 1,
    repo_path: str = "~/polybench_repos",
    delete_image: bool = False,
    repair_native_packages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Score predictions with the official SWE-PolyBench evaluator.

    Only instances present in the predictions (or ``instance_ids``) are evaluated, and all
    evaluator outputs (result.json, run logs) are written under ``output_dir``.
    """
    from poly_bench_evaluation.run_evaluation import evaluate_predictions

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    predictions_path = str(Path(predictions_path).resolve())
    predictions = pd.read_json(predictions_path, lines=True, dtype=False)
    ids = list(instance_ids) if instance_ids else predictions["instance_id"].tolist()

    df = load_dataset_frame(dataset)
    subset = df[df["instance_id"].isin(ids)]
    subset_path = out / "dataset_subset.csv"
    subset.to_csv(subset_path, index=False)

    instance_results = out / "instances"
    instance_results.mkdir(exist_ok=True)
    with _chdir(out):
        evaluate_predictions(
            dataset_path=str(subset_path),
            predictions_path=predictions_path,
            result_path=str(instance_results),
            num_threads=num_threads,
            evaluate_gold=False,
            repo_path=os.path.expanduser(repo_path),
            delete_image=delete_image,
            skip_existing=False,
            repair_native_packages=repair_native_packages,
        )
    return summarize(out / "result.json")


def summarize(result_json: Path) -> Dict[str, Any]:
    data = json.loads(Path(result_json).read_text())
    total = data.get("total_instances", 0)
    resolved = data.get("total_resolved", 0)
    summary = {
        "total": total,
        "resolved": resolved,
        "resolve_rate": round(resolved / total, 4) if total else 0.0,
        "patch_applied": len(data.get("patch_applied", [])),
        "resolved_ids": data.get("resolved", []),
        "file_retrieval": data.get("file_retrieval"),
    }
    Path(result_json).with_name("summary.json").write_text(json.dumps(summary, indent=2))
    return summary
