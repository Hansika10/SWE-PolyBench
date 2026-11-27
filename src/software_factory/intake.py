import math
from typing import Iterable, List, Optional

import pandas as pd

from software_factory.models import Issue

DEFAULT_DATASET = "AmazonScience/SWE-PolyBench_Verified"


def load_dataset_frame(dataset: str) -> pd.DataFrame:
    """Load a SWE-PolyBench split from a .csv/.jsonl/.json file or a Hugging Face dataset id."""
    if dataset.endswith(".csv"):
        return pd.read_csv(dataset)
    if dataset.endswith(".jsonl"):
        return pd.read_json(dataset, lines=True, dtype=False)
    if dataset.endswith(".json"):
        return pd.read_json(dataset, dtype=False)
    from datasets import load_dataset

    return load_dataset(dataset, split="test").to_pandas()


def select_instances(
    df: pd.DataFrame,
    instance_ids: Optional[Iterable[str]] = None,
    languages: Optional[Iterable[str]] = None,
    per_language: Optional[int] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    if instance_ids:
        wanted = list(instance_ids)
        missing = set(wanted) - set(df["instance_id"])
        if missing:
            raise ValueError(f"Unknown instance ids: {sorted(missing)}")
        df = df[df["instance_id"].isin(wanted)]
    if languages:
        lowered = {lang.lower() for lang in languages}
        df = df[df["language"].str.lower().isin(lowered)]
    if per_language:
        df = df.groupby("language", sort=False).head(per_language)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def issue_from_row(row, include_hints: bool = False) -> Issue:
    """Only agent-safe fields are copied: no gold patch, test patch, F2P/P2P or test command."""
    return Issue(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=_text(row.get("base_commit")),
        language=_text(row.get("language")),
        problem_statement=_text(row.get("problem_statement")),
        hints_text=_text(row.get("hints_text")) if include_hints else "",
        dockerfile=_text(row.get("Dockerfile")),
    )


def load_issues(
    dataset: str,
    instance_ids: Optional[Iterable[str]] = None,
    languages: Optional[Iterable[str]] = None,
    per_language: Optional[int] = None,
    limit: Optional[int] = None,
    include_hints: bool = False,
) -> List[Issue]:
    df = select_instances(load_dataset_frame(dataset), instance_ids, languages, per_language, limit)
    return [issue_from_row(row, include_hints) for _, row in df.iterrows()]
