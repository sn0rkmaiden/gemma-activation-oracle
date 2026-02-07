from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


AMBik_REPO = "https://github.com/cog-model/AmbiK-dataset"


def ensure_ambik(local_dir: str = "AmbiK-dataset") -> str:
    """Clone AmbiK dataset repo if missing. Returns dataset folder path."""
    if not os.path.exists(local_dir):
        subprocess.check_call(["git", "clone", AMBik_REPO, local_dir])
    data_dir = os.path.join(local_dir, "ambik_dataset")
    if not os.path.exists(data_dir):
        # some clones may nest differently
        data_dir = os.path.join(local_dir, "ambik_dataset")
    return data_dir


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()) for c in df.columns]
    return df


def _pick(df: pd.DataFrame, *cands: str) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"None of {cands} found. Columns: {df.columns.tolist()}")


def load_ambik_test900(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "ambik_test_900.csv")
    df = pd.read_csv(path)
    return _norm_cols(df)


def build_train_dev_examples(
    df: pd.DataFrame,
    seed: int = 42,
    dev_size: float = 0.2,
    neg_label: str = "NO_QUESTION",
) -> Tuple[List[Dict], List[Dict]]:
    """Build examples from AmbiK rows; each row yields (ambiguous, unambiguous)."""
    C_ENV = _pick(df, "environment_short", "environment_full", "environment")
    C_AMB = _pick(df, "ambiguous_task", "ambiguous")
    C_UNA = _pick(df, "unambiguous_direct", "unambiguous_indirect", "unambiguous")
    C_Q   = _pick(df, "question", "clarifying_question")

    train_rows, dev_rows = train_test_split(df, test_size=dev_size, random_state=seed, shuffle=True)

    def make_target_prompt(env: str, task: str) -> str:
        return f"Environment: {env}\nUser instruction: {task}\n"

    def build(rows_df: pd.DataFrame):
        ex = []
        for _, r in rows_df.iterrows():
            env = str(r[C_ENV]).strip()
            amb = str(r[C_AMB]).strip()
            una = str(r[C_UNA]).strip()
            q   = str(r[C_Q]).strip()

            ex.append({"target_text": make_target_prompt(env, amb), "label_text": q or "What should I clarify?", "is_amb": True})
            ex.append({"target_text": make_target_prompt(env, una), "label_text": neg_label, "is_amb": False})
        return ex

    return build(train_rows), build(dev_rows)
