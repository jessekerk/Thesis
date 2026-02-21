# data/pollution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


PollutionMode = Literal["empty", "nan", "mask_token"]


@dataclass(frozen=True)
class PollutionResult:
    df: pd.DataFrame
    polluted_indices: np.ndarray  # row indices (positional, not label-based)
    feature: str
    pollution_rate: float
    mode: PollutionMode


class DataPolluter:
    """
    Pollutes exactly ONE feature/column in a dataframe for a given percentage of rows.

    Intended use: simulate missing-feature scenarios by blanking out 'title' OR 'description'
    while leaving everything else untouched.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def pollute(
        self,
        df: pd.DataFrame,
        feature: str,
        pollution_rate: float,
        *,
        mode: PollutionMode = "empty",
        mask_token: str = "[MASK]",
        keep_index: bool = True,
    ) -> PollutionResult:
        """
        Args:
            df: input dataframe (will NOT be modified in-place).
            feature: column name to pollute (e.g., "title" or "description").
            pollution_rate: fraction in [0, 1], e.g., 0.1 = 10% of rows.
            mode:
              - "empty": replace polluted rows with "" (recommended for TF-IDF pipelines)
              - "nan": replace polluted rows with np.nan
              - "mask_token": replace polluted rows with a constant token, e.g. "[MASK]"
            mask_token: token used when mode="mask_token"
            keep_index: if True, preserve the original index; pollution uses positional rows.

        Returns:
            PollutionResult with a polluted copy of df and the positional row indices polluted.
        """
        if feature not in df.columns:
            raise ValueError(
                f"feature='{feature}' not found in df.columns={list(df.columns)}"
            )
        if not (0.0 <= pollution_rate <= 1.0):
            raise ValueError("pollution_rate must be in [0, 1].")

        n = len(df)
        k = int(round(n * pollution_rate))

        # Copy, but preserve index if requested
        out = df.copy(deep=True)
        if not keep_index:
            out = out.reset_index(drop=True)

        if k == 0:
            return PollutionResult(
                df=out,
                polluted_indices=np.array([], dtype=int),
                feature=feature,
                pollution_rate=pollution_rate,
                mode=mode,
            )

        polluted_pos = self.rng.choice(n, size=k, replace=False)
        polluted_pos.sort()

        if mode == "empty":
            out.iloc[polluted_pos, out.columns.get_loc(feature)] = ""
        elif mode == "nan":
            out.iloc[polluted_pos, out.columns.get_loc(feature)] = np.nan
        elif mode == "mask_token":
            out.iloc[polluted_pos, out.columns.get_loc(feature)] = mask_token
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return PollutionResult(
            df=out,
            polluted_indices=polluted_pos,
            feature=feature,
            pollution_rate=pollution_rate,
            mode=mode,
        )

    def pollute_splits(
        self,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        feature: str,
        pollution_rate: float,
        mode: PollutionMode = "empty",
        mask_token: str = "[MASK]",
    ) -> Tuple[PollutionResult, PollutionResult, PollutionResult]:
        """
        Convenience helper if you want consistent settings across splits (each split gets its own random rows).
        """
        r_train = self.pollute(
            train_df, feature, pollution_rate, mode=mode, mask_token=mask_token
        )
        r_dev = self.pollute(
            dev_df, feature, pollution_rate, mode=mode, mask_token=mask_token
        )
        r_test = self.pollute(
            test_df, feature, pollution_rate, mode=mode, mask_token=mask_token
        )
        return r_train, r_dev, r_test
