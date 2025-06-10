import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List


def all_metrics(fixations: pd.DataFrame, trial_durations: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Calculate basic fixation metrics for each subject and stimulus.

    Parameters
    ----------
    fixations : pd.DataFrame
        DataFrame containing fixation data with columns ``subject``, ``stimulus``,
        ``start_s`` and ``duration_s``.
    trial_durations : Optional[Dict[str, float]], optional
        Mapping of stimulus id to trial duration in seconds. If not provided,
        the duration is inferred from the fixation timings.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics ``n_fixations``, ``mean_fix_dur_ms``, ``dwell_prop``
        and ``ttf_ms`` for each subject/stimulus pair.
    """
    if fixations.empty:
        return pd.DataFrame(columns=[
            "subject",
            "stimulus",
            "n_fixations",
            "mean_fix_dur_ms",
            "dwell_prop",
            "ttf_ms",
        ])

    metrics = []
    grouped = fixations.groupby(["subject", "stimulus"])

    for (subject, stimulus), group in grouped:
        n_fix = len(group)
        mean_dur_ms = group["duration_s"].mean() * 1000.0
        dwell_time = group["duration_s"].sum()

        if trial_durations and stimulus in trial_durations:
            trial_dur = trial_durations[stimulus]
        else:
            trial_dur = group["end_s"].max()
        dwell_prop = dwell_time / trial_dur if trial_dur > 0 else np.nan

        ttf_ms = group["start_s"].min() * 1000.0

        metrics.append({
            "subject": subject,
            "stimulus": stimulus,
            "n_fixations": int(n_fix),
            "mean_fix_dur_ms": mean_dur_ms,
            "dwell_prop": dwell_prop,
            "ttf_ms": ttf_ms,
        })

    return pd.DataFrame(metrics)


def transition_matrix(fixations: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute transitions between stimuli for a sequence of fixations.

    Parameters
    ----------
    fixations : pd.DataFrame
        Fixation data containing at least ``stimulus`` and ``start_s`` columns.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Transition count matrix and the ordered list of stimuli.
    """
    if fixations.empty or "stimulus" not in fixations.columns:
        return pd.DataFrame(), []

    ordered = fixations.sort_values("start_s")
    labels = list(ordered["stimulus"].unique())
    index = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)

    prev = ordered.iloc[0]["stimulus"]
    for stim in ordered["stimulus"].iloc[1:]:
        mat[index[prev], index[stim]] += 1
        prev = stim

    matrix_df = pd.DataFrame(mat, index=labels, columns=labels)
    return matrix_df, labels
