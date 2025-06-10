"""
Group-level analyses and statistics for eye tracking data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import statsmodels.formula.api as smf
from scipy import stats

# Remove all AOI-based group analysis and references. Leave file empty if all group analysis is AOI-based. 

def aggregate_by_group(df: pd.DataFrame, group_var: str, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate metrics by group variable (mean and SEM).
    """
    if metrics is None:
        metrics = [col for col in df.columns if col not in [group_var, 'subject', 'stimulus']]
    grouped = df.groupby(group_var)[metrics]
    means = grouped.mean().add_suffix('_mean')
    sems = grouped.sem().add_suffix('_sem')
    result = pd.concat([means, sems], axis=1).reset_index()
    return result

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_std


def _eta_squared(groups: List[np.ndarray]) -> float:
    """Calculate eta-squared for one-way ANOVA."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def _bootstrap_ci(func, data: List[np.ndarray], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval for the given statistic."""
    stats_bs = []
    for _ in range(n_boot):
        samples = [np.random.choice(d, size=len(d), replace=True) for d in data]
        stats_bs.append(func(*samples))
    lower = np.percentile(stats_bs, (1 - ci) / 2 * 100)
    upper = np.percentile(stats_bs, (1 + ci) / 2 * 100)
    return lower, upper


def compare_groups(df: pd.DataFrame, group_var: str, metric: str, ci: bool = False) -> pd.DataFrame:
    """Compare groups for a given metric using ANOVA or t-test.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_var : str
        Column denoting group membership.
    metric : str
        Metric column to compare.
    ci : bool, optional
        If True, bootstrap 95% confidence interval for the effect size.
    """

    groups = df[group_var].dropna().unique()
    data = [df[df[group_var] == g][metric].dropna().to_numpy() for g in groups]

    if len(groups) == 2:
        stat, p = stats.ttest_ind(data[0], data[1], equal_var=False)
        effect = _cohens_d(data[0], data[1])
        test = "t-test"
        ci_low = ci_high = np.nan
        if ci:
            ci_low, ci_high = _bootstrap_ci(_cohens_d, data)
    else:
        stat, p = stats.f_oneway(*data)
        effect = _eta_squared(data)
        test = "ANOVA"
        ci_low = ci_high = np.nan
        if ci:
            ci_low, ci_high = _bootstrap_ci(lambda *d: _eta_squared(list(d)), data)

    result = pd.DataFrame({
        "test": [test],
        "statistic": [stat],
        "p_value": [p],
        "effect_size": [effect],
        "ci_lower": [ci_low],
        "ci_upper": [ci_high],
        "groups": [str(groups)],
    })
    return result
