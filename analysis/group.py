"""
Group-level analyses and statistics for eye tracking data.
"""
import pandas as pd
from typing import List, Optional
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

def compare_groups(df: pd.DataFrame, group_var: str, metric: str) -> pd.DataFrame:
    """
    Compare groups for a given metric using ANOVA or t-test (if 2 groups).
    """
    groups = df[group_var].dropna().unique()
    data = [df[df[group_var] == g][metric].dropna() for g in groups]
    if len(groups) == 2:
        stat, p = stats.ttest_ind(data[0], data[1], equal_var=False)
        test = 't-test'
    else:
        stat, p = stats.f_oneway(*data)
        test = 'ANOVA'
    result = pd.DataFrame({
        'test': [test],
        'statistic': [stat],
        'p_value': [p],
        'groups': [str(groups)]
    })
    return result 
