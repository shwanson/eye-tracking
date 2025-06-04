import pytest

pd = pytest.importorskip("pandas")
from analysis.run import run_analysis

def test_run_analysis_basic(tmp_path):
    df = pd.DataFrame({
        'subject': ['S1', 'S1', 'S2'],
        'stimulus': ['A', 'B', 'A'],
        'x': [0.1, 0.2, 0.3],
        'y': [0.1, 0.2, 0.3],
    })
    out_dir = tmp_path / "out"
    metrics = run_analysis(df, output_dir=str(out_dir), generate_visualizations=False)
    assert not metrics.empty
    assert set(['subject', 'stimulus', 'n_fixations', 'dwell_prop', 'ttf_ms']).issubset(metrics.columns)


def test_run_analysis_group(tmp_path):
    df = pd.DataFrame({
        'subject': ['S1', 'S1', 'S2'],
        'stimulus': ['A', 'A', 'A'],
        'group': ['G1', 'G1', 'G2']
    })
    out_dir = tmp_path / "out"
    metrics = run_analysis(df, output_dir=str(out_dir), group_var='group', generate_visualizations=False)
    group_file = out_dir / "group_analysis" / "aggregated_metrics.csv"
    assert group_file.exists()
    assert not metrics.empty
