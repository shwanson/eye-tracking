"""Experiment runner used by the GUI data collection tab."""
from __future__ import annotations

from capture.experiment import run_experiment as capture_run_experiment


def run_experiment(subject_id: str) -> None:
    """Run the real experiment for ``subject_id``.

    This simply delegates to :func:`capture.experiment.run_experiment` so that
    invoking the "Start Study" button from the GUI launches the full pygame
    window and recording logic.
    """

    capture_run_experiment(subject_id)

