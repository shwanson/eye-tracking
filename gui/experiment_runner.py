"""Minimal experiment runner used for the GUI data collection tab."""
from __future__ import annotations

import time


def run_experiment(subject_id: str) -> None:
    """Simulate running an eye tracking experiment for ``subject_id``.

    This placeholder simply sleeps for a short duration but can be
    replaced with the actual experiment control logic.
    """
    print(f"Running experiment for {subject_id}")
    # Simulate a delay
    time.sleep(2)
    print("Experiment finished")
