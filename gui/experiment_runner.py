"""Experiment runner used by the GUI data collection tab."""
from __future__ import annotations

import subprocess
import sys


def run_experiment(subject_id: str, num_images: int, min_dur: float, max_dur: float) -> None:
    """Run the real experiment for ``subject_id`` in a separate process."""

    # ``capture.experiment`` relies on pygame which fails when executed from a
    # non-main thread. Running the module via ``subprocess`` avoids this
    # limitation while still letting the GUI remain responsive.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "capture.experiment",
            "--subject",
            subject_id,
            "--num-images",
            str(num_images),
            "--min",
            str(min_dur),
            "--max",
            str(max_dur),
        ],
        check=True,
    )

