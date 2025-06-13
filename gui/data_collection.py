from __future__ import annotations

"""Data collection tab for running live experiments."""

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
)

from .experiment_runner import run_experiment
import traceback


class ExperimentThread(QThread):
    """Thread wrapper to run the experiment without blocking the GUI."""

    finished_signal = Signal(str)

    def __init__(self, subject_id: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.subject_id = subject_id

    def run(self) -> None:  # type: ignore[override]
        msg = ""
        try:
            run_experiment(self.subject_id)
            msg = "Study finished"
        except Exception as e:  # pragma: no cover - simple passthrough
            msg = f"Error: {e}"
            traceback.print_exc()
        self.finished_signal.emit(msg)


class DataCollectionTab(QWidget):
    """Tab for starting a new data collection study."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Subject ID:"))

        self.subject_edit = QLineEdit()
        layout.addWidget(self.subject_edit)

        self.start_button = QPushButton("Start Study")
        layout.addWidget(self.start_button)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.start_button.clicked.connect(self.start_study)
        self._thread: ExperimentThread | None = None

    @Slot()
    def start_study(self) -> None:
        """Launch the experiment in a background thread."""
        subject_id = self.subject_edit.text().strip()
        if not subject_id:
            self.status_label.setText("Enter a subject ID")
            return

        self.start_button.setEnabled(False)
        self.status_label.setText("Running...")

        self._thread = ExperimentThread(subject_id)
        self._thread.finished_signal.connect(self._on_finished)
        self._thread.start()

    @Slot(str)
    def _on_finished(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.status_label.setText(message)
        self._thread = None
