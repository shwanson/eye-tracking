from __future__ import annotations

"""Data collection tab for running live experiments."""

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
)

from .experiment_runner import run_experiment
import traceback


class ExperimentThread(QThread):
    """Thread wrapper to run the experiment without blocking the GUI."""

    finished_signal = Signal(str)

    def __init__(
        self,
        subject_id: str,
        num_images: int,
        min_dur: float,
        max_dur: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.subject_id = subject_id
        self.num_images = num_images
        self.min_dur = min_dur
        self.max_dur = max_dur

    def run(self) -> None:  # type: ignore[override]
        msg = ""
        try:
            run_experiment(
                self.subject_id,
                self.num_images,
                self.min_dur,
                self.max_dur,
            )
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

        layout.addWidget(QLabel("Number of Images:"))
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 1000)
        self.num_spin.setValue(1)
        layout.addWidget(self.num_spin)

        layout.addWidget(QLabel("Min Duration (s):"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0.1, 60.0)
        self.min_spin.setSingleStep(0.1)
        self.min_spin.setValue(1.0)
        layout.addWidget(self.min_spin)

        layout.addWidget(QLabel("Max Duration (s):"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0.1, 60.0)
        self.max_spin.setSingleStep(0.1)
        self.max_spin.setValue(3.0)
        layout.addWidget(self.max_spin)

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

        num_images = self.num_spin.value()
        min_dur = self.min_spin.value()
        max_dur = self.max_spin.value()
        self._thread = ExperimentThread(subject_id, num_images, min_dur, max_dur)
        self._thread.finished_signal.connect(self._on_finished)
        self._thread.start()

    @Slot(str)
    def _on_finished(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.status_label.setText(message)
        self._thread = None
