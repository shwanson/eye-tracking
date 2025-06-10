"""
Eye-tracking data viewer application.
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import json

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QComboBox, QFileDialog, 
    QTableView, QSplitter, QMessageBox, QGroupBox, QFormLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox, QToolBar, QStatusBar
)
from PySide6.QtCore import Qt, QSortFilterProxyModel, Signal, Slot, QSize, QModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem, QAction, QIcon

# Import ETL and analysis modules
from etl.io import load_all, load_subject
from etl.preprocess import preprocess_pipeline
from analysis.viz import (
    plot_gaze_timeseries, plot_heatmap, plot_scanpath, plot_group_heatmap
)


class PlotTab(QWidget):
    """Widget for displaying matplotlib figures in a tab."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.layout = QVBoxLayout(self)
        self.canvas = None
        self.figure = None
    
    def set_figure(self, figure: Figure):
        """Set the matplotlib figure to display."""
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        
        self.figure = figure
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)


class DataTableModel(QStandardItemModel):
    """Model for displaying pandas DataFrame data in a QTableView."""
    
    def __init__(self, data: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self.df = pd.DataFrame() if data is None else data
        self.update_model()
    
    def update_data(self, data: pd.DataFrame):
        """Update the model with new data."""
        self.df = data
        self.update_model()
    
    def update_model(self):
        """Rebuild the model from the DataFrame."""
        self.clear()
        if self.df.empty:
            return
        
        # Set headers
        self.setColumnCount(len(self.df.columns))
        self.setHorizontalHeaderLabels(self.df.columns)
        
        # Set data
        self.setRowCount(len(self.df))
        for row in range(len(self.df)):
            for col in range(len(self.df.columns)):
                value = str(self.df.iloc[row, col])
                item = QStandardItem(value)
                self.setItem(row, col, item)


class MainWindow(QMainWindow):
    """Main window for the eye tracking viewer application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking Viewer")
        self.resize(1200, 800)
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.fixations = None
        self.metrics = None
        self.current_subject = None
        self.current_stimulus = None
        
        # UI setup
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Create filters section
        self.create_filters()
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create initial tabs
        self.create_initial_tabs()
    
    def create_toolbar(self):
        """Create the application toolbar."""
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)
        
        # Add actions
        self.load_action = QAction("Load Data", self)
        self.load_action.triggered.connect(self.load_data_dialog)
        self.toolbar.addAction(self.load_action)
        
        self.toolbar.addSeparator()
        
        self.export_action = QAction("Export Results", self)
        self.export_action.triggered.connect(self.export_dialog)
        self.toolbar.addAction(self.export_action)
    
    def create_filters(self):
        """Create the filter controls."""
        filter_layout = QHBoxLayout()
        
        # Subject filter
        self.subject_combo = QComboBox()
        self.subject_combo.setMinimumWidth(150)
        self.subject_combo.currentIndexChanged.connect(self.on_subject_changed)
        
        # Stimulus filter
        self.stimulus_combo = QComboBox()
        self.stimulus_combo.setMinimumWidth(150)
        self.stimulus_combo.currentIndexChanged.connect(self.on_stimulus_changed)
        
        # Add to layout
        filter_layout.addWidget(QLabel("Subject:"))
        filter_layout.addWidget(self.subject_combo)
        filter_layout.addWidget(QLabel("Stimulus:"))
        filter_layout.addWidget(self.stimulus_combo)
        filter_layout.addStretch(1)
        

        
        self.main_layout.addLayout(filter_layout)
    
    def create_initial_tabs(self):
        """Create the initial empty tabs."""
        # Timeseries tab
        self.timeseries_tab = PlotTab("Time Series")
        self.tabs.addTab(self.timeseries_tab, "Time Series")
        
        # Heatmap tab
        self.heatmap_tab = PlotTab("Heatmap")
        self.tabs.addTab(self.heatmap_tab, "Heatmap")
        
        # Scanpath tab
        self.scanpath_tab = PlotTab("Scanpath")
        self.tabs.addTab(self.scanpath_tab, "Scanpath")
        
        # Create group analysis tab
        self.group_tab = self.create_group_tab()
        self.tabs.addTab(self.group_tab, "Group Analysis")
        
        # Create metrics tab
        self.metrics_tab = self.create_metrics_tab()
        self.tabs.addTab(self.metrics_tab, "Metrics")
    
    def create_group_tab(self) -> QWidget:
        """Create the group analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Group variable selection
        group_box = QGroupBox("Group Analysis")
        group_layout = QFormLayout(group_box)
        
        self.group_var_combo = QComboBox()
        self.group_var_combo.addItem("None")
        group_layout.addRow("Group Variable:", self.group_var_combo)
        
        self.metric_combo = QComboBox()
        for metric in ["dwell_prop", "ttf_ms", "n_fixations"]:
            self.metric_combo.addItem(metric)
        group_layout.addRow("Metric:", self.metric_combo)
        
        run_analysis_btn = QPushButton("Run Analysis")
        run_analysis_btn.clicked.connect(self.run_group_analysis)
        group_layout.addRow("", run_analysis_btn)
        
        controls_layout.addWidget(group_box)
        
        # Visualization controls
        viz_box = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_box)
        
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(50, 500)
        self.bins_spin.setValue(200)
        self.bins_spin.setSingleStep(50)
        viz_layout.addRow("Bins:", self.bins_spin)
        
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 30)
        self.sigma_spin.setValue(10)
        viz_layout.addRow("Sigma:", self.sigma_spin)
        
        controls_layout.addWidget(viz_box)
        layout.addLayout(controls_layout)
        
        # Split view: plot and results
        splitter = QSplitter(Qt.Vertical)
        
        # Plot area
        self.group_plot = PlotTab("Group Comparison")
        splitter.addWidget(self.group_plot)
        
        # Results table
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        
        self.results_table = QTableView()
        self.results_model = DataTableModel()
        self.results_table.setModel(self.results_model)
        
        results_layout.addWidget(QLabel("Statistical Results:"))
        results_layout.addWidget(self.results_table)
        
        splitter.addWidget(results_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 200])
        
        layout.addWidget(splitter)
        
        return tab
    
    def create_metrics_tab(self) -> QWidget:
        """Create the metrics visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Metrics selection
        metrics_box = QGroupBox("Metrics")
        metrics_layout = QFormLayout(metrics_box)
        
        self.metric_type_combo = QComboBox()
        for metric in ["dwell_prop", "ttf_ms", "n_fixations", "transitions"]:
            self.metric_type_combo.addItem(metric)
        metrics_layout.addRow("Metric:", self.metric_type_combo)
        
        self.by_subject_check = QCheckBox("Group by Subject")
        metrics_layout.addRow("", self.by_subject_check)
        
        controls_layout.addWidget(metrics_box)
        
        # Plot area
        self.metrics_plot = PlotTab("Metrics")
        layout.addWidget(self.metrics_plot)
        
        return tab
    
    def load_data_dialog(self):
        """Show dialog to load eye tracking data."""
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("CSV Files (*.csv);;All Files (*)")
        dialog.setDirectory(str(Path.cwd() / "data"))
        
        if dialog.exec():
            file_paths = dialog.selectedFiles()
            if file_paths:
                self.load_data(file_paths)
    
    def export_dialog(self):
        """Show dialog to export results."""
        if self.fixations is None or self.metrics is None:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", str(Path.cwd()),
            QFileDialog.ShowDirsOnly
        )
        
        if output_dir:
            self.export_results(output_dir)
    
    def load_data(self, file_paths: List[str]):
        """Load eye tracking data from the selected files."""
        self.statusBar.showMessage("Loading data...")
        
        try:
            # Get screen size
            screen_w, screen_h = 1920, 1080  # Default
            
            # If files are in the same directory, use load_all
            if len(set(Path(f).parent for f in file_paths)) == 1:
                data_dir = Path(file_paths[0]).parent
                self.raw_data = load_all(data_dir, (screen_w, screen_h), parallel=True)
            else:
                # Load files individually and concatenate
                dfs = []
                for path in file_paths:
                    df = load_subject(path, (screen_w, screen_h))
                    dfs.append(df)
                self.raw_data = pd.concat(dfs, ignore_index=True)
            
            print('DEBUG: Raw data columns:', self.raw_data.columns.tolist())
            print('DEBUG: First few rows of raw data:', self.raw_data.head())
            
            # Preprocess data
            self.processed_data, self.fixations = preprocess_pipeline(self.raw_data)
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Update UI
            self.update_filters()
            self.update_plots()
            
            self.statusBar.showMessage(f"Loaded {len(file_paths)} files with {len(self.raw_data)} samples.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
            self.statusBar.showMessage("Error loading data.")
    
    def calculate_metrics(self):
        """Calculate metrics from fixation data."""
        if self.fixations is None:
            return
        # self.metrics = all_metrics(self.fixations)
        self.metrics = None  # No metrics available
    
    def update_filters(self):
        """Update the filter dropdowns with available subjects and stimuli."""
        self.subject_combo.blockSignals(True)
        self.stimulus_combo.blockSignals(True)
        
        self.subject_combo.clear()
        self.stimulus_combo.clear()
        
        if self.raw_data is not None:
            # Add "All" option
            self.subject_combo.addItem("All")
            self.stimulus_combo.addItem("All")
            
            # Add subjects
            subjects = sorted(self.raw_data['subject'].unique())
            for subject in subjects:
                self.subject_combo.addItem(str(subject))
            
            # Add stimuli
            stimuli = sorted(self.raw_data['stimulus'].unique())
            for stimulus in stimuli:
                self.stimulus_combo.addItem(str(stimulus))
            
            # Also update group variable selector
            self.update_group_variables()
        
        self.subject_combo.blockSignals(False)
        self.stimulus_combo.blockSignals(False)
    
    def update_group_variables(self):
        """Update the group variable dropdown with available columns."""
        self.group_var_combo.blockSignals(True)
        self.group_var_combo.clear()
        
        self.group_var_combo.addItem("None")
        self.group_var_combo.addItem("subject")
        
        if self.raw_data is not None:
            # Add other potential group variables
            for col in self.raw_data.columns:
                if col not in ['subject', 'stimulus', 'x', 'y', 'x_px', 'y_px', 'time_s']:
                    self.group_var_combo.addItem(col)
        
        self.group_var_combo.blockSignals(False)
    
    def on_subject_changed(self, index: int):
        """Handle subject selection changed."""
        if index == -1:
            return
        
        text = self.subject_combo.currentText()
        self.current_subject = None if text == "All" else text
        self.update_plots()
    
    def on_stimulus_changed(self, index: int):
        """Handle stimulus selection changed."""
        if index == -1:
            return
        
        text = self.stimulus_combo.currentText()
        self.current_stimulus = None if text == "All" else text
        self.update_plots()
    
    def get_filtered_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data filtered by current subject and stimulus selections."""
        if self.processed_data is None or self.fixations is None:
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter raw data
        filtered_raw = self.processed_data.copy()
        filtered_fix = self.fixations.copy()

        print('DEBUG: Unique subjects in fixations:', filtered_fix['subject'].unique())
        print('DEBUG: Unique stimuli in fixations:', filtered_fix['stimulus'].unique())
        print('DEBUG: Current subject filter:', self.current_subject)
        print('DEBUG: Current stimulus filter:', self.current_stimulus)

        if self.current_subject:
            filtered_raw = filtered_raw[filtered_raw['subject'].astype(str) == str(self.current_subject)]
            filtered_fix = filtered_fix[filtered_fix['subject'].astype(str) == str(self.current_subject)]
        if self.current_stimulus:
            filtered_raw = filtered_raw[filtered_raw['stimulus'].astype(str) == str(self.current_stimulus)]
            filtered_fix = filtered_fix[filtered_fix['stimulus'].astype(str) == str(self.current_stimulus)]

        print('DEBUG: Filtered fixations shape:', filtered_fix.shape)
        print('DEBUG: Filtered fixations head:', filtered_fix.head())

        # GUI warning if all x_px/y_px are NaN
        if not filtered_fix.empty and filtered_fix[['x_px', 'y_px']].dropna().empty:
            QMessageBox.warning(None, 'No Valid Fixations', 'All fixations for this selection have NaN coordinates and cannot be displayed.')
        
        return filtered_raw, filtered_fix
    
    def update_plots(self):
        """Update all plots with current data selections."""
        if self.processed_data is None:
            return
        
        self.statusBar.showMessage("Updating plots...")
        
        try:
            # Get filtered data
            raw_data, fixations = self.get_filtered_data()
            
            if raw_data.empty:
                self.statusBar.showMessage("No data for current selection.")
                return
            
            # Update timeseries plot
            fig = plot_gaze_timeseries(raw_data)
            self.timeseries_tab.set_figure(fig)
            
            # Update heatmap plot
            fig = plot_heatmap(raw_data)
            self.heatmap_tab.set_figure(fig)
            
            # Update scanpath plot
            print('DEBUG: Fixations DataFrame shape:', fixations.shape)
            print('DEBUG: Fixations columns:', fixations.columns.tolist())
            print('DEBUG: First few rows of fixations:', fixations.head())
            if fixations.empty or fixations[['x_px', 'y_px']].dropna().empty:
                self.statusBar.showMessage('No fixations to display for current selection.')
                QMessageBox.warning(self, 'No Scanpath', 'No fixations to display for current selection.')
                # Optionally clear the plot
                fig = Figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                ax.set_title('No Scanpath Data')
                self.scanpath_tab.set_figure(fig)
            else:
                fig = plot_scanpath(fixations)
                self.scanpath_tab.set_figure(fig)
            
            # Update metrics plot
            self.update_metrics_plot()
            
            self.statusBar.showMessage("Plots updated.")
        except Exception as e:
            self.statusBar.showMessage(f"Error updating plots: {str(e)}")
    
    def update_metrics_plot(self):
        """Update the metrics visualization tab."""
        if self.metrics is None:
            return
        # No metrics plotting available
        return
    
    def run_group_analysis(self):
        """Run group analysis and update the group tab."""
        if self.fixations is None or self.metrics is None:
            QMessageBox.warning(self, "No Data", "No data loaded for group analysis.")
            return

        group_var = self.group_var_combo.currentText()
        metric = self.metric_combo.currentText()
        bins = self.bins_spin.value()
        sigma = self.sigma_spin.value()

        if group_var == "None":
            QMessageBox.warning(self, "No Group Variable", "Please select a group variable.")
            return

        # Only use rows with valid group_var and metric
        df = self.metrics.copy()
        if group_var not in df.columns:
            if group_var in self.fixations.columns:
                # Try to merge group_var from fixations
                df = df.merge(self.fixations[["subject", group_var]].drop_duplicates(), on="subject", how="left")
            else:
                QMessageBox.warning(self, "Invalid Group Variable", f"Group variable '{group_var}' not found in data.")
                return
        if metric not in df.columns:
            QMessageBox.warning(self, "Invalid Metric", f"Metric '{metric}' not found in data.")
            return
        df = df.dropna(subset=[group_var, metric])
        if df.empty:
            QMessageBox.warning(self, "No Data", "No data available for selected group and metric.")
            return

        # Aggregate by group
        from analysis.group import aggregate_by_group, compare_groups
        agg = aggregate_by_group(df, group_var, [metric])
        comp = compare_groups(df, group_var, metric)

        # Update results table
        self.results_model.update_data(agg)

        # Plot group heatmap (if possible)
        from analysis.viz import plot_group_heatmap
        group_labels = list(df[group_var].dropna().unique())
        group_dfs = [self.fixations[self.fixations[group_var] == g] for g in group_labels]
        try:
            fig = plot_group_heatmap(group_dfs, group_labels, bins=bins, sigma=sigma)
            self.group_plot.set_figure(fig)
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Could not plot group heatmap: {str(e)}")

        # Show statistical results in a popup
        msg = f"Test: {comp['test'].iloc[0]}\nStatistic: {comp['statistic'].iloc[0]:.3f}\nP-value: {comp['p_value'].iloc[0]:.4g}\nGroups: {comp['groups'].iloc[0]}"
        QMessageBox.information(self, "Group Comparison Result", msg)
    
    def export_results(self, output_dir: str):
        """Export results to the specified directory."""
        if self.fixations is None or self.metrics is None:
            return
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export fixations
            self.fixations.to_csv(output_path / "fixations.csv", index=False)
            
            # Export metrics
            self.metrics.to_csv(output_path / "metrics.csv", index=False)
            
            # Export plots for current selection
            from analysis.viz import save_all_visualizations
            
            raw_data, fixations = self.get_filtered_data()
            filtered_metrics = self.metrics.copy()
            
            if self.current_subject:
                filtered_metrics = filtered_metrics[filtered_metrics['subject'] == self.current_subject]
            if self.current_stimulus:
                filtered_metrics = filtered_metrics[filtered_metrics['stimulus'] == self.current_stimulus]
            
            # Calculate transition matrix
            transitions, _ = transition_matrix(fixations)
            
            # Save visualizations
            viz_dir = output_path / "visualizations"
            save_all_visualizations(
                fixations, filtered_metrics, viz_dir,
                subject=self.current_subject,
                stimulus=self.current_stimulus,
                transition_matrix=transitions
            )
            
            self.statusBar.showMessage(f"Results exported to {output_dir}")
            QMessageBox.information(self, "Export Complete", f"Results exported to {output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
            self.statusBar.showMessage("Error exporting results.")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
