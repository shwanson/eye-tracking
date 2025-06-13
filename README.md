# Eye Tracking Analysis Framework

A comprehensive framework for processing, analyzing, and visualizing eye tracking data with support for multiple subjects and multiple stimuli.

## Project Structure

```
project/
│
├─ data/                    # Raw CSV files per subject and stimulus
│   └─ P01_face01.csv       # Naming format: SubjectID_StimulusID.csv
│
├─ etl/                     # Extract-Transform-Load modules
│   ├─ io.py                # Data loading and saving functions
│   ├─ preprocess.py        # Data filtering and fixation detection
│
├─ analysis/
│   ├─ metrics.py           # Metrics calculations (TTF, Dwell, etc.)
│   ├─ group.py             # Group-level analysis and statistics
│   └─ viz.py               # Visualization functions
│
├─ gui/                     # PySide6 GUI application
│   └─ viewer.py            # Main viewer interface
│
└─ notebooks/               # Jupyter notebooks for exploratory analysis
```

## Getting Started

### Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install pandas numpy scipy matplotlib seaborn shapely statsmodels pingouin PySide6

# Extra dependencies for data collection
pip install pygame tobii_research
```

### Running the GUI

```bash
python -m gui.viewer
```

### Data Format

The framework expects Tobii eye tracking data in CSV format. Files should be named using the pattern: `SubjectID_StimulusID.csv` (e.g., `P01_face01.csv`).

## Core Functionality

### ETL Pipeline

Run the ETL (Extract-Transform-Load) pipeline to process raw data:

```bash
python -m etl.run --data-folder data --output processed_data.parquet --fixations-output fixations.parquet
```

### Analysis Pipeline

Run the analysis pipeline to calculate metrics and generate visualizations:

```bash
python -m analysis.run --fixations fixations.parquet --output-dir results
```

## Key Features

- **Multi-subject support**: Analyze data from multiple participants
- **Multi-stimulus support**: Compare gaze patterns across different stimuli
- **Statistical analysis**: Compare groups and run mixed effects models
- **Visualizations**: Generate heatmaps, scanpaths, and metric visualizations
- **GUI application**: User-friendly interface for data exploration

## Command Line Tools

### ETL Pipeline

```
usage: python -m etl.run [-h] [--data-folder DATA_FOLDER] [--pattern PATTERN]
                        [--output OUTPUT] [--fixations-output FIXATIONS_OUTPUT]
                        [--confidence CONFIDENCE]
                        [--max-dispersion MAX_DISPERSION]
                        [--min-duration MIN_DURATION] [--no-blink-detection]
                        [--no-fixation-detection] [--screen-width SCREEN_WIDTH]
                        [--screen-height SCREEN_HEIGHT] [-v]
```

### Analysis Pipeline

```
usage: python -m analysis.run [-h] --fixations FIXATIONS
                             [--output-dir OUTPUT_DIR]
                             [--metrics-output METRICS_OUTPUT]
                             [--trial-durations TRIAL_DURATIONS]
                             [--background-images BACKGROUND_IMAGES]
                             [--group-var GROUP_VAR]
                             [--no-visualizations]
                             [--screen-width SCREEN_WIDTH]
                             [--screen-height SCREEN_HEIGHT] [-v]
```

## Examples

### Calculating Metrics

```python
import pandas as pd
from etl.io import load_all
from etl.preprocess import preprocess_pipeline

# Load and preprocess data
df = load_all("data")
processed_data, fixations = preprocess_pipeline(df)

# Metrics can be calculated or visualized as needed
```

### Creating Visualizations

```python
from analysis.viz import plot_heatmap, plot_scanpath

# Create a heatmap
fig = plot_heatmap(fixations)
fig.savefig("heatmap.png")

# Create a scanpath
fig = plot_scanpath(fixations)
fig.savefig("scanpath.png")
```

### Using the GUI for Analysis

The GUI viewer (`python -m gui.viewer`) provides an interactive way to explore
eye tracking data. Use **Load Data** in the toolbar to select one or more CSV
files. Once loaded, available subjects and stimuli appear in the drop-down
filters. Selecting a subject or stimulus updates the plots for that subset, or
choose **All** to view everything.

In the **Group Analysis** tab pick a group variable (such as `subject` or
`stimulus`) and a metric (`dwell_prop`, `ttf_ms`, or `n_fixations`). Click
**Run Analysis** to compute statistics and draw the group heatmap. After
changing the number of bins or the smoothing sigma you can press **Refresh
Plot** to redraw the heatmap without rerunning the statistics.

The **Metrics** tab visualizes metrics including `dwell_prop`, `ttf_ms`,
`n_fixations`, and `transitions`. Choose the desired metric and optionally group
by subject to update the plot. These metrics are calculated automatically when
data are loaded.

Finally, the **Export Results** button saves the current fixations, computed
metrics, and visualizations to a directory of your choice.

### Recording New Data

1. Start the application with `python -m gui.viewer`.
2. Choose **Run Experiment** from the toolbar.
3. Enter a subject ID, number of images, and timing parameters then click **OK** to begin recording.
4. Place the stimulus images you want to present in the `data/current_images`
   directory and any control images in `data/control_images`. Both folders
   will be created automatically. When no images are found a blank window is
   shown for a few seconds.
5. After the session, CSV files are saved in the `data` folder using the
   `SubjectID_StimulusID.csv` naming pattern.
