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