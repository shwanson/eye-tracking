"""
ETL Pipeline runner for eye tracking data.

This module provides a command-line interface to run the ETL pipeline.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from etl.io import load_all, save_processed
from etl.preprocess import preprocess_pipeline


def setup_logging(verbosity: int = 0) -> None:
    """
    Set up logging with appropriate verbosity.
    
    Parameters:
    -----------
    verbosity : int, optional
        0 = WARNING, 1 = INFO, 2 = DEBUG, by default 0
    """
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    level = log_levels.get(verbosity, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_pipeline(
    data_folder: str = "data",
    pattern: str = "*.csv",
    screen_size: tuple = (1920, 1080),
    output_path: Optional[str] = None,
    **kwargs: Dict[str, Any]
) -> tuple:
    """
    Run the ETL pipeline.
    
    Parameters:
    -----------
    data_folder : str, optional
        Path to the folder containing raw data, by default "data"
    pattern : str, optional
        File pattern to match, by default "*.csv"
    screen_size : tuple, optional
        Screen dimensions (width, height), by default (1920, 1080)
    output_path : Optional[str], optional
        Path to save processed data, by default None
    **kwargs : Dict[str, Any]
        Additional parameters for preprocessing
    
    Returns:
    --------
    tuple
        (processed_data, fixations_data)
    """
    logging.info(f"Loading data from {data_folder}")
    raw_data = load_all(data_folder, screen_size, pattern)
    
    logging.info(f"Preprocessing data")
    processed_data, fixations = preprocess_pipeline(raw_data, **kwargs)
    
    # Save processed data if output path provided
    if output_path:
        logging.info(f"Saving processed data to {output_path}")
        save_processed(processed_data, output_path)
    
    return processed_data, fixations


def main():
    """
    Main entry point for the ETL pipeline.
    """
    parser = argparse.ArgumentParser(description="Eye-tracking ETL Pipeline")
    parser.add_argument("--data-folder", type=str, default="data",
                        help="Folder containing raw data files")
    parser.add_argument("--pattern", type=str, default="*.csv",
                        help="File pattern to match")
    parser.add_argument("--output", type=str, default="processed_data.parquet",
                        help="Path to save processed data")
    parser.add_argument("--fixations-output", type=str, default="fixations.parquet",
                        help="Path to save fixations data")
    parser.add_argument("--confidence", type=float, default=0.8,
                        help="Confidence threshold for quality filtering")
    parser.add_argument("--max-dispersion", type=float, default=0.04,
                        help="Maximum dispersion for fixation detection")
    parser.add_argument("--min-duration", type=float, default=0.1,
                        help="Minimum fixation duration (seconds)")
    parser.add_argument("--no-blink-detection", action="store_true",
                        help="Skip blink detection")
    parser.add_argument("--no-fixation-detection", action="store_true",
                        help="Skip fixation detection")
    parser.add_argument("--screen-width", type=int, default=1920,
                        help="Screen width in pixels")
    parser.add_argument("--screen-height", type=int, default=1080,
                        help="Screen height in pixels")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (can be used multiple times)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Run the pipeline
    try:
        run_pipeline(
            data_folder=args.data_folder,
            pattern=args.pattern,
            screen_size=(args.screen_width, args.screen_height),
            output_path=args.output,
            confidence_threshold=args.confidence,
            detect_blinks_flag=not args.no_blink_detection,
            detect_fixations_flag=not args.no_fixation_detection,
            max_dispersion=args.max_dispersion,
            min_duration_s=args.min_duration
        )
        logging.info("ETL pipeline completed successfully")
    except Exception as e:
        logging.error(f"Error in ETL pipeline: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 