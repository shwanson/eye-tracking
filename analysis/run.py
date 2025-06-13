"""
Analysis Pipeline runner for eye tracking data.

This module provides a command-line interface to run the analysis pipeline.
"""
import argparse
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from analysis.metrics import all_metrics, transition_matrix
from analysis.group import aggregate_by_group, compare_groups, mixed_effects_model
from analysis.viz import save_all_visualizations


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


def load_data(fixations_path: str, trial_durations_path: Optional[str] = None) -> Tuple:
    """
    Load data for analysis.
    
    Parameters:
    -----------
    fixations_path : str
        Path to the fixations data file
    trial_durations_path : Optional[str], optional
        Path to trial durations JSON file, by default None
    
    Returns:
    --------
    Tuple
        (fixations_df, trial_durations_dict)
    """
    # Load fixations
    fixations_file = Path(fixations_path)
    if fixations_file.suffix.lower() == '.parquet':
        fixations_df = pd.read_parquet(fixations_file)
    else:
        fixations_df = pd.read_csv(fixations_file)
    
    # Load trial durations if provided
    trial_durations_dict = None
    if trial_durations_path:
        with open(trial_durations_path, 'r') as f:
            trial_durations_dict = json.load(f)
    
    return fixations_df, trial_durations_dict


def run_analysis(
    fixations_df: pd.DataFrame, 
    output_dir: str, 
    output_metrics_path: Optional[str] = None,
    group_var: Optional[str] = None,
    trial_durations_dict: Optional[Dict] = None,
    background_images: Optional[Dict] = None,
    generate_visualizations: bool = True,
    screen_size: Tuple[int, int] = (1920, 1080)
) -> pd.DataFrame:
    """
    Run the analysis pipeline.
    
    Parameters:
    -----------
    fixations_df : pd.DataFrame
        DataFrame with fixation data
    output_dir : str
        Directory to save analysis results
    output_metrics_path : Optional[str], optional
        Path to save metrics data, by default None
    group_var : Optional[str], optional
        Column name for grouping subjects, by default None
    trial_durations_dict : Optional[Dict], optional
        Dictionary mapping stimulus IDs to trial durations, by default None
    background_images : Optional[Dict], optional
        Dictionary mapping stimulus IDs to background images, by default None
    generate_visualizations : bool, optional
        Whether to generate visualizations, by default True
    screen_size : Tuple[int, int], optional
        Screen dimensions (width, height), by default (1920, 1080)
    
    Returns:
    --------
    pd.DataFrame
        Metrics DataFrame
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Calculating metrics")
    metrics_df = all_metrics(fixations_df, trial_durations_dict)
    
    # Calculate transition matrix
    logging.info("Calculating transition matrices")
    transitions, _ = transition_matrix(fixations_df)
    
    # Save metrics if output path provided
    if output_metrics_path:
        logging.info(f"Saving metrics to {output_metrics_path}")
        metrics_df.to_csv(output_metrics_path, index=False)
    
    # Group analysis if group variable provided
    if group_var and group_var in fixations_df.columns:
        logging.info(f"Performing group analysis by {group_var}")
        
        # Create group output directory
        group_dir = output_path / "group_analysis"
        group_dir.mkdir(exist_ok=True)
        
        # Aggregate metrics by group
        agg_metrics = aggregate_by_group(metrics_df, group_var)
        agg_metrics.to_csv(group_dir / "aggregated_metrics.csv", index=False)
        
        # Compare groups
        for metric in ['dwell_prop', 'ttf_ms', 'n_fixations']:
            if metric in metrics_df.columns:
                comparison = compare_groups(metrics_df, group_var, metric)
                comparison.to_csv(group_dir / f"{metric}_group_comparison.csv", index=False)
        
        # Run mixed effects model
        try:
            if 'dwell_prop' in metrics_df.columns:
                model_results = mixed_effects_model(
                    metrics_df, formula="dwell_prop ~ subject", group_var="subject"
                )
                model_results.to_csv(
                    group_dir / "mixed_effects_model.csv", index=False
                )
        except Exception as e:
            logging.error(f"Error running mixed effects model: {e}")
    
    # Generate visualizations if requested
    if generate_visualizations:
        logging.info("Generating visualizations")
        
        # Create visualization directory
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Get all subjects and stimuli
        subjects = sorted(fixations_df['subject'].unique())
        stimuli = sorted(fixations_df['stimulus'].unique())
        
        # Group visualizations - all subjects, all stimuli
        save_all_visualizations(
            fixations_df, metrics_df, 
            viz_dir, 
            transition_matrix=transitions,
            screen_size=screen_size
        )
        
        # Individual subject visualizations
        for subject in subjects:
            subject_dir = viz_dir / subject
            subject_dir.mkdir(exist_ok=True)
            
            # All stimuli for this subject
            subj_fixations = fixations_df[fixations_df['subject'] == subject]
            subj_metrics = metrics_df[metrics_df['subject'] == subject]
            
            save_all_visualizations(
                subj_fixations, subj_metrics,
                subject_dir,
                subject=subject,
                transition_matrix=transition_matrix(subj_fixations)[0],
                screen_size=screen_size
            )
            
            # Each stimulus for this subject
            for stimulus in stimuli:
                if stimulus in subj_fixations['stimulus'].unique():
                    stim_fixations = subj_fixations[subj_fixations['stimulus'] == stimulus]
                    stim_metrics = subj_metrics[subj_metrics['stimulus'] == stimulus]
                    
                    # Get background image if available
                    bg_img = None
                    if background_images and stimulus in background_images:
                        bg_img = background_images[stimulus]
                    
                    save_all_visualizations(
                        stim_fixations, stim_metrics,
                        subject_dir,
                        subject=subject,
                        stimulus=stimulus,
                        background_image=bg_img,
                        screen_size=screen_size
                    )
        
        # Stimulus visualizations (across subjects)
        for stimulus in stimuli:
            stimulus_dir = viz_dir / "stimuli" / stimulus
            stimulus_dir.mkdir(parents=True, exist_ok=True)
            
            stim_fixations = fixations_df[fixations_df['stimulus'] == stimulus]
            stim_metrics = metrics_df[metrics_df['stimulus'] == stimulus]
            
            # Get background image if available
            bg_img = None
            if background_images and stimulus in background_images:
                bg_img = background_images[stimulus]
            
            save_all_visualizations(
                stim_fixations, stim_metrics,
                stimulus_dir,
                stimulus=stimulus,
                background_image=bg_img,
                screen_size=screen_size
            )
    
    return metrics_df


def main():
    """
    Main entry point for the analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="Eye-tracking Analysis Pipeline")
    parser.add_argument("--fixations", type=str, required=True,
                        help="Path to the fixations data file")
    parser.add_argument("--output-dir", type=str, default="analysis_results",
                        help="Directory to save analysis results")
    parser.add_argument("--metrics-output", type=str, default="metrics.csv",
                        help="Path to save metrics data")
    parser.add_argument("--trial-durations", type=str, 
                       help="Path to trial durations JSON file")
    parser.add_argument("--background-images", type=str,
                       help="Path to background images mapping JSON file")
    parser.add_argument("--group-var", type=str,
                       help="Column name for grouping subjects")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip generating visualizations")
    parser.add_argument("--screen-width", type=int, default=1920,
                       help="Screen width in pixels")
    parser.add_argument("--screen-height", type=int, default=1080,
                       help="Screen height in pixels")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                       help="Increase verbosity (can be used multiple times)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Load data
    try:
        logging.info(f"Loading data from {args.fixations}")
        fixations_df, trial_durations_dict = load_data(
            args.fixations, args.trial_durations
        )
        
        # Load background images if provided
        background_images = None
        if args.background_images:
            with open(args.background_images, 'r') as f:
                background_images = json.load(f)
        
        # Run analysis
        run_analysis(
            fixations_df=fixations_df,
            output_dir=args.output_dir,
            output_metrics_path=args.metrics_output,
            group_var=args.group_var,
            trial_durations_dict=trial_durations_dict,
            background_images=background_images,
            generate_visualizations=not args.no_visualizations,
            screen_size=(args.screen_width, args.screen_height)
        )
        
        logging.info("Analysis pipeline completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis pipeline: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 