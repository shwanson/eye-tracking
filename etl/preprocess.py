"""
Eye tracking data preprocessing, filtering, and fixation detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def filter_quality(df: pd.DataFrame, confidence_threshold: float = 0.8) -> pd.DataFrame:
    """
    Filter out low-quality eye tracking data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Eye tracking data
    confidence_threshold : float, optional
        Minimum acceptable confidence value, by default 0.8
    
    Returns:
    --------
    pd.DataFrame
        Filtered eye tracking data
    """
    # Calculate mean confidence for both eyes
    if 'left_gaze_point_validity' in df.columns and 'right_gaze_point_validity' in df.columns:
        df['confidence'] = df[['left_gaze_point_validity', 'right_gaze_point_validity']].mean(axis=1)
        return df[df['confidence'] >= confidence_threshold].copy()
    else:
        # If validity columns not available, return original data
        logger.warning("Validity columns not found. Data not filtered.")
        return df


def detect_blinks(df: pd.DataFrame, max_gap_ms: float = 100.0) -> pd.DataFrame:
    """
    Detect blinks in eye tracking data and mark them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Eye tracking data
    max_gap_ms : float, optional
        Maximum time gap (in ms) to be considered as a blink, by default 100.0
    
    Returns:
    --------
    pd.DataFrame
        Eye tracking data with blinks marked
    """
    # Calculate gaps in data
    df = df.sort_values('time_s').reset_index(drop=True)
    df['time_gap_ms'] = (df['time_s'].diff() * 1000)
    
    # Mark blinks where either validity is 0 or there's a time gap
    if 'left_gaze_point_validity' in df.columns and 'right_gaze_point_validity' in df.columns:
        df['is_blink'] = ((df['left_gaze_point_validity'] == 0) | 
                          (df['right_gaze_point_validity'] == 0) | 
                          (df['time_gap_ms'] > max_gap_ms))
    else:
        # If validity columns not available, use only time gaps
        df['is_blink'] = df['time_gap_ms'] > max_gap_ms
    
    return df


def detect_fixations(df: pd.DataFrame, max_dispersion: float = 0.04, 
                    min_duration_s: float = 0.1) -> pd.DataFrame:
    """
    Detect fixations using the I-DT (Dispersion-Threshold) algorithm.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Eye tracking data
    max_dispersion : float, optional
        Maximum dispersion threshold, by default 0.04
    min_duration_s : float, optional
        Minimum fixation duration in seconds, by default 0.1
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with detected fixations
    """
    logger.debug('detect_fixations input shape: %s', df.shape)
    logger.debug('detect_fixations input columns: %s', df.columns.tolist())
    fixations = []
    window = []
    start_t = None
    
    # Sort data by time
    df = df.sort_values('time_s')
    
    # Skip rows marked as blinks if the column exists
    if 'is_blink' in df.columns:
        df = df[~df['is_blink']]
    logger.debug('detect_fixations after blink filter shape: %s', df.shape)
    logger.debug('detect_fixations after blink filter columns: %s', df.columns.tolist())
    
    for _, row in df.iterrows():
        # Skip NaN coordinates
        if pd.isna(row['x']) or pd.isna(row['y']):
            continue
            
        pt = (row['x'], row['y'], row['time_s'])
        
        if not window:
            window, start_t = [pt], pt[2]
            continue
        
        window.append(pt)
        xs, ys = [p[0] for p in window], [p[1] for p in window]
        disp = (max(xs) - min(xs)) + (max(ys) - min(ys))
        dur = window[-1][2] - start_t
        
        if disp > max_dispersion:
            if dur >= min_duration_s:
                # Create fixation record
                # Calculate pixel coordinates directly from the dataframe
                window_data = df[(df['time_s'] >= start_t) & (df['time_s'] <= window[-2][2])]
                
                fix = {
                    'start_s': start_t,
                    'end_s': window[-2][2],
                    'duration_s': window[-2][2] - start_t,
                    'x': np.mean(xs[:-1]),
                    'y': np.mean(ys[:-1]),
                    'x_px': np.mean(window_data['x_px']),
                    'y_px': np.mean(window_data['y_px']),
                }
                
                # Add subject and stimulus if available
                if 'subject' in df.columns:
                    fix['subject'] = df['subject'].iloc[0]
                if 'stimulus' in df.columns:
                    fix['stimulus'] = df['stimulus'].iloc[0]
                
                fixations.append(fix)
            
            window, start_t = [pt], pt[2]
    
    # Check if last window was a fixation
    if window and (window[-1][2] - start_t) >= min_duration_s:
        xs, ys = [p[0] for p in window], [p[1] for p in window]
        disp = (max(xs) - min(xs)) + (max(ys) - min(ys))
        
        if disp <= max_dispersion:
            # Calculate pixel coordinates directly from the dataframe
            window_data = df[(df['time_s'] >= start_t) & (df['time_s'] <= window[-1][2])]
            
            # Create fixation record for last window
            fix = {
                'start_s': start_t,
                'end_s': window[-1][2],
                'duration_s': window[-1][2] - start_t,
                'x': np.mean(xs),
                'y': np.mean(ys),
                'x_px': np.mean(window_data['x_px']),
                'y_px': np.mean(window_data['y_px']),
            }
            
            # Add subject and stimulus if available
            if 'subject' in df.columns:
                fix['subject'] = df['subject'].iloc[0]
            if 'stimulus' in df.columns:
                fix['stimulus'] = df['stimulus'].iloc[0]
            
            fixations.append(fix)
    
    result = pd.DataFrame(fixations)
    logger.debug('detect_fixations output shape: %s', result.shape)
    logger.debug('detect_fixations output columns: %s', result.columns.tolist())
    return result


def preprocess_pipeline(df: pd.DataFrame, 
                       confidence_threshold: float = 0.8,
                       detect_blinks_flag: bool = True,
                       max_gap_ms: float = 100.0,
                       detect_fixations_flag: bool = True,
                       max_dispersion: float = 0.04,
                       min_duration_s: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline for eye tracking data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw eye tracking data
    confidence_threshold : float, optional
        Minimum confidence value, by default 0.8
    detect_blinks_flag : bool, optional
        Whether to detect blinks, by default True
    max_gap_ms : float, optional
        Maximum time gap for blink detection, by default 100.0
    detect_fixations_flag : bool, optional
        Whether to detect fixations, by default True
    max_dispersion : float, optional
        Maximum dispersion for fixation detection, by default 0.04
    min_duration_s : float, optional
        Minimum fixation duration, by default 0.1
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (processed_data, fixations)
    """
    logger.debug('preprocess_pipeline input shape: %s', df.shape)
    logger.debug('preprocess_pipeline input columns: %s', df.columns.tolist())
    # Filter by quality
    filtered_df = filter_quality(df, confidence_threshold)
    logger.debug('After filter_quality shape: %s', filtered_df.shape)
    logger.debug('After filter_quality columns: %s', filtered_df.columns.tolist())
    # Detect blinks
    if detect_blinks_flag:
        filtered_df = detect_blinks(filtered_df, max_gap_ms)
        logger.debug('After detect_blinks shape: %s', filtered_df.shape)
        logger.debug('After detect_blinks columns: %s', filtered_df.columns.tolist())
    # Detect fixations PER SUBJECT/STIMULUS
    fixations_df = None
    if detect_fixations_flag:
        fixations_list = []
        for (subject, stimulus), group in filtered_df.groupby(['subject', 'stimulus']):
            group_fix = detect_fixations(group, max_dispersion, min_duration_s)
            if not group_fix.empty:
                group_fix['subject'] = subject
                group_fix['stimulus'] = stimulus
                fixations_list.append(group_fix)
        if fixations_list:
            fixations_df = pd.concat(fixations_list, ignore_index=True)
        else:
            fixations_df = pd.DataFrame(columns=['start_s', 'end_s', 'duration_s', 'x', 'y', 'x_px', 'y_px', 'subject', 'stimulus'])
        logger.debug('After detect_fixations shape: %s', fixations_df.shape if fixations_df is not None else None)
        logger.debug('After detect_fixations columns: %s', fixations_df.columns.tolist() if fixations_df is not None else None)
    return filtered_df, fixations_df 