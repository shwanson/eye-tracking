"""
Functions for loading and saving eye tracking data
"""
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Optional
from multiprocessing import Pool
import numpy as np


def _parse_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Tobii coordinate data from strings to numeric columns
    """
    try:
        # Parse left eye coordinates
        # First ensure the column exists
        if 'left_gaze_point_on_display_area' not in df.columns:
            raise ValueError("Required column 'left_gaze_point_on_display_area' not found in DataFrame")
            
        # Convert to string to ensure consistent handling
        df['left_gaze_point_on_display_area'] = df['left_gaze_point_on_display_area'].astype(str)
        
        # Extract x and y values from the string tuples
        # Handle format like "(0.491, 0.638)" by removing parentheses and splitting by comma
        # Pandas str.replace treats patterns as regular expressions by default,
        # which causes an error when using unescaped parentheses. Explicitly
        # disable regex mode so that literal characters are replaced.
        left_coords = (
            df['left_gaze_point_on_display_area']
            .str.replace('(', '', regex=False)
            .str.replace(')', '', regex=False)
        )
        left_x = left_coords.str.split(',', expand=True)[0].str.strip().replace('nan', np.nan).astype(float)
        left_y = left_coords.str.split(',', expand=True)[1].str.strip().replace('nan', np.nan).astype(float)
        
        # Create new columns
        df['lx'] = left_x
        df['ly'] = left_y

        # Parse right eye coordinates with the same approach
        if 'right_gaze_point_on_display_area' not in df.columns:
            raise ValueError("Required column 'right_gaze_point_on_display_area' not found in DataFrame")
            
        df['right_gaze_point_on_display_area'] = df['right_gaze_point_on_display_area'].astype(str)
        
        right_coords = (
            df['right_gaze_point_on_display_area']
            .str.replace('(', '', regex=False)
            .str.replace(')', '', regex=False)
        )
        right_x = right_coords.str.split(',', expand=True)[0].str.strip().replace('nan', np.nan).astype(float)
        right_y = right_coords.str.split(',', expand=True)[1].str.strip().replace('nan', np.nan).astype(float)
        
        df['rx'] = right_x
        df['ry'] = right_y

        # Calculate binocular mean
        df['x'] = df[['lx', 'rx']].mean(axis=1)
        df['y'] = df[['ly', 'ry']].mean(axis=1)

        # Calculate time relative to start (s)
        if 'device_time_stamp' in df.columns:
            t0 = df['device_time_stamp'].iloc[0]
            df['time_s'] = (df['device_time_stamp'] - t0) / 1e6
        else:
            print("Warning: 'device_time_stamp' column not found, skipping time calculation")

        return df
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        # Provide helpful debugging information
        print("DataFrame columns:", df.columns.tolist())
        try:
            sample_left = df['left_gaze_point_on_display_area'].head(3).tolist() if 'left_gaze_point_on_display_area' in df.columns else "Column not found"
            sample_right = df['right_gaze_point_on_display_area'].head(3).tolist() if 'right_gaze_point_on_display_area' in df.columns else "Column not found"
            print(f"Sample left coordinates: {sample_left}")
            print(f"Sample right coordinates: {sample_right}")
        except:
            pass
        raise


def load_subject(path: Path, screen_size: Tuple[int, int] = (1920, 1080)) -> pd.DataFrame:
    """
    Load eye tracking data for a single subject and stimulus.
    
    Parameters:
    -----------
    path : Path
        Path to the CSV file
    screen_size : Tuple[int, int], optional
        Screen dimensions in pixels (width, height), by default (1920, 1080)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed eye tracking data
    
    Notes:
    ------
    Expects filename format like "P01_face01.csv" where:
    - P01 is the subject ID
    - face01 is the stimulus ID
    """
    path = Path(path)  # Ensure path is a Path object
    
    try:
        # Read the CSV file
        df = pd.read_csv(path)
        
        # Parse coordinates
        df = _parse_coordinates(df)
        
        # Extract metadata from filename
        parts = path.stem.split('_')
        if len(parts) >= 2:
            df['subject'] = parts[0]    # P01, P02, etc.
            df['stimulus'] = parts[1]   # face01, face02, etc.
        else:
            # Fallback if filename doesn't match expected pattern
            df['subject'] = path.stem
            df['stimulus'] = 'unknown'
        
        # Add screen dimensions
        screen_w, screen_h = screen_size
        df['screen_w'] = screen_w
        df['screen_h'] = screen_h
        
        # Calculate pixel coordinates
        df['x_px'] = df['x'] * screen_w
        df['y_px'] = df['y'] * screen_h
        
        return df
    except Exception as e:
        print(f"Error loading file {path}: {e}")
        raise


def load_all(folder: str = "data", screen_size: Tuple[int, int] = (1920, 1080), 
             pattern: str = "*.csv", parallel: bool = True) -> pd.DataFrame:
    """
    Load all eye tracking data files from a folder.
    
    Parameters:
    -----------
    folder : str, optional
        Path to the folder containing CSV files, by default "data"
    screen_size : Tuple[int, int], optional
        Screen dimensions in pixels (width, height), by default (1920, 1080)
    pattern : str, optional
        File pattern to match, by default "*.csv"
    parallel : bool, optional
        Whether to use parallel processing, by default True
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all eye tracking data
    """
    folder_path = Path(folder)
    files = list(folder_path.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {folder}")
    
    if parallel and len(files) > 1:
        # Use parallel processing for multiple files
        with Pool() as p:
            dfs = p.starmap(load_subject, [(f, screen_size) for f in files])
    else:
        # Sequential processing
        dfs = [load_subject(f, screen_size) for f in files]
    
    # Combine all DataFrames
    return pd.concat(dfs, ignore_index=True)


def save_processed(df: pd.DataFrame, output_path: str, format: str = "parquet") -> None:
    """
    Save processed eye tracking data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str
        Path to save the DataFrame to
    format : str, optional
        File format ("csv", "parquet"), by default "parquet"
    """
    path = Path(output_path)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in the specified format
    if format.lower() == "csv":
        df.to_csv(path, index=False)
    elif format.lower() == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.") 