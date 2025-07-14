"""
Visualization functions for eye tracking data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from typing import List, Optional, Tuple
from pathlib import Path


def _to_image_coords(
    df: pd.DataFrame,
    screen_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> pd.DataFrame:
    """Convert screen pixel coordinates to original image coordinates."""
    screen_w, screen_h = screen_size
    img_w, img_h = image_size

    scale = min(screen_w / img_w, screen_h / img_h)
    disp_w = img_w * scale
    disp_h = img_h * scale
    margin_x = (screen_w - disp_w) / 2
    margin_y = (screen_h - disp_h) / 2

    out = df.copy()
    out["x_img"] = (out["x_px"] - margin_x) / scale
    out["y_img"] = (out["y_px"] - margin_y) / scale

    mask = (
        (out["x_img"] >= 0)
        & (out["x_img"] <= img_w)
        & (out["y_img"] >= 0)
        & (out["y_img"] <= img_h)
    )
    return out[mask]

def plot_gaze_timeseries(df: pd.DataFrame, fig: Optional[Figure] = None) -> Figure:
    """
    Plot gaze X/Y coordinates over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Eye tracking data with 'time_s', 'x', and 'y' columns
    fig : Optional[Figure], optional
        Matplotlib figure to plot on, by default None
    
    Returns:
    --------
    Figure
        Matplotlib figure with the plot
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    
    ax = fig.add_subplot(111)
    ax.plot(df['time_s'], df['x'], label='Gaze X')
    ax.plot(df['time_s'], df['y'], label='Gaze Y')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Position')
    ax.set_title('Gaze X/Y over Time')
    ax.legend()
    
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    fig: Optional[Figure] = None,
    bins: int = 200,
    sigma: int = 10,
    screen_size: Tuple[int, int] = (1920, 1080),
    image_size: Optional[Tuple[int, int]] = None,
    cmap: str = 'viridis',
    alpha: float = 0.7,
    background_image: Optional[str] = None,
) -> Figure:
    """
    Plot gaze heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Eye tracking data with 'x_px' and 'y_px' columns
    fig : Optional[Figure], optional
        Matplotlib figure to plot on, by default None
    bins : int, optional
        Number of bins for heatmap, by default 200
    sigma : int, optional
        Gaussian blur sigma, by default 10
    screen_size : Tuple[int, int], optional
        Screen dimensions (width, height), by default (1920, 1080)
    image_size : Optional[Tuple[int, int]], optional
        Stimulus image dimensions (width, height). If provided the data is
        centered within the screen dimensions, by default None
    cmap : str, optional
        Colormap name, by default 'viridis'
    alpha : float, optional
        Heatmap transparency, by default 0.7
    background_image : Optional[str], optional
        Path to background image to show under the heatmap, by default None
    
    Returns:
    --------
    Figure
        Matplotlib figure with the heatmap
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    
    screen_w, screen_h = screen_size
    # Determine original image size either from argument or from the file
    if image_size:
        img_w, img_h = image_size
    elif background_image:
        try:
            img_tmp = plt.imread(background_image)
            img_h, img_w = img_tmp.shape[:2]
        except Exception:
            img_w, img_h = screen_size
    else:
        img_w, img_h = screen_size

    # Scale image to fill the screen height while preserving aspect ratio
    scale = screen_h / img_h
    disp_w = img_w * scale
    offset_x = (screen_w - disp_w) / 2
    
    # Filter for valid coordinates
    valid_data = df.dropna(subset=['x_px', 'y_px'])
    if image_size:
        valid_data = valid_data.copy()
        mask = (
            (valid_data['x_px'] >= offset_x) &
            (valid_data['x_px'] <= offset_x + disp_w) &
            (valid_data['y_px'] >= 0) & (valid_data['y_px'] <= screen_h)
        )
        valid_data = valid_data[mask]
    
    # Create heatmap
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')

    # Add background image if provided
    if background_image:
        try:
            img = plt.imread(background_image)
            ax.imshow(img, extent=[offset_x, offset_x + disp_w, screen_h, 0])
        except Exception as e:
            print(f"Error loading background image: {e}")

    x_col, y_col = 'x_px', 'y_px'
    
    if not valid_data.empty:
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            valid_data['x_px'], valid_data['y_px'],
            bins=bins, range=[[0, screen_w], [0, screen_h]]
        )
        
        # Apply Gaussian filter for smoothing
        heat = gaussian_filter(hist, sigma=sigma)
        
        # Normalize
        if heat.max() > 0:
            heat = heat / heat.max()
        
        # Plot heatmap
        im = ax.imshow(
            heat.T, extent=[0, screen_w, screen_h, 0],
            cmap=cmap, alpha=alpha
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Density')

    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_title('Gaze Heatmap')
    
    return fig


def plot_group_heatmap(
    dfs: List[pd.DataFrame],
    labels: List[str],
    fig: Optional[Figure] = None,
    bins: int = 200,
    sigma: int = 10,
    screen_size: Tuple[int, int] = (1920, 1080),
    image_size: Optional[Tuple[int, int]] = None,
    cmap: str = 'viridis'
) -> Figure:
    """
    Plot heatmaps for multiple groups.
    
    Parameters:
    -----------
    dfs : List[pd.DataFrame]
        List of eye tracking DataFrames for different groups
    labels : List[str]
        Labels for each group
    fig : Optional[Figure], optional
        Matplotlib figure to plot on, by default None
    bins : int, optional
        Number of bins for heatmap, by default 200
    sigma : int, optional
        Gaussian blur sigma, by default 10
    screen_size : Tuple[int, int], optional
        Screen dimensions (width, height), by default (1920, 1080)
    image_size : Optional[Tuple[int, int]], optional
        Stimulus image dimensions (width, height). If provided the data is
        centered within the screen dimensions, by default None
    cmap : str, optional
        Colormap name, by default 'viridis'
    
    Returns:
    --------
    Figure
        Matplotlib figure with the heatmaps
    """
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    
    n_groups = len(dfs)
    if n_groups != len(labels):
        raise ValueError("Number of DataFrames must match number of labels")
    
    screen_w, screen_h = screen_size
    img_w, img_h = image_size if image_size else screen_size
    offset_x = (screen_w - img_w) / 2 if image_size else 0
    offset_y = (screen_h - img_h) / 2 if image_size else 0

    # Create heatmaps for each group
    heatmaps = []
    
    for i, df in enumerate(dfs):
        # Filter for valid coordinates
        valid_data = df.dropna(subset=['x_px', 'y_px'])
        if image_size:
            valid_data = valid_data.copy()
            valid_data['x_px'] = valid_data['x_px'] - offset_x
            valid_data['y_px'] = valid_data['y_px'] - offset_y
            valid_data = valid_data[
                (valid_data['x_px'] >= 0) & (valid_data['x_px'] <= img_w) &
                (valid_data['y_px'] >= 0) & (valid_data['y_px'] <= img_h)
            ]
        
        if not valid_data.empty:
            # Create 2D histogram
            hist, x_edges, y_edges = np.histogram2d(
                valid_data['x_px'], valid_data['y_px'],
                bins=bins, range=[[0, img_w], [0, img_h]]
            )
            
            # Apply Gaussian filter for smoothing
            heat = gaussian_filter(hist, sigma=sigma)
            
            # Normalize
            if heat.max() > 0:
                heat = heat / heat.max()
            
            heatmaps.append(heat)
        else:
            heatmaps.append(np.zeros((bins, bins)))
    
    # Determine subplot layout
    n_rows = int(np.ceil(n_groups / 2))
    n_cols = min(2, n_groups)
    
    # Plot each heatmap
    for i, (heat, label) in enumerate(zip(heatmaps, labels)):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        im = ax.imshow(
            heat.T, extent=[0, img_w, img_h, 0],
            cmap=cmap
        )
        plt.colorbar(im, ax=ax, label='Density')
        ax.set_title(label)
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
    
    # Add difference map if there are exactly 2 groups
    if n_groups == 2:
        ax = fig.add_subplot(n_rows, n_cols, 3)
        diff = heatmaps[0] - heatmaps[1]
        im = ax.imshow(
            diff.T, extent=[0, img_w, img_h, 0],
            cmap='coolwarm'
        )
        plt.colorbar(im, ax=ax, label='Difference')
        ax.set_title(f'Difference ({labels[0]} - {labels[1]})')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
    
    fig.tight_layout()
    return fig


def plot_scanpath(
    df: pd.DataFrame,
    fig: Optional[Figure] = None,
    screen_size: Tuple[int, int] = (1920, 1080),
    image_size: Optional[Tuple[int, int]] = None,
    show_fixations: bool = True,
    min_duration_s: float = 0.1,
    background_image: Optional[str] = None,
) -> Figure:
    """
    Plot scanpath trajectory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Eye tracking data with 'x_px' and 'y_px' columns
    fig : Optional[Figure], optional
        Matplotlib figure to plot on, by default None
    screen_size : Tuple[int, int], optional
        Screen dimensions (width, height), by default (1920, 1080)
    image_size : Optional[Tuple[int, int]], optional
        Stimulus image dimensions (width, height). If provided the data is
        centered within the screen dimensions, by default None
    show_fixations : bool, optional
        Whether to show fixations as circles, by default True
    min_duration_s : float, optional
        Minimum fixation duration to show, by default 0.1
    background_image : Optional[str], optional
        Path to background image, by default None
    
    Returns:
    --------
    Figure
        Matplotlib figure with the scanpath
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    
    screen_w, screen_h = screen_size
    
    ax = fig.add_subplot(111)
    
    # Add background image if provided
    if background_image:
        try:
            img = plt.imread(background_image)
            ax.imshow(img, extent=[0, img_w, img_h, 0])
        except Exception as e:
            print(f"Error loading background image: {e}")
    
    # Filter for valid coordinates
    valid_data = df.dropna(subset=['x_px', 'y_px'])
    if image_size:
        valid_data = valid_data.copy()
        valid_data['x_px'] = valid_data['x_px'] - offset_x
        valid_data['y_px'] = valid_data['y_px'] - offset_y
        valid_data = valid_data[
            (valid_data['x_px'] >= 0) & (valid_data['x_px'] <= img_w) &
            (valid_data['y_px'] >= 0) & (valid_data['y_px'] <= img_h)
        ]

    if not valid_data.empty:
        # Plot raw gaze trajectory
        ax.plot(valid_data[x_col], valid_data[y_col], '-', alpha=0.5, linewidth=1, color='blue')
        
        # Plot fixations if requested
        if show_fixations and 'duration_s' in valid_data.columns:
            fixations = valid_data[valid_data['duration_s'] >= min_duration_s]
            
            if not fixations.empty:
                # Scale marker size by duration
                sizes = fixations['duration_s'] * 100
                
                ax.scatter(
                    fixations[x_col], fixations[y_col],
                    s=sizes, alpha=0.6, edgecolors='black', color='red'
                )
    
    # Set plot limits and labels
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)  # Invert Y axis to match screen coordinates
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_title('Scanpath Trajectory')
    
    return fig


def save_all_visualizations(fixations: pd.DataFrame, metrics: pd.DataFrame,
                           output_dir: str, subject: Optional[str] = None,
                           stimulus: Optional[str] = None,
                           screen_size: Tuple[int, int] = (1920, 1080),
                           image_size: Optional[Tuple[int, int]] = None,
                           background_image: Optional[str] = None,
                           transition_matrix: Optional[pd.DataFrame] = None) -> None:
    """
    Generate and save all visualizations for a dataset.
    
    Parameters:
    -----------
    fixations : pd.DataFrame
        DataFrame with fixation data
    metrics : pd.DataFrame
        DataFrame with AOI metrics
    output_dir : str
        Directory to save visualizations
    subject : Optional[str], optional
        Subject ID to filter by, by default None
    stimulus : Optional[str], optional
        Stimulus ID to filter by, by default None
    screen_size : Tuple[int, int], optional
        Screen dimensions (width, height), by default (1920, 1080)
    image_size : Optional[Tuple[int, int]], optional
        Stimulus image dimensions (width, height). If provided the data is
        centered within the screen dimensions, by default None
    background_image : Optional[str], optional
        Path to background image, by default None
    transition_matrix : Optional[pd.DataFrame], optional
        Transition matrix DataFrame, by default None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter data if needed
    filtered_fixations = fixations.copy()
    filtered_metrics = metrics.copy()
    
    if subject is not None:
        filtered_fixations = filtered_fixations[filtered_fixations['subject'] == subject]
        filtered_metrics = filtered_metrics[filtered_metrics['subject'] == subject]
    
    if stimulus is not None:
        filtered_fixations = filtered_fixations[filtered_fixations['stimulus'] == stimulus]
        filtered_metrics = filtered_metrics[filtered_metrics['stimulus'] == stimulus]
    
    # Create file name prefix
    prefix = ""
    if subject:
        prefix += f"{subject}_"
    if stimulus:
        prefix += f"{stimulus}_"
    
    # Generate and save visualizations
    
    # 1. Scanpath
    if not filtered_fixations.empty:
        fig = plot_scanpath(
            filtered_fixations,
            screen_size=screen_size,
            image_size=image_size,
            background_image=background_image,
        )
        fig.savefig(output_path / f"{prefix}scanpath.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 2. Heatmap
    if not filtered_fixations.empty:
        fig = plot_heatmap(
            filtered_fixations,
            screen_size=screen_size,
            image_size=image_size,
            background_image=background_image,
        )
        fig.savefig(output_path / f"{prefix}heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 3. AOI Metrics - Dwell Time
    if not filtered_metrics.empty and 'dwell_prop' in filtered_metrics.columns:
        fig = plot_aoi_metrics(
            filtered_metrics,
            'dwell_prop',
            by_subject=subject is None  # Only by subject if not already filtered
        )
        fig.savefig(output_path / f"{prefix}dwell_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 4. AOI Metrics - Time to First Fixation
    if not filtered_metrics.empty and 'ttf_ms' in filtered_metrics.columns:
        fig = plot_aoi_metrics(
            filtered_metrics,
            'ttf_ms',
            by_subject=subject is None  # Only by subject if not already filtered
        )
        fig.savefig(output_path / f"{prefix}ttf.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 5. Transition Matrix
    if transition_matrix is not None:
        fig = plot_transition_matrix(transition_matrix)
        fig.savefig(output_path / f"{prefix}transitions.png", dpi=300, bbox_inches='tight')
        plt.close(fig) 