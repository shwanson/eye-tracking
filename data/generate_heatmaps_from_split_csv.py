import os, re
from pathlib import Path
import pandas as pd, numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ---------- Parameter ----------
PROBAND   = "P009"              # Ordnerpräfix / Kennung
CSV_DIR   = Path("data/split_csv")
IMG_DIR   = Path("data/current_images")
OUT_DIR   = Path("data/heatmap_overlays") / PROBAND
IMG_SIZE  = 800                 # 800×800 px Target­auflösung
SIGMA     = 25                  # Gauß-Blur in Pixel
VEL_THR   = 1     # px/ms – Mikrosakkaden-Filter
AMP_THR   = 25    # px
DUR_THR   = 50    # ms
# --------------------------------

def filter_noise(df,
                 vel_thresh=VEL_THR,
                 amp_thresh=AMP_THR,
                 dur_thresh=DUR_THR):
    """Identisch zu deiner alten Funktion (leicht gekürzt)."""
    df = df.sort_values('system_time_stamp').reset_index(drop=True)
    dt  = df['system_time_stamp'].diff().bfill()
    dx  = df['x'].diff().fillna(0)
    dy  = df['y'].diff().fillna(0)
    vel = np.hypot(dx, dy) / dt.clip(lower=1)          # px/ms
    is_sacc  = vel > vel_thresh                        # pot. Sakkade
    is_blink = (df['x'] == 0) | (df['y'] == 0) | df[['x','y']].isna().any(axis=1)

    noise = is_blink.copy()
    sacc_idx = np.where(is_sacc)[0]
    if len(sacc_idx):
        gaps       = np.where(np.diff(sacc_idx) > 1)[0]
        seg_start  = np.insert(sacc_idx[gaps+1], 0, sacc_idx[0])
        seg_end    = np.append(sacc_idx[gaps],    sacc_idx[-1])
        for s,e in zip(seg_start, seg_end):
            dur = df.loc[e,'system_time_stamp'] - df.loc[s,'system_time_stamp']
            amp = np.hypot(df.loc[e,'x']-df.loc[s,'x'], df.loc[e,'y']-df.loc[s,'y'])
            if dur < dur_thresh and amp < amp_thresh:
                noise[s:e+1] = True

    clean = df.copy()
    clean.loc[noise, ['x','y']] = np.nan
    clean[['x','y']] = clean[['x','y']].interpolate().ffill().bfill()
    return clean

def rel2px(df_rel, img_w=IMG_SIZE, img_h=IMG_SIZE):
    """Konvertiert relative Koordinaten (0–1) in Pixel im 800×800-Bildfenster."""
    # Screen-Ränder (Bild zentriert auf 16:9-Screen)
    screen_h = img_h
    screen_w = 16/9 * screen_h
    margin_x = (screen_w - img_w) / 2
    min_rx   = margin_x / screen_w
    max_rx   = (margin_x + img_w) / screen_w

    df = df_rel[(df_rel['x'] >= min_rx) & (df_rel['x'] <= max_rx)].copy()
    df['x_px'] = ((df['x'] - min_rx) / (max_rx - min_rx)) * img_w
    df['y_px'] = df['y'] * img_h
    return df

def csv_to_heatmap(csv_path: Path):
    """Liest eine Split-CSV, erzeugt zwei PNGs (Overlay + reine Heatmap)."""
    # -------  Bildname ableiten  -------
    # Beispiel:  P005_id039_person_.csv  ->  id039_person_.jpg
    stem      = csv_path.stem          # ohne .csv
    img_core  = re.sub(r"^P\d+[_ -]*", "", stem)   # Proband-Prefix abwerfen
    img_file  = IMG_DIR / f"{img_core}.jpg"

    if not img_file.exists():
        print(f"[WARN] Bild {img_file.name} nicht gefunden – übersprungen.")
        return

    # -------  Daten vorbereiten  -------
    df_raw   = pd.read_csv(csv_path)
    coords   = df_raw['left_gaze_point_on_display_area'].str.strip("()").str.split(",", expand=True).astype(float)
    df_coords          = pd.DataFrame({'x': coords[0], 'y': coords[1]})
    df_coords['system_time_stamp'] = df_raw['system_time_stamp']

    df_clean = filter_noise(df_coords)
    df_px    = rel2px(df_clean)

    # -------  Heatmap erstellen  -------
    img      = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)
    hist, _, _ = np.histogram2d(df_px['x_px'], df_px['y_px'],
                                bins=[IMG_SIZE, IMG_SIZE],
                                range=[[0, IMG_SIZE], [0, IMG_SIZE]])
    heat     = gaussian_filter(hist.T, sigma=SIGMA)
    heat     = heat / heat.max()                # 0–1 normalisiert
    heatmask = np.ma.masked_where(heat < 0.01, heat)

    # -------  Plot & Save  -------
    out_dir = OUT_DIR / img_core
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = 100
    figsize = (IMG_SIZE/dpi, IMG_SIZE/dpi)

    # Overlay
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img, zorder=1)
    plt.imshow(heatmask, cmap='jet', alpha=0.6, zorder=2, origin='upper', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(out_dir / f"heatmap_overlay_{img_core}.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Nur Heatmap, transparent
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(heatmask, cmap='jet', alpha=1.0, origin='upper', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(out_dir / f"heatmap_only_{img_core}.png",
                dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    print(f"[OK ] {img_core}: Heatmaps gespeichert → {out_dir}")

# ---------- Hauptschleife ----------
for csv_file in CSV_DIR.glob(f"{PROBAND}_*.csv"):
    csv_to_heatmap(csv_file)
