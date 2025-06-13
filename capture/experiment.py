"""Run a Tobii based gaze experiment."""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pygame

try:
    import tobii_research as tr  # type: ignore
except Exception:  # pragma: no cover - library not available
    tr = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parents[1]
CURRENT_DIR = BASE_DIR / "data" / "current_images"
CONTROL_DIR = BASE_DIR / "data" / "control_images"
CSV_DIR = BASE_DIR / "data" / "csv"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def prompt_experiment_params() -> Tuple[str, int, float, float]:
    """Interactively ask for experiment parameters."""
    subject = input("Subject ID: ").strip()
    while not subject:
        subject = input("Subject ID: ").strip()

    num = input("Number of images to show: ").strip()
    while not num.isdigit() or int(num) <= 0:
        num = input("Number of images to show: ").strip()
    num_images = int(num)

    def _float(prompt: str) -> float:
        while True:
            val = input(prompt).strip()
            try:
                return float(val)
            except ValueError:
                continue

    min_dur = _float("Minimum display time (s): ")
    max_dur = _float("Maximum display time (s): ")
    if max_dur < min_dur:
        max_dur = min_dur
    return subject, num_images, min_dur, max_dur


def _load_images(folder: Path) -> List[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    paths.extend(sorted(folder.glob("*.jpg")))
    paths.extend(sorted(folder.glob("*.png")))
    random.shuffle(paths)
    return paths


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_tracker(tracker: "tr.EyeTracker", screen: pygame.Surface) -> None:
    if tr is None:
        raise RuntimeError("tobii_research library is not available")

    calib = tr.ScreenBasedCalibration(tracker)
    calib.enter_calibration_mode()

    rect = screen.get_rect()
    points = [
        (0.5, 0.5),
        (0.1, 0.1),
        (0.9, 0.1),
        (0.1, 0.9),
        (0.9, 0.9),
    ]

    for x_rel, y_rel in points:
        x = int(rect.width * x_rel)
        y = int(rect.height * y_rel)
        start = time.time()
        status = tr.CALIBRATION_STATUS_NEW_DATA
        while status != tr.CALIBRATION_STATUS_SUCCESS:
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 0, 0), (x, y), 20)
            pygame.display.flip()
            status = calib.collect_data(x_rel, y_rel)
            if status != tr.CALIBRATION_STATUS_SUCCESS:
                time.sleep(0.1)
        elapsed = time.time() - start
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)

    calib.compute_and_apply()
    calib.leave_calibration_mode()


# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

def _run_validity_check(
    screen: pygame.Surface, subject: str, shown: List[Path]
) -> None:
    control = _load_images(CONTROL_DIR)
    trials = [(p, "Y") for p in shown] + [(p, "N") for p in control]
    random.shuffle(trials)

    font = pygame.font.SysFont(None, 36)
    results: List[Dict[str, object]] = []

    rect = screen.get_rect()
    for path, expected in trials:
        img = pygame.image.load(str(path))
        iw, ih = img.get_size()
        scale = min(rect.width / iw, rect.height / ih)
        img = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        img_rect = img.get_rect(center=rect.center)

        answered = False
        resp = ""
        while not answered:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if event.key == pygame.K_y:
                        resp = "Y"
                        answered = True
                    if event.key == pygame.K_n:
                        resp = "N"
                        answered = True
            screen.fill((0, 0, 0))
            screen.blit(img, img_rect)
            prompt = font.render(
                "Was this shown before? (Y=Yes, N=No)", True, (255, 255, 255)
            )
            prompt_rect = prompt.get_rect(
                center=(rect.centerx, rect.height - 30)
            )
            screen.blit(prompt, prompt_rect)
            pygame.display.flip()
        results.append({"image": path.name, "expected": expected, "response": resp})

    _write_csv(CSV_DIR / f"P{subject}_Valididitätskontrolle.csv", results)


# ---------------------------------------------------------------------------
# Experiment logic
# ---------------------------------------------------------------------------

def run_experiment(subject: str, num_images: int, min_dur: float, max_dur: float) -> None:
    if tr is None:
        raise RuntimeError("tobii_research library is not available")

    trackers = tr.find_all_eyetrackers()
    if not trackers:
        raise RuntimeError("No Tobii eye tracker found")
    tracker = trackers[0]
    print(f"Using eye tracker: {tracker.model}\n")

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    calibrate_tracker(tracker, screen)

    stimuli = _load_images(CURRENT_DIR)
    if not stimuli:
        print("No images found in", CURRENT_DIR)
        pygame.quit()
        return
    if num_images > len(stimuli):
        num_images = len(stimuli)
    stimuli = stimuli[:num_images]

    shown: List[Path] = []
    rect = screen.get_rect()

    half = len(stimuli) // 2 if stimuli else 0

    for idx, path in enumerate(stimuli):
        img = pygame.image.load(str(path))
        iw, ih = img.get_size()
        scale = min(rect.width / iw, rect.height / ih)
        img = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        img_rect = img.get_rect(center=rect.center)

        samples: List[Dict[str, object]] = []

        def gaze_callback(data: Dict) -> None:
            left = data.get("left_gaze_point_on_display_area") or (None, None)
            right = data.get("right_gaze_point_on_display_area") or (None, None)
            samples.append(
                {
                    "system_time_stamp": data.get("system_time_stamp"),
                    "device_time_stamp": data.get("device_time_stamp"),
                    "left_x": left[0],
                    "left_y": left[1],
                    "right_x": right[0],
                    "right_y": right[1],
                }
            )

        tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_callback, as_dictionary=True)

        start = time.time()
        allow_skip = False
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)
                        pygame.quit()
                        return
                    if event.key == pygame.K_SPACE and allow_skip:
                        running = False
            screen.fill((0, 0, 0))
            screen.blit(img, img_rect)
            pygame.display.flip()

            if time.time() - start >= min_dur:
                allow_skip = True
            if time.time() - start >= max_dur:
                running = False

        tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)
        shown.append(path)
        _write_csv(CSV_DIR / f"P{subject}_{path.stem}.csv", samples)

        if half and idx + 1 == half:
            font = pygame.font.SysFont(None, 36)
            text = font.render("Paused, press 'R' to resume", True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            paused = True
            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        if event.key == pygame.K_r:
                            calibrate_tracker(tracker, screen)
                            paused = False
                screen.fill((0, 0, 0))
                screen.blit(text, text_rect)
                pygame.display.flip()
                time.sleep(0.1)

    _run_validity_check(screen, subject, shown)
    pygame.quit()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    _ = argv  # unused for now
    subject, num, min_d, max_d = prompt_experiment_params()
    run_experiment(subject, num, min_d, max_d)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
