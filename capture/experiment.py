"""Experiment capture utilities using Tobii eye tracker."""
from __future__ import annotations

import csv
import time
import random
from pathlib import Path
from typing import Dict, List

# Resolve the repository root to load data regardless of the cwd
BASE_DIR = Path(__file__).resolve().parents[1]

import pygame
try:
    import tobii_research as tr
except Exception:  # pragma: no cover - library not available in tests
    tr = None  # type: ignore


def calibrate_tracker(tracker: "tr.EyeTracker", screen: pygame.Surface) -> None:
    """Calibrate the Tobii eye tracker using a simple five point procedure."""
    if tr is None:
        raise RuntimeError("tobii_research library is not available")

    calib = tr.ScreenBasedCalibration(tracker)
    calib.enter_calibration_mode()

    screen_rect = screen.get_rect()
    points = [
        (0.5, 0.5),  # center
        (0.1, 0.1),  # top-left
        (0.9, 0.1),  # top-right
        (0.1, 0.9),  # bottom-left
        (0.9, 0.9),  # bottom-right
    ]

    for x_rel, y_rel in points:
        x = int(screen_rect.width * x_rel)
        y = int(screen_rect.height * y_rel)

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


def _load_images() -> List[Path]:
    """Load stimulus image file paths from the ``data/current_images`` folder."""
    folder = BASE_DIR / "data" / "current_images"
    # Ensure the folder exists so users know where to place images
    folder.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    paths.extend(sorted(folder.glob("*.jpg")))
    paths.extend(sorted(folder.glob("*.png")))
    random.shuffle(paths)
    return paths


def _load_control_images() -> List[Path]:
    """Load control image file paths from the ``data/control_images`` folder."""
    folder = BASE_DIR / "data" / "control_images"
    # Ensure the folder exists so users know where to place images
    folder.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    paths.extend(sorted(folder.glob("*.jpg")))
    paths.extend(sorted(folder.glob("*.png")))
    random.shuffle(paths)
    return paths


def _write_samples(csv_path: Path, samples: List[Dict[str, float]]) -> None:
    """Write gaze samples to CSV."""
    if not samples:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)

def prompt_experiment_params() -> tuple[str, int, float, float]:
    """Interactively ask the user for experiment parameters."""
    subject_id = input("Subject ID: ").strip()
    num_images = int(input("Number of images to present: "))
    min_duration = float(input("Minimum display duration (s): "))
    max_duration = float(input("Maximum display duration (s): "))
    return subject_id, num_images, min_duration, max_duration


def run_experiment(
    subject_id: str,
    num_images: int,
    min_duration_s: float,
    max_duration_s: float,
  
) -> None:
    """Run the gaze recording experiment for a subject.

    If a Tobii eye tracker is available it will be used, otherwise the
    experiment runs in a simulated mode that simply displays the images and
    records timestamps without gaze coordinates. Images can be skipped with the
    space bar after ``min_duration_s`` seconds and will automatically advance
    after ``max_duration_s`` seconds.
    """

    tracker = None
    if tr is not None:
        try:
            trackers = tr.find_all_eyetrackers()
            if trackers:
                tracker = trackers[0]
        except Exception:  # pragma: no cover - fail gracefully without tracker
            tracker = None

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    calibrate_tracker(tracker, screen)

    current_paths = _load_images()
    control_paths = _load_control_images()
    image_paths = current_paths + control_paths

    output_dir = BASE_DIR / "data" / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print("No stimulus images found in data/current_images or data/control_images")
        start = time.time()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            screen.fill((0, 0, 0))
            pygame.display.flip()

            if time.time() - start >= max_duration_s:

                running = False

        pygame.quit()
        return

    pause_after = len(image_paths) // 2

    for i, img_path in enumerate(image_paths):

        gaze_samples: List[Dict[str, float]] = []

        if tracker:
            def gaze_callback(gaze_data: Dict) -> None:
                left = gaze_data.get("left_gaze_point_on_display_area") or (None, None)
                right = gaze_data.get("right_gaze_point_on_display_area") or (None, None)
                gaze_samples.append({
                    "system_time": gaze_data.get("system_time_stamp"),
                    "device_time": gaze_data.get("device_time_stamp"),
                    "left_x": left[0],
                    "left_y": left[1],
                    "right_x": right[0],
                    "right_y": right[1],
                })

            tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_callback, as_dictionary=True)
        else:
            # Fallback sample collection when no tracker is present
            def gaze_callback() -> None:
                gaze_samples.append({
                    "system_time": time.time(),
                    "device_time": None,
                    "left_x": None,
                    "left_y": None,
                    "right_x": None,
                    "right_y": None,
                })

        img = pygame.image.load(str(img_path))
        screen_rect = screen.get_rect()
        iw, ih = img.get_size()
        scale = min(screen_rect.width / iw, screen_rect.height / ih)
        new_size = (int(iw * scale), int(ih * scale))
        img = pygame.transform.smoothscale(img, new_size)
        img_rect = img.get_rect(center=screen_rect.center)

        duration_s = random.uniform(min_duration_s, max_duration_s)
        start = time.time()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if (
                        event.key == pygame.K_SPACE
                        and time.time() - start >= min_duration_s
                    ):
                        running = False

            screen.fill((0, 0, 0))
            screen.blit(img, img_rect)
            pygame.display.flip()

            # Collect a sample roughly every frame if no tracker
            if not tracker:
                gaze_callback()

            if time.time() - start >= max_duration_s:
                running = False

        if tracker:
            tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)  # type: ignore

        csv_name = f"P{subject_id}_{img_path.stem}.csv"
        _write_samples(output_dir / csv_name, gaze_samples)

        # Pause halfway through the stimuli
        if pause_after and i + 1 == pause_after:
            font = pygame.font.Font(None, 36)
            text = font.render("Paused, press 'R' to resume.", True, (255, 255, 255))
            text_rect = text.get_rect(center=screen.get_rect().center)
            screen.fill((0, 0, 0))
            screen.blit(text, text_rect)
            pygame.display.flip()

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
                            if tracker:
                                calibrate_tracker(tracker, screen)
                            paused = False
                time.sleep(0.1)

    pygame.quit()


def main(argv: List[str] | None = None) -> int:
    """Command line entry point to run the experiment."""
    _ = argv  # unused but kept for backward compatibility
    subject_id, num_images, min_dur, max_dur = prompt_experiment_params()
    run_experiment(subject_id, num_images, min_dur, max_dur)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

