"""Experiment capture utilities using Tobii eye tracker."""
from __future__ import annotations

import csv
import time
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
        (0.1, 0.1),
        (0.9, 0.1),
        (0.5, 0.5),
        (0.1, 0.9),
        (0.9, 0.9),
    ]
    for x_rel, y_rel in points:
        x = int(screen_rect.width * x_rel)
        y = int(screen_rect.height * y_rel)
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 20)
        pygame.display.flip()
        time.sleep(0.5)
        calib.collect_data(x_rel, y_rel)
        time.sleep(0.2)

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
    return paths


def _write_samples(csv_path: Path, samples: List[Dict[str, float]]) -> None:
    """Write gaze samples to CSV."""
    if not samples:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)


def run_experiment(subject_id: str, duration_s: float = 5.0) -> None:
    """Run the gaze recording experiment for a subject.

    If a Tobii eye tracker is available it will be used, otherwise the
    experiment runs in a simulated mode that simply displays the images and
    records timestamps without gaze coordinates.
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
    screen = pygame.display.set_mode((1280, 720))
    pygame.mouse.set_visible(False)

    if tracker:
        calibrate_tracker(tracker, screen)
    else:
        print("Running without eye tracker - demo mode")

    image_paths = _load_images()
    output_dir = BASE_DIR / "data" / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print("No stimulus images found in data/current_images")
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

            if time.time() - start >= duration_s:
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
                    "system_time_stamp": gaze_data.get("system_time_stamp"),
                    "device_time_stamp": gaze_data.get("device_time_stamp"),
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
                    "system_time_stamp": time.time(),
                    "device_time_stamp": None,
                    "left_x": None,
                    "left_y": None,
                    "right_x": None,
                    "right_y": None,
                })

        img = pygame.image.load(str(img_path))
        img_rect = img.get_rect(center=screen.get_rect().center)

        start = time.time()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            screen.fill((0, 0, 0))
            screen.blit(img, img_rect)
            pygame.display.flip()

            # Collect a sample roughly every frame if no tracker
            if not tracker:
                gaze_callback()

            if time.time() - start >= duration_s:
                running = False

        if tracker:
            tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)  # type: ignore

        csv_name = f"{subject_id}_{img_path.stem}.csv"
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
    import argparse

    parser = argparse.ArgumentParser(description="Run eye tracking experiment")
    parser.add_argument("subject_id", type=str, help="Subject identifier")
    args = parser.parse_args(argv)

    run_experiment(args.subject_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

