"""Experiment capture utilities using Tobii eye tracker."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

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
    """Load stimulus image file paths from default folders."""
    paths: List[Path] = []
    for folder in [Path("data/current_images"), Path("data/control_images")]:
        if folder.exists():
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
    """Run the gaze recording experiment for a subject."""
    if tr is None:
        raise RuntimeError("tobii_research library is not available")

    trackers = tr.find_all_eyetrackers()
    if not trackers:
        raise RuntimeError("No Tobii eye tracker found")
    tracker = trackers[0]

    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.mouse.set_visible(False)

    calibrate_tracker(tracker, screen)

    image_paths = _load_images()
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    for img_path in image_paths:
        gaze_samples: List[Dict[str, float]] = []

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

            if time.time() - start >= duration_s:
                running = False

        tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)

        csv_name = f"{subject_id}_{img_path.stem}.csv"
        _write_samples(output_dir / csv_name, gaze_samples)

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

