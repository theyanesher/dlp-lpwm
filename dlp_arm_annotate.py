"""Collect per-particle robot-arm labels to train the arm classifier.

Samples every Nth frame of a video (default: 1 in 20), seeds each sampled
frame's particle selection from a SAM 2 mask propagated across the whole
video, and lets you confirm or correct it by clicking before saving. Each
confirmed frame is appended to annotations.pt in the exact schema
particle_arm_pipeline.py's `train` subcommand expects, so afterwards:

    python particle_arm_pipeline.py train --annotations <output>/annotations.pt --output classifier.pth

trains the MLP directly on the result. Re-running against the same --output
resumes: already-confirmed frames (by video + timestamp) are skipped.

Run this file from the `lpwm` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from particle_arm_pipeline import (
    _device,
    _without_time,
    load_lpwm,
    particle_indices_in_mask,
    preprocess_frame,
)
from dlp_particle_removal import build_canvas, decode_with_removed, sam_prompt_and_propagate

HELP_TEXT = "click: toggle | Enter: confirm & next | B: back | C: clear | Q: stop & save"


def collect_labels(model, config: dict, video: Path, device: torch.device,
                   masks: dict[int, np.ndarray], sample_stride: int, start_time: float,
                   display_size: int, dilation: int, fps: float,
                   skip_times: set[float]) -> list[dict]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    start_index = min(frame_count - 1, max(0, round(start_time * fps)))
    sample_indices = [i for i in range(start_index, frame_count, sample_stride)]

    records: list[dict] = []
    cursor = 0
    window = "DLP arm-label collection"
    mouse = {"click": None}

    def on_mouse(event, x, y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
    except cv2.error as error:
        capture.release()
        raise RuntimeError(
            "Could not open the annotation popup. Run in a graphical desktop session."
        ) from error

    cached_index = -1
    frame = positions = enc = None
    selected: set[int] = set()
    try:
        while cursor < len(sample_indices):
            index = sample_indices[cursor]
            time_sec = index / fps

            if str(video.resolve()) + f"@{time_sec:.6f}" in skip_times:
                cursor += 1
                continue

            if cached_index != index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {index} from {video}")
                image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
                with torch.inference_mode():
                    enc = model.encode_all(image, deterministic=True)
                positions = _without_time(enc["z"])[0]
                cached_index = index
                mask = masks.get(index)
                selected = set(particle_indices_in_mask(
                    positions, mask, model.kp_range, dilation)) if mask is not None else set()

            removed_rgb = decode_with_removed(model, enc, selected)
            canvas, points = build_canvas(frame, positions, selected, removed_rgb, model.kp_range, display_size)
            status = (f"sample {cursor + 1}/{len(sample_indices)}  frame {index + 1}/{frame_count}  "
                      f"{time_sec:.2f}s  confirmed: {len(records)}  selected: {sorted(selected)}")
            cv2.putText(canvas, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .52,
                       (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, HELP_TEXT, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .45,
                       (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKeyEx(30)
            click = mouse.pop("click", None)
            mouse["click"] = None
            if click is not None and len(points):
                distances = np.linalg.norm(points - np.asarray(click), axis=1)
                nearest = int(distances.argmin())
                if distances[nearest] <= 22:
                    if nearest in selected:
                        selected.remove(nearest)
                    else:
                        selected.add(nearest)

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                selected = set()
            elif key in (ord("b"), ord("B")):
                if records:
                    records.pop()
                    cursor = max(0, cursor - 1)
                    cached_index = -1
            elif key in (10, 13):
                features = _without_time(enc["z_features"])[0]
                labels = [int(particle_index in selected) for particle_index in range(len(features))]
                records.append({
                    "video": str(video.resolve()), "time_sec": time_sec,
                    "features": features.detach().cpu(),
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "positions": positions.detach().cpu(),
                })
                cursor += 1
                cached_index = -1
    finally:
        capture.release()
        cv2.destroyWindow(window)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing hparams.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory to hold annotations.pt")
    parser.add_argument("--sample-stride", type=int, default=20,
                        help="Label 1 in every N frames (default: 20)")
    parser.add_argument("--time", type=float, default=0.0, help="Timestamp for the SAM prompt and first sample")
    parser.add_argument("--display-size", type=int, default=1000)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-config", required=True,
                        help="SAM 2 config name, e.g. configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam-device", default="auto", help="SAM device (default: auto)")
    parser.add_argument("--mask-dilation", type=int, default=0,
                        help="Expand the SAM mask by this many pixels before selecting particle centers")
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be at least 1")

    device = _device(args.device)
    model, config, checkpoint = load_lpwm(args.model_dir, args.checkpoint, device, model_type='gdlp')
    print(f"Loaded model checkpoint {checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output / "annotations.pt"
    existing = torch.load(dataset_path, weights_only=False) if dataset_path.exists() else []
    skip_times = {f"{record['video']}@{record['time_sec']:.6f}" for record in existing}
    print(f"Resuming with {len(existing)} previously confirmed frames." if existing
         else "Starting a new annotation set.")

    print("Prompting SAM 2 on the arm; its mask will propose a label for every sampled frame.")
    masks = sam_prompt_and_propagate(
        model, config, args.video, device, args.sam_checkpoint, args.sam_config,
        _device(args.sam_device), args.time, args.display_size, args.mask_dilation,
        args.output / "sam_particle_sample.png")
    if masks is None:
        print("SAM prompt cancelled; nothing saved.")
        return

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    print(f"Confirm or correct the label on every {args.sample_stride}th frame. {HELP_TEXT}")
    new_records = collect_labels(
        model, config, args.video, device, masks, args.sample_stride, args.time,
        args.display_size, args.mask_dilation, fps, skip_times)

    records = existing + new_records
    if not new_records:
        print("No new frames were confirmed; nothing saved.")
        return
    torch.save(records, dataset_path)
    metadata = {
        "model_checkpoint": str(checkpoint), "frames": len(records),
        "sample_stride": args.sample_stride, "videos": sorted({r["video"] for r in records}),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Confirmed {len(new_records)} new frames ({len(records)} total). Saved {dataset_path.resolve()}")


if __name__ == "__main__":
    main()
