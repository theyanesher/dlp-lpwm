"""Run DLP + arm classifier inference on a video.

For every frame, encodes it with DLP, classifies each particle as
arm/not-arm with the trained MLP, and writes a side-by-side video:
left half is the source frame with every keypoint drawn (arm-classified
particles in red, the rest in cyan), right half is the DLP reconstruction
decoded from all particles.

Run this file from the `lpwm` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from particle_arm_pipeline import (
    _device,
    _without_time,
    decoded_rgb,
    load_classifier,
    load_lpwm,
    particle_pixels,
    preprocess_frame,
)

DEFAULT_CLASSIFIER = Path("/home/theyanesh/lpwm_masking/lpwm/dlp_arm_dataset/classifier.pth")

ARM_COLOR = (0, 0, 255)      # red, BGR
OTHER_COLOR = (255, 255, 0)  # cyan, BGR


def draw_keypoints(frame: np.ndarray, positions: torch.Tensor, arm: torch.Tensor, kp_range) -> np.ndarray:
    height, width = frame.shape[:2]
    points = particle_pixels(positions, width, height, kp_range)
    canvas = frame.copy()
    for index, (x, y) in enumerate(points):
        center = (round(x), round(y))
        color = ARM_COLOR if arm[index] else OTHER_COLOR
        cv2.circle(canvas, center, 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return canvas


def run_inference(model, classifier, threshold: float, config: dict, video: Path,
                  output: Path, device: torch.device) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    image_size = config["image_size"]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (image_size * 2, image_size))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = preprocess_frame(frame, image_size, device, model.normalize_rgb)
            with torch.inference_mode():
                enc = model.encode_all(image, deterministic=True)
                arm_full = classifier(enc["z_features"]).sigmoid() >= threshold
                keep = enc["obj_on"].clone()
                keep[arm_full.unsqueeze(-1)] = 0
                dec = model.decode_all(enc["z"], enc["z_scale"], enc["z_features"], keep,
                                       enc["z_depth"], enc["z_bg_features"], None)
            positions = _without_time(enc["z"])[0]
            arm = _without_time(arm_full.unsqueeze(-1)).squeeze(-1)[0].cpu()

            source = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            annotated = draw_keypoints(source, positions, arm, model.kp_range)

            recon = (decoded_rgb(dec) * 255).round().clip(0, 255).astype(np.uint8)
            recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
            if recon_bgr.shape[:2] != (image_size, image_size):
                recon_bgr = cv2.resize(recon_bgr, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            writer.write(np.hstack([annotated, recon_bgr]))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return frame_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing hparams.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--threshold", type=float, help="Override the classifier's saved threshold")
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.classifier.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {args.classifier}")

    device = _device(args.device)
    model, config, checkpoint = load_lpwm(args.model_dir, args.checkpoint, device, model_type="gdlp")
    print(f"Loaded DLP checkpoint {checkpoint}")
    classifier, saved_threshold = load_classifier(args.classifier, device)
    threshold = saved_threshold if args.threshold is None else args.threshold
    print(f"Loaded arm classifier {args.classifier} (threshold={threshold})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = run_inference(model, classifier, threshold, config, args.video, args.output, device)
    print(f"Wrote {frame_count} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
