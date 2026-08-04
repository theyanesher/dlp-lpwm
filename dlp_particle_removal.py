"""Interactively remove DLP particles from the frames of a video.

For each frame, click particle centers to toggle them; the right-hand panel
re-decodes live with the selected particles suppressed. Accepting writes an
mp4 stitched from every frame in the video, using each frame's selection
(frames you never visited keep every particle).

Pass --sam-checkpoint/--sam-config to seed the selection from SAM 2 instead of
clicking every frame by hand: prompt SAM once on the arm, its propagated mask
pre-selects the particles inside it on every frame, and you can still click to
correct individual frames before accepting.

Run this file from the `lpwm` directory. Unlike particle_arm_pipeline.py this
only touches a single DLP frame at a time - no arm classifier.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from particle_arm_pipeline import (
    _device,
    _sam_mask,
    _without_time,
    decoded_rgb,
    extract_video_frames,
    load_lpwm,
    particle_indices_in_mask,
    particle_pixels,
    preprocess_frame,
    prompt_sam_sample,
)

HEADER_HEIGHT = 58


def decode_with_removed(model, enc: dict, indices: set[int]) -> np.ndarray:
    keep = enc["obj_on"].clone()
    if indices:
        keep[..., sorted(indices), :] = 0
    with torch.inference_mode():
        dec = model.decode_all(
            enc["z"], enc["z_scale"], enc["z_features"], keep,
            enc["z_depth"], enc["z_bg_features"], None,
        )
    return decoded_rgb(dec)


def build_canvas(frame: np.ndarray, positions: torch.Tensor, selected: set[int],
                 removed_rgb: np.ndarray, kp_range, display_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Side-by-side clickable source (left) and live decode with removals (right)."""
    height, width = frame.shape[:2]
    points = particle_pixels(positions, width, height, kp_range)
    scale = display_size / max(height, width)
    body_size = (round(width * scale), round(height * scale))
    left = cv2.resize(frame, body_size, interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    scaled_points = points * scale
    for index, (x, y) in enumerate(scaled_points):
        center = (round(x), round(y))
        chosen = index in selected
        color = (0, 80, 255) if chosen else (0, 255, 255)
        cv2.circle(left, center, 8 if chosen else 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(left, center, 10 if chosen else 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(left, str(index), (center[0] + 9, center[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(left, str(index), (center[0] + 9, center[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)

    right = cv2.cvtColor((removed_rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    right = cv2.resize(right, body_size, interpolation=cv2.INTER_NEAREST)

    body = np.hstack([left, right])
    canvas = cv2.copyMakeBorder(body, HEADER_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(canvas, "source (click particles)", (10, HEADER_HEIGHT - 40),
               cv2.FONT_HERSHEY_SIMPLEX, .45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "decode with selection removed", (body_size[0] + 10, HEADER_HEIGHT - 40),
               cv2.FONT_HERSHEY_SIMPLEX, .45, (200, 200, 200), 1, cv2.LINE_AA)
    scaled_points[:, 1] += HEADER_HEIGHT
    return canvas, scaled_points


def annotate_video(model, config: dict, video: Path, device: torch.device,
                   start_time: float, display_size: int,
                   masks: dict[int, np.ndarray] | None = None,
                   dilation: int = 0) -> tuple[dict[int, list[int]], float] | None:
    """Step through a video, selecting particles to suppress on each frame.

    If ``masks`` (per-frame SAM masks) is given, each frame's selection is
    pre-seeded with the particles centered inside its mask; clicks still
    toggle particles on top of that starting point.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    index = min(frame_count - 1, max(0, round(start_time * fps)))

    selections: dict[int, set[int]] = {}
    playing = False
    window = "DLP particle removal"
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
    try:
        while True:
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
                if masks is not None and index not in selections:
                    mask = masks.get(index)
                    if mask is not None:
                        selections[index] = set(particle_indices_in_mask(
                            positions, mask, model.kp_range, dilation))

            selected = selections.get(index, set())
            removed_rgb = decode_with_removed(model, enc, selected)
            canvas, points = build_canvas(frame, positions, selected, removed_rgb, model.kp_range, display_size)
            source_label = "SAM-seeded" if masks is not None else "manual"
            status = (f"frame {index + 1}/{frame_count}  {index / fps:.2f}s  {source_label}  "
                      f"selected: {sorted(selected)}  labeled frames: {len(selections)}")
            help_text = "click: toggle | space: play | A/D: seek | C: clear | Enter: decode all | Q: cancel"
            cv2.putText(canvas, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .52,
                       (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, help_text, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .45,
                       (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None

            delay = max(1, round(1000 / fps)) if playing else 30
            key = cv2.waitKeyEx(delay)
            click = mouse.pop("click", None)
            mouse["click"] = None
            if click is not None and len(points):
                distances = np.linalg.norm(points - np.asarray(click), axis=1)
                nearest = int(distances.argmin())
                if distances[nearest] <= 22:
                    selected = selections.setdefault(index, set())
                    if nearest in selected:
                        selected.remove(nearest)
                    else:
                        selected.add(nearest)

            if key in (ord("q"), ord("Q"), 27):
                return None
            if key in (10, 13):
                return {i: sorted(values) for i, values in selections.items()}, fps
            if key in (ord("c"), ord("C")):
                selections[index] = set()
            if key == ord(" "):
                playing = not playing
            elif key in (ord("a"), ord("A"), 2424832, 65361):
                index = max(0, index - 1)
                playing = False
            elif key in (ord("d"), ord("D"), 2555904, 65363):
                index = min(frame_count - 1, index + 1)
                playing = False
            elif key in (2162688, 65360):
                index, playing = 0, False
            elif key in (2293760, 65367):
                index, playing = frame_count - 1, False
            elif playing:
                index += 1
                if index >= frame_count:
                    index, playing = frame_count - 1, False
    finally:
        capture.release()
        cv2.destroyWindow(window)


def decode_video_with_selections(model, config: dict, video: Path,
                                 selections: dict[int, list[int]], destination: Path,
                                 device: torch.device, fps: float) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (config["image_size"], config["image_size"]),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {destination}")
    frames_written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
            with torch.inference_mode():
                enc = model.encode_all(image, deterministic=True)
            removed_rgb = decode_with_removed(model, enc, set(selections.get(frames_written, [])))
            bgr = cv2.cvtColor((removed_rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frames_written += 1
    finally:
        capture.release()
        writer.release()
    return frames_written


def propagate_sam_masks(predictor, state, prompt_frame: int) -> dict[int, np.ndarray]:
    """Track the prompted object both ways from prompt_frame and collect its mask per frame."""
    masks: dict[int, np.ndarray] = {}
    passes = ((False, None), (True, prompt_frame + 1)) if prompt_frame > 0 else ((False, None),)
    for reverse, limit in passes:
        kwargs = {"start_frame_idx": prompt_frame, "reverse": reverse}
        if limit is not None:
            kwargs["max_frame_num_to_track"] = limit
        for frame_index, object_ids, logits in predictor.propagate_in_video(state, **kwargs):
            object_ids = object_ids.tolist() if hasattr(object_ids, "tolist") else list(object_ids)
            if 1 in object_ids:
                masks[int(frame_index)] = _sam_mask(logits[object_ids.index(1):object_ids.index(1) + 1])
    return masks


def sam_prompt_and_propagate(model, config: dict, video: Path, device: torch.device,
                             sam_checkpoint: Path, sam_config: str, sam_device: torch.device,
                             start_time: float, display_size: int, dilation: int,
                             sample_path: Path) -> dict[int, np.ndarray] | None:
    """Prompt SAM 2 once on the arm and propagate the mask across the whole video."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise RuntimeError(
            "SAM 2 is not installed. Install Meta's `sam2` package in this environment "
            "(see docs/arm_particle_pipeline.md) or omit --sam-checkpoint for manual selection."
        ) from error

    print("Loading SAM 2 video predictor ...")
    predictor = build_sam2_video_predictor(sam_config, str(sam_checkpoint), device=str(sam_device))

    with tempfile.TemporaryDirectory(prefix="dlp_sam_frames_") as temp_path:
        frame_dir = Path(temp_path)
        print("Extracting video frames for SAM 2 ...")
        fps, frame_count = extract_video_frames(video, frame_dir)
        prompt_frame = min(frame_count - 1, max(0, round(start_time * fps)))
        frame = cv2.imread(str(frame_dir / f"{prompt_frame:08d}.jpg"))
        if frame is None:
            raise RuntimeError(f"Could not load SAM prompt frame {prompt_frame}")
        image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
        with torch.inference_mode():
            enc = model.encode_all(image, deterministic=True)
        positions = _without_time(enc["z"])[0]

        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if sam_device.type == "cuda" else contextlib.nullcontext())
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(frame_dir))
            sample = prompt_sam_sample(
                predictor, state, frame, prompt_frame, positions, model.kp_range,
                sample_path, display_size, dilation)
            if sample is None:
                return None
            print(f"Saved SAM prompt sample to {sample_path.resolve()}")
            print("Propagating the SAM mask across the video ...")
            masks = propagate_sam_masks(predictor, state, prompt_frame)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing hparams.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--time", type=float, default=0.0, help="Initial popup timestamp in seconds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display-size", type=int, default=1000,
                        help="Popup size along its longest image edge (default: 1000)")
    parser.add_argument("--sam-checkpoint", type=Path,
                        help="SAM 2 checkpoint; if given, seeds each frame's selection from a propagated mask")
    parser.add_argument("--sam-config", help="SAM 2 config name, e.g. configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam-device", default="auto", help="SAM device (default: auto)")
    parser.add_argument("--mask-dilation", type=int, default=0,
                        help="Expand the SAM mask by this many pixels before selecting particle centers")
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if bool(args.sam_checkpoint) != bool(args.sam_config):
        raise ValueError("--sam-checkpoint and --sam-config must be given together")

    device = _device(args.device)
    model, config, checkpoint = load_lpwm(args.model_dir, args.checkpoint, device, model_type='gdlp')
    print(f"Loaded model checkpoint {checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    masks = None
    if args.sam_checkpoint is not None:
        if not args.sam_checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
        print("Prompting SAM 2 on the arm; its mask will pre-select particles on every frame.")
        masks = sam_prompt_and_propagate(
            model, config, args.video, device, args.sam_checkpoint, args.sam_config,
            _device(args.sam_device), args.time, args.display_size, args.mask_dilation,
            args.output / "sam_particle_sample.png")
        if masks is None:
            print("SAM prompt cancelled.")
            return

    print("Opening per-frame annotation. Click particles to add or remove them from the selection.")
    result = annotate_video(model, config, args.video, device, args.time, args.display_size,
                            masks=masks, dilation=args.mask_dilation)
    if result is None:
        print("Annotation cancelled; no video was decoded.")
        return
    selections, fps = result
    if not any(selections.values()):
        print("No particles were selected; no video was decoded.")
        return

    selection_path = args.output / "selections.json"
    selection_path.write_text(json.dumps({
        "video": str(args.video.resolve()),
        "model_checkpoint": str(checkpoint),
        "frame_selections": {
            str(index): values for index, values in sorted(selections.items())
        },
    }, indent=2) + "\n")
    print(f"Wrote {selection_path.resolve()}")

    output_video = args.output / f"{args.video.stem}_particles_removed.mp4"
    print(f"Decoding all frames to {output_video.resolve()} ...")
    frames_written = decode_video_with_selections(
        model, config, args.video, selections, output_video, device, fps)
    print(f"Wrote {frames_written} frames to {output_video.resolve()}")


if __name__ == "__main__":
    main()
