"""Annotate LPWM particles, train an arm classifier, and remove arm particles.

Run this file from the ``lpwm`` directory.  See docs/arm_particle_pipeline.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_lpwm_video_prediction import load_dlp_from_config


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class ArmParticleClassifier(nn.Module):
    """Binary MLP over one particle's LPWM visual feature vector."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, hidden_layers: int = 3):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        layers: list[nn.Module] = [nn.Linear(feature_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_lpwm(model_dir: Path, checkpoint: Path | None, device: torch.device,
              model_type: str = 'gddlp'):
    config_path = model_dir / "hparams.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LPWM config not found: {config_path}")
    with config_path.open() as stream:
        config = json.load(stream)
    model = load_dlp_from_config(str(config_path), ckpt_path=None, model_type=model_type)
    if checkpoint is None:
        # train_dlp_accelerate.py names checkpoint files "gddlp" even for plain DLP runs
        # (run directories are correctly named "gdlp"), so accept either spelling here.
        prefixes = dict.fromkeys((model_type, "gddlp"))
        candidates = [
            model_dir / "saves" / f"{config['ds']}_{prefix}{suffix}"
            for prefix in prefixes
            for suffix in ("_best_lpips.pth", ".pth")
        ]
        checkpoint = next((path for path in candidates if path.is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError("No default checkpoint found; pass --checkpoint explicitly")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return model, config, checkpoint


def iter_videos(folder: Path, recursive: bool) -> Iterable[Path]:
    paths = folder.rglob("*") if recursive else folder.glob("*")
    yield from sorted(path for path in paths if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def preprocess_frame(frame_bgr: np.ndarray, image_size: int, device: torch.device,
                     normalize_rgb: bool = False) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
    if normalize_rgb:
        image = image.mul(2).sub(1)
    return image.unsqueeze(0).to(device)


def encode_decode(model, image: torch.Tensor):
    enc = model.encode_all(image, deterministic=True)
    dec = model.decode_all(
        enc["z"], enc["z_scale"], enc["z_features"], enc["obj_on"],
        enc["z_depth"], enc["z_bg_features"], None,
    )
    return enc, dec


def _without_time(x: torch.Tensor) -> torch.Tensor:
    # LPWM generally returns [B,T,N,...], but static models may return [B,N,...].
    return x[:, 0] if x.ndim >= 4 else x


def make_contact_sheet(frame: np.ndarray, glimpses: torch.Tensor,
                       destination: Path, pages: int = 4) -> list[Path]:
    # The decoder flattens B and T, so this is [B*T,N,4,H,W].
    glimpses = glimpses[0].detach().cpu().clamp(0, 1)
    count = len(glimpses)
    page_count = min(max(1, pages), count)
    page_indices = np.array_split(np.arange(count), page_count)
    paths = []
    for page, indices in enumerate(page_indices, start=1):
        path = destination.with_name(
            f"{destination.stem}_{page}_of_{page_count}{destination.suffix}")
        cols = min(8, len(indices))
        rows = math.ceil(len(indices) / cols) + 1
        fig = plt.figure(figsize=(2.2 * cols, 2.2 * rows))
        ax = fig.add_subplot(rows, 1, 1)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{destination.stem} — page {page} of {page_count}")
        ax.axis("off")
        for slot, index in enumerate(indices):
            glimpse = glimpses[index]
            ax = fig.add_subplot(rows, cols, cols + slot + 1)
            rgba = glimpse.permute(1, 2, 0).numpy()
            rgb = rgba[..., 1:] if rgba.shape[-1] == 4 else rgba
            ax.imshow(rgb)
            if rgba.shape[-1] == 4:
                ax.imshow(np.ones_like(rgb), alpha=1 - rgba[..., 0])
            ax.set_title(str(index))
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths

def make_particle_overlays(frame: np.ndarray, positions: torch.Tensor,
                           destination: Path, kp_range=(-1, 1), pages: int = 4) -> list[Path]:
    """Draw indexed particle centers on the source frame, plus less-crowded pages."""
    positions = positions.detach().cpu().numpy()
    height, width = frame.shape[:2]
    low, high = kp_range
    y = np.clip((positions[:, 0] - low) / (high - low) * (height - 1), 0, height - 1)
    x = np.clip((positions[:, 1] - low) / (high - low) * (width - 1), 0, width - 1)
    indices = np.arange(len(positions))
    groups = [indices, *np.array_split(indices, min(max(1, pages), len(indices)))]
    paths = []
    for page, group in enumerate(groups):
        suffix = "all" if page == 0 else f"{page}_of_{len(groups) - 1}"
        path = destination.with_name(f"{destination.stem}_{suffix}{destination.suffix}")
        scale = max(2, math.ceil(900 / max(height, width)))
        canvas = cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_LANCZOS4)
        for index in group:
            center = (round(x[index] * scale), round(y[index] * scale))
            cv2.circle(canvas, center, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, center, 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, str(index), (center[0] + 7, center[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (center[0] + 7, center[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), canvas)
        paths.append(path)
    return paths


def particle_pixels(positions: torch.Tensor | np.ndarray, width: int, height: int,
                    kp_range=(-1, 1)) -> np.ndarray:
    """Convert LPWM ``(y, x)`` coordinates to source-image ``(x, y)`` pixels."""
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    low, high = kp_range
    if high <= low:
        raise ValueError("kp_range upper bound must be greater than its lower bound")
    y = np.clip((positions[:, 0] - low) / (high - low) * (height - 1), 0, height - 1)
    x = np.clip((positions[:, 1] - low) / (high - low) * (width - 1), 0, width - 1)
    return np.column_stack((x, y))


def particle_indices_in_mask(positions: torch.Tensor | np.ndarray, mask: np.ndarray,
                             kp_range=(-1, 1), dilation: int = 0) -> list[int]:
    """Return particle indices whose center pixels fall inside a binary mask."""
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D SAM mask, got shape {mask.shape}")
    mask = mask.astype(np.uint8)
    if dilation > 0:
        size = 2 * dilation + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    height, width = mask.shape
    points = particle_pixels(positions, width, height, kp_range)
    indices = []
    for index, (x, y) in enumerate(points):
        if mask[round(y), round(x)]:
            indices.append(index)
    return indices


def draw_interactive_overlay(frame: np.ndarray, positions: torch.Tensor,
                             selected: set[int], kp_range=(-1, 1),
                             display_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Draw clickable particle centers and return the canvas and their pixel locations."""
    height, width = frame.shape[:2]
    points = particle_pixels(positions, width, height, kp_range)
    if display_size < 256:
        raise ValueError("display_size must be at least 256 pixels")
    scale = display_size / max(height, width)
    image = cv2.resize(frame, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    # Keep status and controls outside the video so they never cover the arm.
    header_height = 58
    canvas = cv2.copyMakeBorder(
        image, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    scaled_points = points * scale
    scaled_points[:, 1] += header_height
    for index, (x, y) in enumerate(scaled_points):
        center = (round(x), round(y))
        chosen = index in selected
        color = (0, 80, 255) if chosen else (0, 255, 255)
        cv2.circle(canvas, center, 8 if chosen else 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, 10 if chosen else 8, (255, 255, 255), 1,
                   lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(index), (center[0] + 9, center[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (center[0] + 9, center[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas, scaled_points


def select_particles_over_video(model, config: dict, video: Path, device: torch.device,
                                start_time: float = 0.0,
                                display_size: int = 1000) -> tuple[dict[int, list[int]], int] | None:
    """Open a video player in which clicks label particles independently per frame.

    The frame is re-encoded whenever it changes, so the user can inspect how each
    particle slot behaves throughout the clip instead of assuming stable slot IDs.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_index = min(frame_count - 1, max(0, round(start_time * fps)))
    selections: dict[int, set[int]] = {}
    playing = False
    window = "LPWM manual video annotation"
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
            "Could not open the annotation popup. Run in a graphical desktop session "
            "or use --remove for a headless run."
        ) from error

    cached_index = -1
    frame = positions = points = canvas = None
    try:
        while True:
            if cached_index != frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {frame_index} from {video}")
                image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
                with torch.inference_mode():
                    enc = model.encode_all(image, deterministic=True)
                positions = _without_time(enc["z"])[0]
                cached_index = frame_index

            selected = selections.get(frame_index, set())
            canvas, points = draw_interactive_overlay(
                frame, positions, selected, model.kp_range, display_size)
            status = (f"frame {frame_index + 1}/{frame_count}  {frame_index / fps:.2f}s  "
                      f"this frame: {sorted(selected)}  labeled frames: {len(selections)}")
            help_text = "click: toggle this frame | space: play/pause | A/D: seek | C: clear | Enter: decode | Q: cancel"
            cv2.putText(canvas, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .52,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, help_text, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None

            # Keep polling while paused so mouse-only interaction redraws at once.
            delay = max(1, round(1000 / fps)) if playing else 30
            key = cv2.waitKeyEx(delay)
            click = mouse.pop("click", None)
            mouse["click"] = None
            if click is not None and len(points):
                distances = np.linalg.norm(points - np.asarray(click), axis=1)
                nearest = int(distances.argmin())
                if distances[nearest] <= 22:
                    selected = selections.setdefault(frame_index, set())
                    if nearest in selected:
                        selected.remove(nearest)
                    else:
                        selected.add(nearest)
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key in (10, 13):
                return ({index: sorted(values) for index, values in selections.items()},
                        frame_index)
            if key in (ord("c"), ord("C")):
                selections[frame_index] = set()
            if key == ord(" "):
                playing = not playing
            elif key in (ord("a"), ord("A"), 2424832, 65361):
                frame_index = max(0, frame_index - 1)
                playing = False
            elif key in (ord("d"), ord("D"), 2555904, 65363):
                frame_index = min(frame_count - 1, frame_index + 1)
                playing = False
            elif key in (2162688, 65360):
                frame_index, playing = 0, False
            elif key in (2293760, 65367):
                frame_index, playing = frame_count - 1, False
            elif playing:
                frame_index += 1
                if frame_index >= frame_count:
                    frame_index, playing = frame_count - 1, False
    finally:
        capture.release()
        cv2.destroyWindow(window)



def parse_labels(text: str, count: int) -> list[int]:
    labels = text.replace(",", " ").split()
    if len(labels) != count or any(value not in {"0", "1"} for value in labels):
        raise ValueError(f"enter exactly {count} binary labels")
    return [int(value) for value in labels]


def parse_particle_indices(text: str, count: int) -> list[int]:
    """Parse a comma/space-separated set of particle indices."""
    values = text.replace(",", " ").split()
    try:
        indices = sorted(set(int(value) for value in values))
    except ValueError as error:
        raise ValueError("indices must be integers separated by spaces or commas") from error
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise ValueError(f"indices out of range [0, {count - 1}]: {invalid}")
    return indices


def decoded_rgb(decoder_output: dict) -> np.ndarray:
    """Convert the first decoded LPWM frame to an HWC float RGB array."""
    rgb = decoder_output["rec_rgb"][0]
    if rgb.ndim == 4:
        rgb = rgb[0]
    return rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def save_manual_comparison(source_bgr: np.ndarray, normal_rgb: np.ndarray,
                           removed_rgb: np.ndarray, indices: list[int], destination: Path) -> None:
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = (
        (source_rgb, "Source frame"),
        (normal_rgb, "LPWM reconstruction"),
        (removed_rgb, f"Removed particles: {indices}"),
    )
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def decode_video_with_indices(model, config: dict, video: Path, indices: list[int],
                              destination: Path, device: torch.device) -> int:
    """Decode every frame in the video while suppressing fixed particle slots."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
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
                keep = enc["obj_on"].clone()
                if indices:
                    keep[..., indices, :] = 0
                dec = model.decode_all(
                    enc["z"], enc["z_scale"], enc["z_features"], keep,
                    enc["z_depth"], enc["z_bg_features"], None,
                )
            rgb = decoded_rgb(dec)
            bgr = cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frames_written += 1
    finally:
        capture.release()
        writer.release()
    return frames_written


def decode_video_with_frame_indices(model, config: dict, video: Path,
                                    frame_indices: dict[int, list[int]],
                                    destination: Path, device: torch.device) -> int:
    """Decode a video using independently annotated particle slots per frame."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
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
                keep = enc["obj_on"].clone()
                indices = frame_indices.get(frames_written, [])
                if indices:
                    keep[..., indices, :] = 0
                dec = model.decode_all(
                    enc["z"], enc["z_scale"], enc["z_features"], keep,
                    enc["z_depth"], enc["z_bg_features"], None,
                )
            rgb = decoded_rgb(dec)
            bgr = cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frames_written += 1
    finally:
        capture.release()
        writer.release()
    return frames_written


def extract_video_frames(video: Path, destination: Path) -> tuple[float, int]:
    """Extract numerically named JPEG frames in the layout expected by SAM 2."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            path = destination / f"{count:08d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write temporary SAM frame: {path}")
            count += 1
    finally:
        capture.release()
    if count == 0:
        raise RuntimeError(f"Video contains no readable frames: {video}")
    return fps, count


def _sam_mask(mask_logits: torch.Tensor) -> np.ndarray:
    """Convert one-object SAM logits to a CPU boolean mask."""
    return (mask_logits[0].detach().float().cpu().numpy().squeeze() > 0)


def prompt_sam_sample(predictor, state, frame_bgr: np.ndarray, frame_index: int,
                      positions: torch.Tensor, kp_range, destination: Path,
                      display_size: int, dilation: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Collect SAM foreground/background clicks and confirm a particle-mask sample."""
    window = "SAM 2 Franka prompt and LPWM sample"
    prompts: list[tuple[float, float]] = []
    labels: list[int] = []
    mask = None
    mouse = {"event": None}

    def on_mouse(event, x, y, _flags, _parameter):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            mouse["event"] = (event, x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            height, width = frame_bgr.shape[:2]
            scale = display_size / max(height, width)
            header_height = 66
            image = frame_bgr.copy()
            selected: list[int] = []
            if mask is not None:
                shown_mask = cv2.resize(mask.astype(np.uint8), (width, height),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                tint = np.zeros_like(image)
                tint[shown_mask] = (40, 210, 40)
                image = cv2.addWeighted(image, 1.0, tint, .45, 0)
                selected = particle_indices_in_mask(
                    positions, shown_mask, kp_range, dilation)
            resized = cv2.resize(
                image, (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
            canvas = cv2.copyMakeBorder(
                resized, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            points = particle_pixels(positions, width, height, kp_range) * scale
            points[:, 1] += header_height
            for index, (x, y) in enumerate(points):
                color = (0, 80, 255) if index in selected else (0, 255, 255)
                cv2.circle(canvas, (round(x), round(y)), 7, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, str(index), (round(x) + 8, round(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2, cv2.LINE_AA)
            for (x, y), label in zip(prompts, labels):
                center = (round(x * scale), round(y * scale + header_height))
                cv2.drawMarker(canvas, center, (0, 255, 0) if label else (0, 0, 255),
                               cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            status = f"SAM sample frame {frame_index + 1} | selected LPWM particles: {selected}"
            help_text = "left: arm | right: background | R: reset | Enter: accept sample | Q: cancel"
            cv2.putText(canvas, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .54,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, help_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None

            key = cv2.waitKeyEx(30)
            event = mouse.pop("event", None)
            mouse["event"] = None
            if event is not None:
                button, x, y = event
                image_y = y - header_height
                if 0 <= image_y < resized.shape[0] and 0 <= x < resized.shape[1]:
                    prompts.append((x / scale, image_y / scale))
                    labels.append(1 if button == cv2.EVENT_LBUTTONDOWN else 0)
                    point_array = np.asarray(prompts, dtype=np.float32)
                    label_array = np.asarray(labels, dtype=np.int32)
                    _, _, logits = predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=frame_index, obj_id=1,
                        points=point_array, labels=label_array,
                    )
                    mask = _sam_mask(logits)
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key in (ord("r"), ord("R")):
                predictor.reset_state(state)
                prompts.clear()
                labels.clear()
                mask = None
            if key in (10, 13) and mask is not None and labels.count(1) > 0:
                cv2.imwrite(str(destination), canvas)
                return mask, np.asarray(prompts, dtype=np.float32), np.asarray(labels, dtype=np.int32)
    finally:
        cv2.destroyWindow(window)


def decode_with_sam_masks(model, config: dict, predictor, state, video: Path,
                          destination: Path, device: torch.device, kp_range,
                          prompt_frame: int, dilation: int) -> tuple[int, dict[int, list[int]]]:
    """Propagate SAM masks and suppress LPWM particles centered inside each mask."""
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

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (config["image_size"], config["image_size"]))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {destination}")
    selections: dict[int, list[int]] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                enc = model.encode_all(image, deterministic=True)
                mask = masks.get(frame_index)
                indices = (particle_indices_in_mask(
                    _without_time(enc["z"])[0], mask, kp_range, dilation)
                    if mask is not None else [])
                selections[frame_index] = indices
                keep = enc["obj_on"].clone()
                if indices:
                    keep[..., indices, :] = 0
                dec = model.decode_all(
                    enc["z"], enc["z_scale"], enc["z_features"], keep,
                    enc["z_depth"], enc["z_bg_features"], None,
                )
            rgb = decoded_rgb(dec)
            writer.write(cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return frame_index, selections


def manual_test(args) -> None:
    """Inspect slots over a complete video, then suppress them throughout it."""
    device = _device(args.device)
    model, config, checkpoint = load_lpwm(args.model_dir, args.checkpoint, device)
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.time < 0:
        raise ValueError("--time must be non-negative")

    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read {args.video} at {args.time:.3f}s")

    args.output.mkdir(parents=True, exist_ok=True)
    image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
    with torch.inference_mode():
        enc, normal_dec = encode_decode(model, image)
    particle_count = _without_time(enc["z_features"]).shape[1]
    sheet_path = args.output / "particle_glimpses.png"
    comparison_path = args.output / "manual_comparison.png"
    selection_path = args.output / "selection.json"
    sheet_paths = make_contact_sheet(frame, normal_dec["dec_objects_original_rgb"], sheet_path)
    overlay_paths = make_particle_overlays(
        frame, _without_time(enc["z"])[0], args.output / "particle_overlay.png", model.kp_range)
    normal_rgb = decoded_rgb(normal_dec)

    print("Particle overlays for the initial frame (also useful for --remove runs):")
    for path in overlay_paths:
        print(f"  {path.resolve()}")
    print("Particle sheets:")
    for path in sheet_paths:
        print(f"  {path.resolve()}")
    print(f"Frame has {particle_count} particles (0 through {particle_count - 1}).")
    pending = args.remove
    if pending is None:
        print("Opening per-frame video annotation. Click the arm particles on each frame you label.")
        result = select_particles_over_video(
            model, config, args.video, device, start_time=args.time,
            display_size=args.display_size)
        if result is None:
            print("Annotation cancelled; no video was decoded.")
            return
        frame_indices, preview_frame_index = result
        if not any(frame_indices.values()):
            print("No particles were selected; no video was decoded.")
            return
        indices = frame_indices.get(preview_frame_index, [])

        # Make the still comparison correspond to the frame visible on acceptance.
        capture = cv2.VideoCapture(str(args.video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Could not read preview frame {preview_frame_index}")
        image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
        with torch.inference_mode():
            enc, normal_dec = encode_decode(model, image)
        normal_rgb = decoded_rgb(normal_dec)
    else:
        indices = sorted(set(pending))
        frame_indices = None
        capture = cv2.VideoCapture(str(args.video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        preview_frame_index = round(args.time * fps)
        invalid = [index for index in indices if index < 0 or index >= particle_count]
        if invalid:
            raise ValueError(f"--remove indices out of range [0, {particle_count - 1}]: {invalid}")

    keep = enc["obj_on"].clone()
    if indices:
        keep[..., indices, :] = 0
    with torch.inference_mode():
        removed_dec = model.decode_all(
            enc["z"], enc["z_scale"], enc["z_features"], keep,
            enc["z_depth"], enc["z_bg_features"], None,
        )
    save_manual_comparison(frame, normal_rgb, decoded_rgb(removed_dec), indices, comparison_path)
    selection_data = {
        "video": str(args.video.resolve()), "initial_time_sec": args.time,
        "annotation_scope": "per_frame" if frame_indices is not None else "fixed_slots",
        "preview_frame": preview_frame_index,
        "model_checkpoint": str(checkpoint),
    }
    if frame_indices is None:
        selection_data["removed_particles"] = indices
    else:
        selection_data["frame_selections"] = {
            str(index): values for index, values in sorted(frame_indices.items())
        }
    selection_path.write_text(json.dumps(selection_data, indent=2) + "\n")
    print(f"Preview updated: {comparison_path.resolve()}")

    output_video = args.output / f"{args.video.stem}_arm_removed.mp4"
    print(f"Decoding all frames to {output_video.resolve()} ...")
    if frame_indices is None:
        frames_written = decode_video_with_indices(
            model, config, args.video, indices, output_video, device)
    else:
        frames_written = decode_video_with_frame_indices(
            model, config, args.video, frame_indices, output_video, device)
    print(f"Wrote {frames_written} frames to {output_video.resolve()}")


def sam_test(args) -> None:
    """Prompt SAM 2 once, preview its LPWM choices, then propagate and decode."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise RuntimeError(
            "SAM 2 is not installed. Install Meta's `sam2` package in this environment "
            "or run this command from a SAM 2 environment that can also import LPWM."
        ) from error

    device = _device(args.device)
    sam_device = _device(args.sam_device)
    model, config, checkpoint = load_lpwm(args.model_dir, args.checkpoint, device)
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
    if args.time < 0:
        raise ValueError("--time must be non-negative")
    if args.mask_dilation < 0:
        raise ValueError("--mask-dilation must be non-negative")

    args.output.mkdir(parents=True, exist_ok=True)
    sample_path = args.output / "sam_particle_sample.png"
    output_video = args.output / f"{args.video.stem}_sam_arm_removed.mp4"
    selection_path = args.output / "sam_selection.json"
    print("Loading SAM 2 video predictor ...")
    predictor = build_sam2_video_predictor(
        args.sam_config, str(args.sam_checkpoint), device=str(sam_device))

    with tempfile.TemporaryDirectory(prefix="lpwm_sam_frames_") as temp_path:
        frame_dir = Path(temp_path)
        print("Extracting video frames for SAM 2 ...")
        fps, frame_count = extract_video_frames(args.video, frame_dir)
        prompt_frame = min(frame_count - 1, max(0, round(args.time * fps)))
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
                sample_path, args.display_size, args.mask_dilation)
            if sample is None:
                print("SAM prompt cancelled.")
                return
            sample_mask, prompt_points, prompt_labels = sample
            sample_indices = particle_indices_in_mask(
                positions, sample_mask, model.kp_range, args.mask_dilation)
            print(f"Saved sample preview to {sample_path.resolve()}")
            print(f"Sample selected LPWM particles: {sample_indices}")
            if args.sample_only:
                print("Sample-only mode: skipping video propagation and decoding.")
                return
            answer = input("Propagate this Franka mask and decode the full video? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Stopped after sample preview; no video was decoded.")
                return
            print("Propagating the SAM mask and decoding LPWM frames ...")
            frames_written, selections = decode_with_sam_masks(
                model, config, predictor, state, args.video, output_video, device,
                model.kp_range, prompt_frame, args.mask_dilation)

    selection_path.write_text(json.dumps({
        "video": str(args.video.resolve()),
        "model_checkpoint": str(checkpoint),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "sam_config": args.sam_config,
        "prompt_frame": prompt_frame,
        "prompt_points_xy": prompt_points.tolist(),
        "prompt_labels": prompt_labels.tolist(),
        "mask_dilation": args.mask_dilation,
        "frame_selections": {str(index): values for index, values in selections.items()},
    }, indent=2) + "\n")
    print(f"Wrote {frames_written} frames to {output_video.resolve()}")
    print(f"Saved SAM-derived particle labels to {selection_path.resolve()}")


def annotate(args) -> None:
    device = _device(args.device)
    model, config, checkpoint = load_lpwm(args.model_dir, args.checkpoint, device)
    videos = list(iter_videos(args.videos, args.recursive))
    if not videos:
        raise FileNotFoundError(f"No supported videos found in {args.videos}")
    args.output.mkdir(parents=True, exist_ok=True)
    sheet_dir = args.output / "sheets"
    sheet_dir.mkdir(exist_ok=True)
    dataset_path = args.output / "annotations.pt"
    records = torch.load(dataset_path, weights_only=False) if dataset_path.exists() else []
    completed = {(record["video"], record["time_sec"]) for record in records}

    print("Enter only Franka particle indices; all other particles are labeled 0. q saves and exits.")
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        for time_sec in np.arange(0, duration, args.interval):
            key = (str(video.resolve()), float(time_sec))
            if key in completed:
                continue
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            with torch.inference_mode():
                image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
                enc, dec = encode_decode(model, image)
            features = _without_time(enc["z_features"])[0]
            glimpses = dec["dec_objects_original_rgb"]
            sheet = sheet_dir / f"{video.stem}_{time_sec:010.3f}.png"
            sheet_paths = make_contact_sheet(frame, glimpses, sheet)
            overlay_paths = make_particle_overlays(
                frame, _without_time(enc["z"])[0],
                sheet.with_name(f"{sheet.stem}_overlay.png"), model.kp_range)
            print(f"\n{video.name} at {time_sec:.3f}s")
            print("Particle overlays on source frame:")
            for path in overlay_paths:
                print(f"  {path}")
            print("Particle crop sheets:")
            for path in sheet_paths:
                print(f"  {path}")
            while True:
                answer = input("Franka particle indices> ").strip().lower()
                if answer in {"q", "quit"}:
                    capture.release()
                    print(f"Saved {len(records)} annotated frames to {dataset_path}")
                    return
                try:
                    franka_indices = parse_particle_indices(answer, len(features))
                    labels = [int(index in franka_indices) for index in range(len(features))]
                    break
                except ValueError as error:
                    print(error)
            records.append({
                "video": key[0], "time_sec": key[1],
                "features": features.detach().cpu(),
                "labels": torch.tensor(labels, dtype=torch.float32),
                "positions": _without_time(enc["z"])[0].detach().cpu(),
            })
            torch.save(records, dataset_path)
        capture.release()
    metadata = {"model_checkpoint": str(checkpoint), "frames": len(records), "interval": args.interval}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(records)} annotated frames to {dataset_path}")


def train(args) -> None:
    device = _device(args.device)
    records = torch.load(args.annotations, map_location="cpu", weights_only=False)
    if len(records) < 2:
        raise ValueError("At least two annotated frames are required")
    random.Random(args.seed).shuffle(records)
    split = min(max(1, round(len(records) * (1 - args.valid_fraction))), len(records) - 1)
    train_records, valid_records = records[:split], records[split:]

    def tensors(items):
        return (torch.cat([item["features"] for item in items]).float(),
                torch.cat([item["labels"] for item in items]).float())

    x_train, y_train = tensors(train_records)
    x_valid, y_valid = tensors(valid_records)
    if y_train.sum() == 0:
        raise ValueError("Training split contains no positive (arm) particles")
    classifier = ArmParticleClassifier(x_train.shape[-1], args.hidden_dim, args.hidden_layers).to(device)
    pos_weight = ((len(y_train) - y_train.sum()) / y_train.sum()).clamp_min(1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    best_state, best_loss = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        for features, labels in train_loader:
            loss = criterion(classifier(features.to(device)), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        classifier.eval()
        with torch.inference_mode():
            logits = classifier(x_valid.to(device))
            valid_loss = criterion(logits, y_valid.to(device)).item()
            accuracy = ((logits.sigmoid() >= .5).cpu() == y_valid.bool()).float().mean().item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu() for key, value in classifier.state_dict().items()}
        print(f"epoch {epoch:03d}: valid_loss={valid_loss:.4f}, valid_accuracy={accuracy:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feature_dim": x_train.shape[-1],
                "hidden_dim": args.hidden_dim, "hidden_layers": args.hidden_layers,
                "threshold": args.threshold}, args.output)
    print(f"Saved classifier to {args.output}")


def load_classifier(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ArmParticleClassifier(checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["hidden_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint.get("threshold", 0.5)


def decode(args) -> None:
    device = _device(args.device)
    model, config, _ = load_lpwm(args.model_dir, args.checkpoint, device)
    classifier, saved_threshold = load_classifier(args.classifier, device)
    threshold = saved_threshold if args.threshold is None else args.threshold
    videos = list(iter_videos(args.videos, args.recursive))
    args.output.mkdir(parents=True, exist_ok=True)
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        output_path = args.output / f"{video.stem}_arm_removed.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (config["image_size"], config["image_size"]))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video: {output_path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = preprocess_frame(frame, config["image_size"], device, model.normalize_rgb)
            with torch.inference_mode():
                enc = model.encode_all(image, deterministic=True)
                arm = classifier(enc["z_features"]).sigmoid() >= threshold
                keep = enc["obj_on"].clone()
                keep[arm.unsqueeze(-1)] = 0
                dec = model.decode_all(enc["z"], enc["z_scale"], enc["z_features"], keep,
                                       enc["z_depth"], enc["z_bg_features"], None)
            rgb = decoded_rgb(dec)
            writer.write(cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        capture.release()
        writer.release()
        print(f"Wrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def model_args(subparser):
        subparser.add_argument("--model-dir", type=Path, required=True, help="Directory containing hparams.json")
        subparser.add_argument("--checkpoint", type=Path)
        subparser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")

    annotate_parser = subparsers.add_parser("annotate", help="Sample frames and label particle glimpses")
    model_args(annotate_parser)
    annotate_parser.add_argument("--videos", type=Path, required=True)
    annotate_parser.add_argument("--output", type=Path, required=True)
    annotate_parser.add_argument("--interval", type=float, default=1.0)
    annotate_parser.add_argument("--recursive", action="store_true")
    annotate_parser.set_defaults(func=annotate)

    manual_parser = subparsers.add_parser(
        "manual-test", help="Select particle slots in an interactive full-video popup")
    model_args(manual_parser)
    manual_parser.add_argument("--video", type=Path, required=True)
    manual_parser.add_argument("--time", type=float, default=0.0,
                               help="Initial popup timestamp in seconds")
    manual_parser.add_argument("--output", type=Path, required=True)
    manual_parser.add_argument(
        "--display-size", type=int, default=1000,
        help="Popup size along its longest image edge (default: 1000)")
    manual_parser.add_argument("--remove", type=int, nargs="*",
                               help="Non-interactive particle indices; omit for the video popup")
    manual_parser.set_defaults(func=manual_test)

    sam_parser = subparsers.add_parser(
        "sam-test", help="Prompt SAM 2 on a sample, track the arm, and decode selected particles")
    model_args(sam_parser)
    sam_parser.add_argument("--video", type=Path, required=True)
    sam_parser.add_argument("--output", type=Path, required=True)
    sam_parser.add_argument("--sam-checkpoint", type=Path, required=True)
    sam_parser.add_argument(
        "--sam-config", required=True,
        help="SAM 2 config name, e.g. configs/sam2.1/sam2.1_hiera_s.yaml")
    sam_parser.add_argument("--sam-device", default="auto", help="SAM device (default: auto)")
    sam_parser.add_argument("--time", type=float, default=0.0,
                            help="Timestamp for the SAM prompt sample")
    sam_parser.add_argument("--display-size", type=int, default=1000)
    sam_parser.add_argument(
        "--mask-dilation", type=int, default=0,
        help="Expand the SAM mask by this many pixels before selecting particle centers")
    sam_parser.add_argument(
        "--sample-only", action="store_true",
        help="Save the prompted sample preview without propagating or decoding")
    sam_parser.set_defaults(func=sam_test)

    train_parser = subparsers.add_parser("train", help="Train the binary per-particle MLP")
    train_parser.add_argument("--annotations", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--hidden-layers", type=int, default=3)
    train_parser.add_argument("--valid-fraction", type=float, default=0.2)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(func=train)

    decode_parser = subparsers.add_parser("decode", help="Decode videos with arm particles disabled")
    model_args(decode_parser)
    decode_parser.add_argument("--classifier", type=Path, required=True)
    decode_parser.add_argument("--videos", type=Path, required=True)
    decode_parser.add_argument("--output", type=Path, required=True)
    decode_parser.add_argument("--threshold", type=float)
    decode_parser.add_argument("--recursive", action="store_true")
    decode_parser.set_defaults(func=decode)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
