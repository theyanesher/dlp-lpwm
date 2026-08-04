"""Convert LIBERO demonstration videos into leakage-free LPWM episodes.

The train/validation split is made by task name before videos are chunked.  As
a result, camera views of the same task remain in the same split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VIEW_SUFFIX = re.compile(r"_(agentview|frontview|galleryview|paperview)$", re.IGNORECASE)
LEADING_ID = re.compile(r"^\d+_")


def task_name(path: Path) -> str:
    """Return a view-independent task identifier from a demo filename."""
    stem = VIEW_SUFFIX.sub("", path.stem)
    return LEADING_ID.sub("", stem)


def stable_score(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


def split_tasks(videos: list[Path], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    tasks = sorted({task_name(video) for video in videos}, key=lambda name: stable_score(name, seed))
    if len(tasks) < 2:
        raise ValueError("At least two distinct LIBERO tasks are required for a train/val split")
    val_count = min(len(tasks) - 1, max(1, round(len(tasks) * val_fraction)))
    return set(tasks[val_count:]), set(tasks[:val_count])


def read_frames(video: Path, image_size: int, frame_step: int):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_step == 0:
            yield cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        index += 1
    capture.release()


def prepare(args) -> None:
    videos = sorted(
        path for path in (args.input.rglob("*") if args.recursive else args.input.glob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise FileNotFoundError(f"No supported videos found in {args.input}")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.episode_length < 2 or args.frame_step < 1:
        raise ValueError("--episode-length must be >= 2 and --frame-step must be >= 1")
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty: {args.output}; pass --overwrite to replace it")
        shutil.rmtree(args.output)

    train_tasks, val_tasks = split_tasks(videos, args.val_fraction, args.seed)
    manifest = {
        "format_version": 1,
        "image_size": args.image_size,
        "episode_length": args.episode_length,
        "frame_step": args.frame_step,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "split_unit": "task",
        "train_tasks": sorted(train_tasks),
        "val_tasks": sorted(val_tasks),
        "episodes": [],
    }
    counters = {"train": 0, "val": 0}
    for video in videos:
        task = task_name(video)
        split = "val" if task in val_tasks else "train"
        frames = list(read_frames(video, args.image_size, args.frame_step))
        chunks = [frames[start:start + args.episode_length]
                  for start in range(0, len(frames), args.episode_length)]
        if chunks and len(chunks[-1]) < args.episode_length:
            if args.pad_last and chunks[-1]:
                chunks[-1] += [chunks[-1][-1]] * (args.episode_length - len(chunks[-1]))
            else:
                chunks.pop()
        for chunk in chunks:
            episode_id = f"{counters[split]:06d}"
            counters[split] += 1
            episode_dir = args.output / split / episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, frame in enumerate(chunk):
                path = episode_dir / f"{frame_index:06d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not write frame: {path}")
            manifest["episodes"].append({
                "split": split,
                "episode": episode_id,
                "source": str(video.resolve()),
                "task": task,
                "frames": len(chunk),
            })
    for split, count in counters.items():
        if count == 0:
            raise RuntimeError(f"Preprocessing produced no {split} episodes")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {counters['train']} train and {counters['val']} val episodes in {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder containing LIBERO videos")
    parser.add_argument("--output", type=Path, required=True, help="Destination dataset root")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--frame-step", type=int, default=1, help="Keep every Nth source frame")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad-last", action="store_true", help="Pad incomplete final chunks with their last frame")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
