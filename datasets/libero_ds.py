"""Video-only LIBERO datasets for LPWM and DLP training.

Expected layout (created by ``datasets/libero_preparation.py``)::

    root/train/000000/000000.png
    root/train/000000/000001.png
    root/val/000000/000000.png

Every episode directory contains the same number of frames.  No actions,
language, goals, or robot state are loaded.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _split_name(mode: str) -> str:
    if mode in {"valid", "validation"}:
        return "val"
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported LIBERO split: {mode}")
    return mode


def _episode_frames(root: str | Path, mode: str) -> list[list[Path]]:
    split_dir = Path(root) / _split_name(mode)
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"LIBERO split not found: {split_dir}. Run datasets/libero_preparation.py first."
        )
    episodes = []
    for episode_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        frames = sorted(episode_dir.glob("*.png"))
        if frames:
            episodes.append(frames)
    if not episodes:
        raise RuntimeError(f"No PNG episodes found under {split_dir}")
    return episodes


class LiberoVideoDataset(Dataset):
    """Fixed-window, unconditioned LIBERO video dataset."""

    def __init__(self, root, mode, sample_length=17, image_size=128):
        self.mode = _split_name(mode)
        self.sample_length = sample_length
        self.episodes = _episode_frames(root, self.mode)
        too_short = [len(frames) for frames in self.episodes if len(frames) < sample_length]
        if too_short:
            raise ValueError(
                f"{len(too_short)} LIBERO episodes are shorter than sample_length={sample_length}; "
                "rerun preprocessing with a large enough --episode-length"
            )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.windows = [
            (episode, offset)
            for episode, frames in enumerate(self.episodes)
            for offset in range(len(frames) - sample_length + 1)
        ] if self.mode == "train" else [(episode, 0) for episode in range(len(self.episodes))]

    def __getitem__(self, index):
        episode, offset = self.windows[index]
        paths = self.episodes[episode]
        if self.mode == "train":
            paths = paths[offset:offset + self.sample_length]
        images = torch.stack([
            self.transform(Image.open(path).convert("RGB")) for path in paths
        ]).float()
        # LPWM's common training/evaluation code expects a five-item tuple.  Only
        # the frames and valid-timestep mask are meaningful for this dataset.
        empty = torch.zeros(0)
        valid = torch.ones(images.shape[0], dtype=torch.int64)
        return images, empty, empty, empty, valid

    def __len__(self):
        return len(self.windows)


class LiberoImageDataset(Dataset):
    """Frame view used by the repository's image-model utilities."""

    def __init__(self, root, mode, sample_length=1, image_size=128):
        episodes = _episode_frames(root, mode)
        self.frames = [path for episode in episodes for path in episode]
        self.sample_length = sample_length
        if sample_length < 1:
            raise ValueError("sample_length must be positive")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        paths = self.frames[index:index + self.sample_length]
        if len(paths) < self.sample_length:
            paths += [paths[-1]] * (self.sample_length - len(paths))
        images = torch.stack([
            self.transform(Image.open(path).convert("RGB")) for path in paths
        ]).float()
        if self.sample_length == 1:
            images = images[0]
        empty = torch.zeros(0)
        return images, empty, empty, empty, empty

    def __len__(self):
        return len(self.frames)
