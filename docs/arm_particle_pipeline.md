# LPWM robot-arm particle pipeline

This tool samples one frame per second from a folder of robot-demo videos, shows indexed LPWM particle positions on each source frame and particle glimpses, trains a binary MLP on the visual particle latents, and reconstructs each video after disabling particles classified as robot arm.

Run commands from the `lpwm` directory in the LPWM environment.

## 1. Manually select particle slots and decode the video

Before annotating a dataset, inspect a complete video and manually remove particles:

```bash
python particle_arm_pipeline.py manual-test \
  --model-dir checkpoints/my_panda_run \
  --checkpoint /path/to/model.pth \
  --video /path/to/robot_demo.mp4 \
  --time 3.0 \
  --output manual_test
```

The command opens an interactive video popup with indexed LPWM particles. Selections are independent on every frame, so click whichever points cover the arm on the current frame even when particle IDs change. Selected particles are red. Press Space to play or pause, use A/D or the arrow keys to move one frame at a time, C to explicitly clear the current frame, and Home/End to jump to the ends. Navigating back restores that frame's choices. Press Enter to save and decode, or Q/Escape to cancel.

The status and controls are displayed in a separate header above the video, so they do not cover the image. The longest image edge is enlarged to 1000 pixels by default; use, for example, `--display-size 1400` for a larger view.

The tool writes `manual_test/manual_comparison.png` for the frame visible when you press Enter. It then decodes the whole source video, suppressing each frame's selected slots, and writes `manual_test/<video-name>_arm_removed.mp4`. Frames you did not label receive a normal LPWM reconstruction with no particles suppressed. The choices are recorded under `frame_selections` in `manual_test/selection.json`. Static overlay and glimpse sheets are still produced as references.

For a fully non-interactive or headless run, add a list such as `--remove 1 4 7`.

## 2. SAM-assisted manual sample and decode

SAM 2 can replace frame-by-frame clicking. This repository includes a pinned optional dependency file. Install it into the LPWM environment and download the official Hiera-S checkpoint from Hugging Face:

```bash
uv pip install --python .venv/bin/python -r requirements-sam.txt

.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="facebook/sam2.1-hiera-small",
    filename="sam2.1_hiera_small.pt",
    local_dir="checkpoints/sam2",
)
PY
```

The checked-in LPWM environment is compatible with this version. If using another environment, current SAM 2 requires Python 3.10 or newer and a recent PyTorch/torchvision pair.

```bash
python particle_arm_pipeline.py sam-test \
  --model-dir checkpoints/my_panda_run \
  --checkpoint /path/to/lpwm.pth \
  --video /path/to/original_robot_demo.mp4 \
  --sam-checkpoint checkpoints/sam2/sam2.1_hiera_small.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --output sam_manual_test \
  --display-size 1400
```

On the sample popup, left-click the Franka arm or gripper to add foreground prompts and right-click nearby objects or the background to exclude them. Press R to reset the prompts. The green region is SAM's mask; LPWM points inside it turn red. Press Enter only when the mask covers the complete manipulator without including manipulated objects.

The accepted preview is saved as `sam_particle_sample.png`. The terminal then asks before running full-video propagation. Answer `y` to track the mask and create `<video-name>_sam_arm_removed.mp4`, or answer `n` to stop after inspecting the sample. Use `--sample-only` to always stop after the preview. SAM-derived per-frame particle selections are saved in `sam_selection.json`.

By default, a particle is selected only when its center falls inside the mask. `--mask-dilation 2` expands the mask by two source-video pixels when points lie just outside its edge.

## 3. Annotate particles

```bash
python particle_arm_pipeline.py annotate \
  --model-dir checkpoints/my_panda_run \
  --videos /path/to/robot_demo_videos \
  --output arm_annotations \
  --interval 1 \
  --recursive
```

Pass `--checkpoint /path/to/model.pth` when the checkpoint is not in the model directory's standard `saves/` location. For every sampled frame, inspect the indexed source-frame overlays and enter only the particle indices positioned on the robot arm. All unlisted particles are labeled `0` automatically. Type `q` to stop safely. Progress is saved after every frame in `arm_annotations/annotations.pt`, and rerunning the same command resumes it.

## 4. Train the classifier

```bash
python particle_arm_pipeline.py train \
  --annotations arm_annotations/annotations.pt \
  --output arm_particle_classifier.pth \
  --epochs 20
```

The split is made by annotated frame, rather than by particle, to reduce train/validation leakage. The classifier uses weighted binary cross entropy to account for the usually rare arm particles.

## 5. Decode arm-free videos

```bash
python particle_arm_pipeline.py decode \
  --model-dir checkpoints/my_panda_run \
  --classifier arm_particle_classifier.pth \
  --videos /path/to/robot_demo_videos \
  --output decoded_without_arm \
  --recursive
```

Every source frame is resized to the LPWM training resolution and encoded. Particles whose arm probability is at least `0.5` have their `obj_on` value set to zero before LPWM decoding, so the learned background and remaining particles form the output. Use `--threshold` during decoding to override the saved threshold; a lower value removes particles more aggressively.

The output is an LPWM reconstruction, not pixel-level inpainting of the original frame. Its clarity therefore depends on the LPWM checkpoint's reconstruction and background quality.
