# Training LPWM on LIBERO demonstration videos

LIBERO support is video-only: actions, language, task IDs, image goals, and robot
state are deliberately not supplied to LPWM. The model learns its latent actions
directly from frame sequences.

## Prepare the dataset

From the repository root:

```bash
python datasets/libero_preparation.py \
  --input /home/theyanesh/LIBERO/demo_videos \
  --output ./data/libero_lpwm \
  --episode-length 30 \
  --image-size 128 \
  --val-fraction 0.2 \
  --seed 0 \
  --pad-last
```

The split is deterministic and is performed by task before episode creation.
Thus, agent, front, gallery, and paper views of one task cannot be divided across
train and validation. Each source video is decoded, resized, and divided into
30-frame episodes. `--pad-last` retains incomplete final chunks by repeating the
last frame; omit it to discard those chunks. Use `--frame-step 2`, for example,
to keep every second input frame.

The command refuses to write into a nonempty destination. Pass `--overwrite`
only when replacing a previously processed dataset is intended.

## Data format

```text
data/libero_lpwm/
├── manifest.json
├── train/
│   ├── 000000/
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── 000029.png
│   └── ...
└── val/
    ├── 000000/
    │   ├── 000000.png
    │   ├── ...
    │   └── 000029.png
    └── ...
```

`manifest.json` records preprocessing parameters, task membership, source paths,
and every generated episode. The training loader creates all consecutive
`timestep_horizon + 1` windows inside each training episode. Validation returns
complete fixed-length episodes for rollout evaluation.

## Train

The default dataset location is `./data/libero_lpwm`. Change `root` in
`configs/libero.json` if another output location was used.

```bash
python train_lpwm.py --dataset libero
```

For multiple GPUs:

```bash
accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py --dataset libero
```

The configuration explicitly disables every external conditioning type and uses
one camera view per video. It defaults to `num_workers: 0` for compatibility
with restricted containers; increase it on a normal host if data loading is a
bottleneck. The two-GPU default uses one sequence per GPU and accumulates four
microbatches, producing an effective global batch size of eight without holding
four sequences in GPU memory simultaneously. Reduce `batch_size` if GPU memory
is insufficient; adjust `gradient_accumulation_steps` to change the effective
batch independently.

W&B logging is enabled for the LIBERO configuration. Authenticate once with
`wandb login`, or provide `WANDB_API_KEY` in the environment. Only the main
process logs: training losses are sent every `wandb_log_interval` optimizer
updates, while validation metrics and the existing reconstruction/particle grid
are sent once per evaluation epoch. Set `wandb_enabled` to `false` for an
entirely local run.

## Validation checks

Before starting a long run, verify the prepared data:

```bash
python - <<'PY'
from datasets.get_dataset import get_video_dataset

train = get_video_dataset("libero", "./data/libero_lpwm", seq_len=17, mode="train", image_size=128)
valid = get_video_dataset("libero", "./data/libero_lpwm", seq_len=17, mode="valid", image_size=128)
print("windows:", len(train), "validation episodes:", len(valid))
print("train tensor:", train[0][0].shape)
print("valid tensor:", valid[0][0].shape)
PY
```

Expected frame shapes are `[17, 3, 128, 128]` for a training window and
`[30, 3, 128, 128]` for a validation episode.
