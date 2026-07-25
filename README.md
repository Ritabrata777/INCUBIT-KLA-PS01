# NAFNet-SR: Semiconductor Image Restoration

A self-training PyTorch pipeline for restoring degraded semiconductor inspection images — denoising, deblurring, and 4× super-resolution — using **NAFNet-SR** (Nonlinear Activation Free Network for Super-Resolution). Built for the **KLA Hackathon (INCUBIT-KLA-PS01)**.

Instead of relying on fixed pre-paired training data, the pipeline **synthesizes degradations on-the-fly** from clean images at every epoch. Each sample gets freshly randomized noise, blur, and downsampling parameters, so the model never sees the same degraded image twice — giving it better generalization to unknown test distributions.

---

## Results

Trained on 4,241 real SEM images ([Carinthia dataset](https://zenodo.org/records/10715190)) and evaluated on 150 held-out images with synthetic degradation:

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic upsample (baseline) | 30.57 | 0.746 |
| **NAFNet-SR (this repo)** | **34.30** | **0.933** |

**+3.7 dB PSNR and +0.187 SSIM** over the naive baseline.

![Degraded input → NAFNet-SR output → ground truth](comparison_demo.png)
*Left: degraded input · Middle: NAFNet-SR restored · Right: ground truth*

A pre-trained checkpoint (`final_model_weights.pt`) is included — you can run evaluation right away without training.

---

## Project files

| File | Purpose |
|---|---|
| `model.py` | `NAFNet_SR` architecture |
| `degradation.py` | On-the-fly synthetic degradation pipeline |
| `dataset.py` | Dataset loader (self-training and static-pair modes) |
| `train.py` | Training loop (AdamW + CosineAnnealingLR + Charbonnier loss, AMP) |
| `evaluation.py` | Batched FP16 GPU inference CLI |
| `compute_metrics.py` | PSNR/SSIM scoring against ground truth |
| `prepare_carinthia.py` | Splits the Carinthia dataset into train/val/holdout |
| `make_demo_testset.py` | Builds a synthetic degraded test set from held-out images |
| `test_single_image.py` | Quick single-image test with before/after comparison |
| `run_pipeline.sh` / `.bat` | One-command train + evaluate (Linux/Windows) |
| `requirements.txt` | Python dependencies |

---

## Setup

### 1. Install PyTorch with CUDA

Plain `pip install torch` gives a CPU-only build. For GPU support (CUDA 12.x, including H100s and most modern consumer GPUs):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for other CUDA versions.

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify GPU is detected

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True` and your GPU name. If it prints `False`, re-check step 1.

---

## Data setup

### Option A — Use the Carinthia SEM dataset (recommended)

Download and split it in three commands:

```bash
# 1. Download (~128 MB)
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://zenodo.org/records/10715190/files/data.zip?download=1',
    'carinthia_data.zip'
)
print('Downloaded.')
"

# 2. Extract
python -c "import zipfile; zipfile.ZipFile('carinthia_data.zip').extractall('carinthia_raw')"

# 3. Split into train / val / holdout
python prepare_carinthia.py --src carinthia_raw/data/images --dst data --val_count 200 --holdout_count 150
```

This creates:

```
data/
├── clean/           # 4,241 training images
├── val_clean/       # 200 validation images
└── holdout_clean/   # 150 held-out images (used to build the test set below)
```

### Option B — Bring your own images

Put clean images in `data/clean/` and `data/val_clean/`. Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`. Images are read as grayscale.

If you have real pre-paired degraded/clean images, skip the `--self_train` flag in training (see §Training below) and point `--degraded_dir` / `--val_degraded_dir` at your LR inputs.

---

## Running the full demo pipeline

After completing data setup, these four commands reproduce the results above end-to-end:

```bash
# Build a synthetic degraded test set from the 150 held-out clean images
python make_demo_testset.py --clean_dir data/holdout_clean \
    --input_dir data/test_input --gt_dir data/test_gt --upscale_factor 4

# Restore with the included pre-trained weights
python evaluation.py --input_dir data/test_input --output_dir data/test_output \
    --weights_path final_model_weights.pt --batch_size 32 \
    --upscale_factor 4 --width 32 --num_blocks 8

# Score the results
python compute_metrics.py --restored_dir data/test_output \
    --gt_dir data/test_gt --degraded_dir data/test_input
```

Expected output:
```
Method                      PSNR (dB)     SSIM
Bicubic upsample (baseline) 30.572        0.7457
NAFNet-SR (ours)            34.303        0.9326
```

---

## Training

### Self-training mode (recommended)

Generates fresh degradations every epoch — no pre-made degraded images needed:

```bash
python train.py --self_train \
    --clean_dir data/clean \
    --val_clean_dir data/val_clean \
    --epochs 100 \
    --batch_size 16 \
    --patch_size 128 \
    --upscale_factor 4 \
    --width 32 \
    --num_blocks 8 \
    --output_path final_model_weights.pt
```

### Static pre-paired mode

If you have real degraded/clean image pairs:

```bash
python train.py \
    --clean_dir data/clean --degraded_dir data/degraded \
    --val_clean_dir data/val_clean --val_degraded_dir data/val_degraded \
    --epochs 100 --upscale_factor 4
```

### Key training flags

| Flag | Default | Description |
|---|---|---|
| `--self_train` | off | Enable on-the-fly synthetic degradation |
| `--upscale_factor` | `4` | SR scale. Must match at evaluation time |
| `--patch_size` | `128` | HR crop size per training sample |
| `--width` | `32` | Base channel width of the NAFNet body |
| `--num_blocks` | `8` | Number of stacked NAFBlocks |
| `--batch_size` | `16` | Lower if you hit CUDA OOM |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `2e-4` | AdamW learning rate (cosine-annealed to `1e-6`) |
| `--amp` / `--no_amp` | on | Toggle mixed-precision training |
| `--output_path` | `final_model_weights.pt` | Where best weights are saved |

On a small GPU (4–8 GB), start with `--width 16 --num_blocks 4 --batch_size 4` to sanity-check, then scale up.

---

## One-command pipeline

**Linux / Git Bash:**
```bash
./run_pipeline.sh

# Overrides:
EPOCHS=50 BATCH_SIZE=32 ./run_pipeline.sh

# Evaluation only (skip training):
SKIP_TRAIN=1 WEIGHTS_PATH=final_model_weights.pt ./run_pipeline.sh
```

**Windows CMD:**
```bat
set EPOCHS=50 & set BATCH_SIZE=32 & run_pipeline.bat

rem Evaluation only:
set SKIP_TRAIN=1 & run_pipeline.bat
```

All overridable variables: `CLEAN_DIR`, `VAL_CLEAN_DIR`, `TEST_INPUT_DIR`, `OUTPUT_DIR`, `WEIGHTS_PATH`, `EPOCHS`, `BATCH_SIZE`, `PATCH_SIZE`, `UPSCALE_FACTOR`, `WIDTH`, `NUM_BLOCKS`, `NUM_WORKERS`, `EVAL_BATCH_SIZE`, `SKIP_TRAIN`.

---

## Quick single-image test

No folder setup needed — just point at any image:

```bash
# Restore a degraded image directly
python test_single_image.py --image path/to/photo.jpg

# Degrade a clean image first, then restore (true before/after)
python test_single_image.py --image path/to/photo.jpg --degrade_first
```

Results are saved to `single_test_output/`: `restored.png`, and a `comparison.png` side-by-side panel. With `--degrade_first` you also get `degraded.png` and a 3-way comparison.

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input image path |
| `--weights_path` | `final_model_weights.pt` | Weights to use |
| `--output_dir` | `single_test_output` | Where results are saved |
| `--upscale_factor`, `--width`, `--num_blocks` | `4`, `32`, `8` | Must match training values |
| `--degrade_first` | off | Synthetically degrade before restoring |

---

## Architecture overview

```
Input (LR grayscale)
  │
  ├─ intro conv (3×3)
  ├─ NAFBlock × N  ──────────────────────────────────────────────────────┐
  │    ├─ LayerNorm → 1×1 expand → depthwise 3×3                        │
  │    │     → SimpleGate → Simplified Channel Attention → 1×1 project  │
  │    │     → residual add                                              │
  │    └─ LayerNorm → 1×1 expand → SimpleGate → 1×1 project             │
  │          → residual add                                              │
  ├─ tail conv (3×3) + long residual ◄────────────────────────────────────┘
  ├─ Sub-pixel conv (3×3) → PixelShuffle(upscale_factor)
  ├─ final conv (3×3)
  └─ + bicubic-upsampled input (long skip connection)
Output (HR grayscale, restored)
```

- **SimpleGate**: splits channels in half and multiplies — no ReLU/GELU anywhere.
- **Simplified Channel Attention**: global average pool + single 1×1 conv; much cheaper than SE-style attention.
- **PixelShuffle head**: sub-pixel upsampling — cleaner than transposed convolutions, no checkerboard artifacts.

---

## Degradation pipeline

`SemiconductorDegradationPipeline.apply_degradation()` randomly combines four operations per sample:

1. **Gaussian blur** — random odd kernel size and sigma (optical defocus / soft edges)
2. **Speckle noise** — multiplicative `img + img × N(0, σ²)`, left unclamped to match real sensor behavior
3. **Gaussian sensor noise** — additive, also unclamped
4. **Resolution downsampling** — random `nearest` or `bicubic` interpolation to `1/upscale_factor` resolution

Parameters are resampled on every call, so the same clean image produces a different degraded version every time — acting as an unlimited augmentation source.

---

## Hardware notes

- `evaluation.py` uses `torch.amp.autocast('cuda')` (FP16) + `torch.no_grad()` with GPU batching. Set `--batch_size 32` or higher on an H100 for best throughput.
- `train.py` also supports AMP (`--amp`, on by default).
- Measured throughput on RTX 3050 Laptop (4 GB): **~1,680 images/sec** at `batch_size=32`, 32×32→128×128. An H100 will substantially exceed this.
- On small GPUs (4–8 GB VRAM), reduce `--batch_size`, `--patch_size`, `--width`, and `--num_blocks` to avoid OOM.

---

## About the included checkpoint

`final_model_weights.pt` was trained with:

```bash
python train.py --self_train \
    --clean_dir data/clean --val_clean_dir data/val_clean \
    --epochs 40 --batch_size 16 --patch_size 128 --upscale_factor 4 \
    --width 32 --num_blocks 8 --num_workers 4 \
    --output_path final_model_weights.pt
```

Trained on a single RTX 3050 Laptop GPU (4 GB), ~20–30 s/epoch, ~15 minutes total. Best validation Charbonnier loss (`0.007814`) at epoch 32.

Two caveats worth knowing:
- This is a solid **starting point**, not a submission-tuned model. If the hackathon provides its own dataset, retrain or fine-tune on that data for best results on the judges' test distribution.
- The Carinthia dataset skews heavily toward defect class 3 (~87% of images). For a more class-balanced model, consider stratified sampling in `prepare_carinthia.py`.

---

## Sanity check

Before a full training run, verify the end-to-end pipeline works with minimal settings:

```bash
python train.py --self_train --clean_dir data/clean --val_clean_dir data/val_clean \
    --epochs 2 --batch_size 4 --patch_size 64 --upscale_factor 2 \
    --width 16 --num_blocks 4 --output_path test_weights.pt

python evaluation.py --input_dir data/test_input --output_dir data/test_output \
    --weights_path test_weights.pt --batch_size 4 --upscale_factor 2 \
    --width 16 --num_blocks 4
```

If both complete and `data/test_output/` has correctly-sized images, you're good to scale up.

---

## License & acknowledgments

- **Code**: written for the KLA Hackathon (INCUBIT-KLA-PS01 submission).
- **Training data**: [Carinthia dataset](https://zenodo.org/records/10715190), DOI [10.5281/zenodo.10715190](https://doi.org/10.5281/zenodo.10715190), released under **CC BY 4.0** by Kofler, C.; Strauß, S.; Zernig, A. (KAI GmbH); Lazaro Garcia, E. (Infineon Technologies Dresden); Boxleitner, M.; Mayr, B.; Dicillia-Kovatsch, I.; Dohr, C.A. (Infineon Technologies Austria) — funded by the European Commission AIMS5.0 project (grant 101112089). Attribution is required for any reuse of the dataset; see the Zenodo page for full terms.
