# Single-Image Reflection Removal Playground

PyTorch implementation of a single-image reflection-removal research playground centered on an
older conditioned diffusion prototype plus newer benchmarking/reference utilities.

## Status

This repository is best understood as a **research playground**.

It now contains two layers:

1. the original conditioned diffusion prototype,
2. lightweight benchmark/reference utilities for stronger newer methods.

The repository slug remains `diffusion-reflection-removal` for continuity, but the contents are
better described as a broader single-image reflection-removal workspace rather than just one model.

## Sample Output

| Input | Final output |
|---|---|
| ![input](output/input.jpg) | ![final](output/final_result.jpg) |

The `output/` directory also contains intermediate denoising snapshots produced during inference.

## Original prototype summary

The implementation combines:

1. **UNet-style backbone**
   - hidden dimension starts at 96
   - progressive channel expansion: `96 → 192 → 384 → 768`
   - encoder / decoder skip connections

2. **Conditioned diffusion process**
   - 1000 diffusion timesteps
   - linear beta schedule
   - image-conditioned denoising

3. **Progressive reconstruction outputs**
   - intermediate samples can be written during inference to visualize denoising behavior

## Repository Layout

```text
.
├── config.py
├── train.py
├── inference.py
├── models/
│   ├── diffusion.py
│   └── swin_transformer.py
├── utils/
│   ├── dataset.py
│   └── training.py
└── output/
    └── sample inference artifacts
```

## Setup

```bash
conda create -n reflection python=3.8
conda activate reflection
pip install -r requirements.txt
```

## Training

```bash
python train.py --batch-size 4 --epochs 100 --lr 2e-4
```

Training details live in `train.py` and use paired reflection / reflection-free supervision.

## Inference

```bash
python inference.py \
  --input path/to/input/image.jpg \
  --output_dir ./outputs/sample1 \
  --checkpoint path/to/checkpoint.pth \
  --save_interval 50
```

### Main arguments

- `--input` — input image with reflections
- `--output_dir` — directory for final and intermediate outputs
- `--checkpoint` — trained checkpoint path
- `--save_interval` — save intermediate denoising results every *N* steps

## Inference outputs

1. input image
2. initial noise image
3. intermediate denoising snapshots
4. final reflection-removed image

## Notes and limitations

- CPU and GPU inference are both supported, but intended usage is GPU-oriented.
- No pretrained checkpoint is bundled in this repository.
- CUDA / PyTorch compatibility should be checked manually for GPU runs.

## 2026 benchmark + reference update

This repository now also includes a lightweight **latest-method benchmarking/reference layer**
for comparing this prototype against stronger modern baselines:

- **RDNet / XReflection** (CVPR 2025 ecosystem baseline)
- **Dereflection Any Image** (modern diffusion-based reference)

Added materials:

- `docs/latest-methods.md` — concise notes on stronger current methods worth using as references
- `docs/benchmarks/2026-04-17-benchmark-report.md` — public-set benchmark summary from a `develop` host run
- `docs/benchmarks/2026-04-17-*-summary.json` — per-dataset metric dumps
- `scripts/run_dai_eval.py` — unified external evaluator for Dereflection Any Image outputs
- `scripts/collect_rdnet_results.py` — converts XReflection visualization outputs into exportable predictions + metrics
- `scripts/make_method_comparison.py` — builds side-by-side comparison boards
- `configs/rdnet_eval.yml` — test-only RDNet evaluation config for XReflection checkpoints

These additions **do not replace** the original prototype model in this repo. They are here to
make future work easier to benchmark against stronger references before investing more time in
the old diffusion prototype.
## License

Apache License 2.0. See [`LICENSE`](LICENSE).
