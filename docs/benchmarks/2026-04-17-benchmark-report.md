# 2026-04-17 reflection-removal benchmark report

## Setup

- Host: `develop` (`super`)
- GPU: RTX 3090 x4
- Public datasets:
  - `Real20` (20 images)
  - `Nature` (20 images)
  - `SIR2-Wild` (55 images)
- Total evaluated images: **95**

## Compared methods

### RDNet / XReflection

- Upstream checkpoint: `rdnet-26.4849.ckpt`
- Evaluated through XReflection in `test_only` mode
- Predictions exported from XReflection visualization outputs and rescored externally

### Dereflection Any Image

- Upstream HF model: `sjtu-deepvision/dereflection-any-image-v0`
- Evaluated directly with a unified external evaluator

## Unified metric summary

| Dataset | Count | RDNet PSNR | RDNet SSIM | DAI PSNR | DAI SSIM | Δ PSNR (DAI-RDNet) | Δ SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real20 | 20 | 24.1751 | 0.8189 | 25.2353 | 0.8367 | +1.0601 | +0.0178 |
| Nature | 20 | 26.3320 | 0.8367 | 27.0547 | 0.8421 | +0.7227 | +0.0054 |
| SIR2-Wild | 55 | 27.1827 | 0.9083 | 27.5731 | 0.9192 | +0.3904 | +0.0109 |

### Weighted overall

- RDNet: **PSNR 26.3704 / SSIM 0.8744**
- DAI: **PSNR 26.9718 / SSIM 0.8856**

## Main takeaways

- DAI beat RDNet on all three tested public subsets in this run.
- RDNet remained faster per image after all auxiliary assets were downloaded.
- DAI was not uniformly better on every single sample, but it won more often on the reflection-heavy
  cases we manually checked.

## Representative examples

### DAI-win samples

- Real20: `39.jpg`, `110.jpg`
- Nature: `3_35.jpg`, `1-2_71.jpg`
- SIR2-Wild: `024.jpg`, `051.jpg`

### RDNet-win samples

- Real20: `107.jpg`
- Nature: `3_85.jpg`
- SIR2-Wild: `004.jpg`

## Artifact locations from the benchmark run

- RDNet exports: `/home/choihy/github_repos/reflection-exp/outputs/rdnet_eval`
- DAI exports: `/home/choihy/github_repos/reflection-exp/outputs/dai_eval`
- Comparison boards: `/home/choihy/github_repos/reflection-exp/outputs/comparisons`

## Included JSON summaries

- `2026-04-17-rdnet-summary.json`
- `2026-04-17-dai-summary.json`

These are the metric dumps copied into this repository from the benchmark run.
