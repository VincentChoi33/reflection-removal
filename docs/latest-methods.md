# Reflection-removal references worth tracking

This project started as an older conditioned diffusion prototype. If the goal is to get materially
better results on real single-image reflection removal in 2026, the most useful references we found
are below.

## Practical recommendation

### 1) Strong baseline path

- **RDNet / XReflection**
- Good choice when the goal is a stable, reproducible baseline on public SIRR datasets.
- Faster than diffusion-style methods once weights/assets are cached.

Relevant upstreams:

- XReflection: <https://github.com/hainuo-wang/XReflection>
- RDNet paper: *Reversible Decoupling Network for Single Image Reflection Removal*

### 2) Modern diffusion reference path

- **Dereflection Any Image**
- Better fit if you want to keep the project aligned with modern diffusion priors instead of
  continuing a 1000-step prototype pipeline.
- In our public benchmark run it beat RDNet on all three tested subsets.

Relevant upstreams:

- Dereflection Any Image: <https://github.com/Abuuu122/Dereflection-Any-Image>
- Paper: <https://arxiv.org/abs/2503.17347>

## Why these matter for this repo

The current repository is best treated as:

- a reflection-removal playground with an old conditioned diffusion prototype inside it,
- useful for understanding an older design,
- **not** the most competitive place to continue from if raw quality is the main goal.

If restarting with performance in mind, the most realistic direction is:

1. benchmark against RDNet/XReflection,
2. benchmark against a modern diffusion reference,
3. only then decide whether to salvage or replace the original prototype.

## Other references to inspect

- **Revisiting Single Image Reflection Removal in the Wild (CVPR 2024)**
  - important mostly for data and reflection-location-awareness
- **L-DiffER (ECCV 2024)**
  - useful if exploring stronger conditioning/control inside diffusion pipelines
- **OpenRR-5k**
  - useful as a dataset/benchmark reference

## Project recommendation

For this repository specifically:

- keep the original prototype code as historical/experimental context,
- use the scripts under `scripts/` and configs under `configs/` to compare against stronger
  external baselines,
- prefer adding new work as benchmarked experiments instead of silently iterating the old model.
