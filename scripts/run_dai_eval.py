import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, UNet2DConditionModel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel


DATASET_MAP = {
    "Real20": "test/real20_420",
    "Nature": "test/Nature",
    "SIR2-Wild": "test/SIR2/WildSceneDataset",
}


def load_upstream_modules(dai_root: Path):
    sys.path.insert(0, str(dai_root.resolve()))
    from DAI.controlnetvae import ControlNetVAEModel
    from DAI.decoder import CustomAutoencoderKL
    from DAI.pipeline_all import DAIPipeline

    return ControlNetVAEModel, CustomAutoencoderKL, DAIPipeline


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(pred: np.ndarray) -> Image.Image:
    pred = ((pred + 1) / 2).clip(0.0, 1.0)
    pred = (pred * 255).astype(np.uint8)
    return Image.fromarray(pred)


def compute_metrics(pred: Image.Image, gt: Image.Image):
    if pred.size != gt.size:
        pred = pred.resize(gt.size, Image.BICUBIC)
    pred_np = np.array(pred)
    gt_np = np.array(gt)
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
    ssim = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255)
    return float(psnr), float(ssim), pred


def make_concat(input_img: Image.Image, pred_img: Image.Image, gt_img: Image.Image) -> Image.Image:
    imgs = [input_img, pred_img, gt_img]
    labels = ["input", "prediction", "gt"]
    h = max(i.height for i in imgs)
    padded = []
    for img in imgs:
        if img.height != h:
            scale = h / img.height
            img = img.resize((int(img.width * scale), h), Image.BICUBIC)
        padded.append(img)
    total_w = sum(i.width for i in padded)
    canvas = Image.new("RGB", (total_w, h + 28), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, img in zip(labels, padded):
        canvas.paste(img, (x, 28))
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        x += img.width
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dai-root", required=True, help="Path to the Dereflection-Any-Image clone")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_MAP.keys()))
    parser.add_argument("--hf-model", default="sjtu-deepvision/dereflection-any-image-v0")
    args = parser.parse_args()

    ControlNetVAEModel, CustomAutoencoderKL, DAIPipeline = load_upstream_modules(Path(args.dai_root))

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    controlnet = ControlNetVAEModel.from_pretrained(args.hf_model, subfolder="controlnet", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.hf_model, subfolder="unet", torch_dtype=dtype).to(device)
    vae_2 = CustomAutoencoderKL.from_pretrained(args.hf_model, subfolder="vae_2", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(args.hf_model, subfolder="vae").to(device)
    text_encoder = CLIPTextModel.from_pretrained(args.hf_model, subfolder="text_encoder").to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, subfolder="tokenizer", use_fast=False)
    pipeline = DAIPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        scheduler=None,
        feature_extractor=None,
        t_start=0,
    ).to(device)

    summary = {}
    for dataset in args.datasets:
        ds_root = data_root / DATASET_MAP[dataset]
        input_dir = ds_root / "blended"
        gt_dir = ds_root / "transmission_layer"
        out_dir = output_root / dataset
        result_dir = out_dir / "result"
        concat_dir = out_dir / "concat"
        result_dir.mkdir(parents=True, exist_ok=True)
        concat_dir.mkdir(parents=True, exist_ok=True)

        metrics = []
        for input_path in tqdm(sorted(input_dir.iterdir()), desc=dataset):
            if not input_path.is_file():
                continue
            gt_path = gt_dir / input_path.name
            if not gt_path.exists():
                continue

            input_img = load_image(input_path)
            gt_img = load_image(gt_path)
            resolution = None if max(input_img.size) < 768 else 0

            pred = pipeline(
                image=pil_to_tensor(input_img, device),
                prompt="remove glass reflection",
                vae_2=vae_2,
                processing_resolution=resolution,
            ).prediction[0]

            pred_img = tensor_to_pil(pred)
            psnr, ssim, pred_img = compute_metrics(pred_img, gt_img)
            pred_img.save(result_dir / input_path.name)
            make_concat(input_img, pred_img, gt_img).save(concat_dir / input_path.name)
            metrics.append({"file": input_path.name, "psnr": psnr, "ssim": ssim})

        summary[dataset] = {
            "count": len(metrics),
            "avg_psnr": float(np.mean([m["psnr"] for m in metrics])) if metrics else None,
            "avg_ssim": float(np.mean([m["ssim"] for m in metrics])) if metrics else None,
            "samples": metrics,
        }

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(summary[dataset], f, indent=2)

    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
