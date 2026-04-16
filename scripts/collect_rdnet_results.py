import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DATASET_MAP = {
    "Real20": "test/real20_420",
    "Nature": "test/Nature",
    "SIR2-Wild": "test/SIR2/WildSceneDataset",
}


def latest_pred(sample_dir: Path):
    preds = sorted(sample_dir.glob("*_clean_*.png"))
    if not preds:
        return None

    def key(path: Path):
        match = re.search(r"epoch_(\d+)_step_(\d+)", path.name)
        return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)

    return sorted(preds, key=key)[-1]


def make_concat(input_img: Image.Image, pred_img: Image.Image, gt_img: Image.Image):
    imgs = [input_img, pred_img, gt_img]
    labels = ["input", "prediction", "gt"]
    h = max(i.height for i in imgs)
    resized = []
    for img in imgs:
        if img.height != h:
            scale = h / img.height
            img = img.resize((int(img.width * scale), h), Image.BICUBIC)
        resized.append(img)
    total_w = sum(i.width for i in resized)
    canvas = Image.new("RGB", (total_w, h + 28), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, img in zip(labels, resized):
        canvas.paste(img, (x, 28))
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        x += img.width
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_MAP.keys()))
    args = parser.parse_args()

    vis_root = Path(args.visualization_root)
    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    for dataset in args.datasets:
        ds_vis = vis_root / dataset
        ds_data = data_root / DATASET_MAP[dataset]
        input_dir = ds_data / "blended"
        gt_dir = ds_data / "transmission_layer"
        basename_to_input = {path.stem: path for path in input_dir.iterdir() if path.is_file()}
        basename_to_gt = {path.stem: path for path in gt_dir.iterdir() if path.is_file()}

        result_dir = out_root / dataset / "result"
        concat_dir = out_root / dataset / "concat"
        result_dir.mkdir(parents=True, exist_ok=True)
        concat_dir.mkdir(parents=True, exist_ok=True)

        metrics = []
        for sample_dir in sorted(ds_vis.iterdir()):
            if not sample_dir.is_dir():
                continue

            pred_path = latest_pred(sample_dir)
            if pred_path is None:
                continue

            stem = sample_dir.name
            input_path = basename_to_input.get(stem)
            gt_path = basename_to_gt.get(stem)
            if input_path is None or gt_path is None:
                continue

            input_img = Image.open(input_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            pred_img = Image.open(pred_path).convert("RGB")
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

            pred_np = np.array(pred_img)
            gt_np = np.array(gt_img)
            psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
            ssim = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255)
            ext = input_path.suffix or ".png"

            pred_img.save(result_dir / f"{stem}{ext}")
            make_concat(input_img, pred_img, gt_img).save(concat_dir / f"{stem}{ext}")
            metrics.append({"file": f"{stem}{ext}", "psnr": float(psnr), "ssim": float(ssim)})

        summary[dataset] = {
            "count": len(metrics),
            "avg_psnr": float(np.mean([m["psnr"] for m in metrics])) if metrics else None,
            "avg_ssim": float(np.mean([m["ssim"] for m in metrics])) if metrics else None,
            "samples": metrics,
        }

        with open(out_root / dataset / "metrics.json", "w") as f:
            json.dump(summary[dataset], f, indent=2)

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
