import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


METHODS = ["input", "rdnet", "dai", "gt"]
DATASET_MAP = {
    "Real20": "test/real20_420",
    "Nature": "test/Nature",
    "SIR2-Wild": "test/SIR2/WildSceneDataset",
}


def open_img(path):
    return Image.open(path).convert("RGB")


def select_files(metrics_path, top_n=3):
    data = json.loads(Path(metrics_path).read_text())
    samples = sorted(data["samples"], key=lambda x: x["psnr"], reverse=True)
    picks = []
    if samples:
        picks.append(samples[0]["file"])
    if len(samples) > 2:
        picks.append(samples[len(samples) // 2]["file"])
    if len(samples) > 1:
        picks.append(samples[-1]["file"])
    return list(dict.fromkeys(picks))[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--rdnet-root", required=True)
    parser.add_argument("--dai-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--files", nargs="*", default=None)
    args = parser.parse_args()

    files = args.files or select_files(Path(args.rdnet_root) / args.dataset / "metrics.json")

    ds_data = Path(args.data_root) / DATASET_MAP[args.dataset]
    rows = []
    for fname in files:
        paths = {
            "input": ds_data / "blended" / fname,
            "rdnet": Path(args.rdnet_root) / args.dataset / "result" / fname,
            "dai": Path(args.dai_root) / args.dataset / "result" / fname,
            "gt": ds_data / "transmission_layer" / fname,
        }
        if not all(path.exists() for path in paths.values()):
            continue

        imgs = [open_img(paths[key]) for key in METHODS]
        target_h = max(img.height for img in imgs)
        row_imgs = []
        for img in imgs:
            if img.height != target_h:
                scale = target_h / img.height
                img = img.resize((int(img.width * scale), target_h), Image.BICUBIC)
            row_imgs.append(img)

        row_w = sum(img.width for img in row_imgs)
        row = Image.new("RGB", (row_w, target_h + 30), color=(255, 255, 255))
        draw = ImageDraw.Draw(row)
        x = 0
        for label, img in zip(METHODS, row_imgs):
            row.paste(img, (x, 30))
            draw.text((x + 4, 8), label, fill=(0, 0, 0))
            x += img.width
        draw.text((4, target_h + 10), fname, fill=(0, 0, 0))
        rows.append(row)

    if not rows:
        raise SystemExit("No rows created")

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    main()
