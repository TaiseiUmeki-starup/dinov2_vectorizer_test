from pathlib import Path
from math import ceil
from PIL import Image
from tqdm import tqdm

def is_blank(im, blank_mean=250, blank_std=2):
    # グレースケールにし、ほぼ真っ白かを判定
    g = im.convert("L")
    arr = g.histogram()
    total = sum(arr)
    # 平均と分散をヒストグラムから計算（速度優先で numpy 不使用）
    mean = sum(i * v for i, v in enumerate(arr)) / total
    var = sum((i - mean) ** 2 * v for i, v in enumerate(arr)) / total
    std = var ** 0.5
    return mean >= blank_mean and std <= blank_std

def tile_image(img_path, out_dir, tile_size=518, overlap=0.2, blank_mean=250, blank_std=2):
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    stride = int(tile_size * (1 - overlap))
    stride = max(1, stride)
    # 端をカバーするために必要なステップ数を計算
    nx = ceil(max(1, (w - tile_size) / stride)) + 1
    ny = ceil(max(1, (h - tile_size) / stride)) + 1

    saved = 0
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * stride
            y0 = iy * stride
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            # 端を超えた場合は右下を合わせる
            if x1 > w:
                x0, x1 = w - tile_size, w
            if y1 > h:
                y0, y1 = h - tile_size, h
            if x0 < 0 or y0 < 0:
                continue  # 元画像より小さい場合に備えたガード

            crop = im.crop((x0, y0, x1, y1))
            if is_blank(crop, blank_mean, blank_std):
                continue

            out_name = f"{img_path.stem}_x{x0}_y{y0}.png"
            crop.save(out_dir / out_name)
            saved += 1
    return saved

def main(input_folder, output_folder, tile_size=518, overlap=0.2):
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    files = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print("No images found.")
        return

    total_saved = 0
    for img_path in tqdm(files, desc="Tiling"):
        total_saved += tile_image(img_path, out_dir, tile_size=tile_size, overlap=overlap)
    print(f"Done. Saved {total_saved} tiles to {out_dir}")

if __name__ == "__main__":
    # 例: python tile_images.py /path/to/input /path/to/output --tile 518 --overlap 0.2
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input_folder")
    ap.add_argument("output_folder")
    ap.add_argument("--tile", type=int, default=518, dest="tile_size")
    ap.add_argument("--overlap", type=float, default=0.2)
    args = ap.parse_args()
    main(args.input_folder, args.output_folder, tile_size=args.tile_size, overlap=args.overlap)
