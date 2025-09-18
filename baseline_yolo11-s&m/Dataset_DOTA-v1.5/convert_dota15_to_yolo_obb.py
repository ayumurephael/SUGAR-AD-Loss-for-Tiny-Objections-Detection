```python
# -*- coding: utf-8 -*-
"""
DOTA-v1.5 -> Ultralytics YOLO OBB converter

How to use on your server
-------------------------
1) Put this file somewhere and run:
   python convert_dota15_to_yolo_obb.py \
       --base /root/autodl-tmp/DOTA-v1.5 \
       --out  /root/autodl-tmp/DOTA15_yolo_obb \
       --filter-difficult 0 \
       --symlink 0

2) Then train with Ultralytics (example):
   yolo obb train data=/root/autodl-tmp/DOTA15_yolo_obb/dota15-obb.yaml model=yolo11s-obb.pt imgsz=1024 epochs=300

Label format produced (Ultralytics YOLO OBB):
---------------------------------------------
Each line in labels/*.txt:
    <class_id> x1 y1 x2 y2 x3 y3 x4 y4
- coordinates are NORMALIZED to [0,1] by image width/height
- no confidence score in training labels
- we keep all objects by default (set --filter-difficult 1 to drop difficult==1)
- Filenames prefixed with 'split___' (e.g., 'val___P0001.png') for Ultralytics JSON eval compatibility

Class order (from your table, 0-based):
---------------------------------------
0 plane, 1 ship, 2 storage tank, 3 baseball diamond, 4 tennis court,
5 basketball court, 6 ground track field, 7 harbor, 8 bridge, 9 large vehicle,
10 small vehicle, 11 helicopter, 12 roundabout, 13 soccer ball field,
14 swimming pool, 15 container crane
"""
# python convert_dota15_to_yolo_obb.py --base /root/autodl-tmp/DOTA-v1.5 --out /root/autodl-tmp/DOTA15_yolo_obb --filter-difficult 0 --symlink 0
import os
import sys
import glob
import math
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

# -----------------------------
# Class mapping (your order)
# -----------------------------
CLASS_NAMES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane",
]

def norm_name(name: str) -> str:
    s = name.strip().lower().replace("_", " ")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s

CANONICAL = {norm_name(n): i for i, n in enumerate(CLASS_NAMES)}
ALIASES = {
    "storage-tank": "storage tank",
    "baseball-diamond": "baseball diamond",
    "tennis-court": "tennis court",
    "basketball-court": "basketball court",
    "ground-track-field": "ground track field",
    "large-vehicle": "large vehicle",
    "small-vehicle": "small vehicle",
    "soccer-ball-field": "soccer ball field",
    "swimming-pool": "swimming pool",
    "container-crane": "container crane",
}
for k, v in ALIASES.items():
    CANONICAL[norm_name(k)] = CLASS_NAMES.index(v)

# -----------------------------
# Helpers
# -----------------------------
def find_all_images(base: Path) -> Dict[str, Path]:
    img_globs = [
        base / "train" / "images" / "part1" / "images" / "*.png",
        base / "train" / "images" / "part2" / "images" / "*.png",
        base / "train" / "images" / "part3" / "images" / "*.png",
        base / "val"   / "images" / "part1" / "images" / "*.png",
        base / "test"  / "images" / "part1" / "images" / "*.png",
        base / "test"  / "images" / "part2" / "images" / "*.png",
    ]
    mapping = {}
    for g in img_globs:
        for p in glob.glob(str(g)):
            pth = Path(p)
            mapping[pth.stem] = pth
    return mapping

def parse_dota_line(line: str) -> Tuple[List[float], str, int]:
    parts = line.strip().split()
    if len(parts) < 10:
        raise ValueError(f"Bad line (len={len(parts)}): {line}")
    coords = list(map(float, parts[:8]))
    cls_name = parts[8]
    try:
        difficult = int(parts[9])
    except ValueError:
        difficult = 0
    return coords, cls_name, difficult

def normalize_polygon(coords: List[float], w: int, h: int) -> List[float]:
    out = []
    for i, v in enumerate(coords):
        if i % 2 == 0:  # x
            out.append(max(0.0, min(1.0, v / max(1, w))))
        else:
            out.append(max(0.0, min(1.0, v / max(1, h))))
    return out

def write_yolo_obb_label(out_txt: Path, items: List[Tuple[int, List[float]]]):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for cls_id, poly in items:
            poly_str = " ".join(f"{v:.6f}" for v in poly)
            f.write(f"{cls_id} {poly_str}\n")

def robust_class_to_id(name: str) -> int:
    key = norm_name(name)
    return CANONICAL.get(key, -1)

def copy_or_link(src: Path, dst: Path, symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def convert_split(
    split: str,
    base: Path,
    out_root: Path,
    img_index: Dict[str, Path],
    label_dir: Path,
    filter_difficult: bool,
    symlink: bool,
) -> Tuple[int, int, int]:
    assert split in {"train", "val"}
    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    label_files = sorted(glob.glob(str(label_dir / "*.txt")))
    n_imgs = n_objs = n_skip = 0

    for lf in label_files:
        lf_path = Path(lf)
        stem = lf_path.stem
        if stem not in img_index:
            print(f"[WARN] label {lf_path.name}: no matching image found")
            n_skip += 1
            continue
        img_path = img_index[stem]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read image: {img_path}")
            n_skip += 1
            continue
        h, w = img.shape[:2]

        items = []
        with open(lf_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start_idx = 0
        if lines and lines[0].lower().startswith("imagesource"):
            start_idx = 2 if len(lines) > 1 and lines[1].lower().startswith("gsd") else 1

        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            try:
                coords, cls_name, difficult = parse_dota_line(line)
            except Exception as e:
                print(f"[WARN] bad line in {lf_path.name}: {line} ({e})")
                n_skip += 1
                continue

            if filter_difficult and difficult == 1:
                continue

            cls_id = robust_class_to_id(cls_name)
            if cls_id < 0:
                print(f"[WARN] unknown class '{cls_name}' in {lf_path.name}, skipped")
                n_skip += 1
                continue

            poly = normalize_polygon(coords, w, h)
            items.append((cls_id, poly))
            n_objs += 1

        # 添加 split___ 前缀到文件名，以支持 Ultralytics JSON eval
        prefixed_stem = f"{split}___{stem}"
        out_lbl_name = f"{prefixed_stem}.txt"
        out_lbl = out_lbl_dir / out_lbl_name
        write_yolo_obb_label(out_lbl, items)

        out_img_name = f"{prefixed_stem}.png"
        out_img = out_img_dir / out_img_name
        copy_or_link(img_path, out_img, symlink=symlink)
        n_imgs += 1

    return n_imgs, n_objs, n_skip

def copy_test_images(base: Path, out_root: Path, symlink: bool):
    out_img_dir = out_root / "images" / "test"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        base / "test" / "images" / "part1" / "images" / "*.png",
        base / "test" / "images" / "part2" / "images" / "*.png",
    ]
    cnt = 0
    for g in patterns:
        for p in glob.glob(str(g)):
            src = Path(p)
            stem = src.stem
            # 添加 test___ 前缀到文件名，以支持 Ultralytics JSON eval
            prefixed_stem = f"test___{stem}"
            dst_name = f"{prefixed_stem}.png"
            dst = out_img_dir / dst_name
            copy_or_link(src, dst, symlink=symlink)
            cnt += 1
    return cnt

def write_yaml(out_root: Path, yaml_name: str = "dota15-obb.yaml"):
    yaml_path = out_root / yaml_name
    names_block = "\n".join([f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)])
    content = f"""# Ultralytics YOLO OBB dataset YAML for DOTA-v1.5
path: {out_root}
task: obb
train: images/train
val: images/val
test: images/test

names:
{names_block}
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote dataset YAML: {yaml_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="Path to DOTA-v1.5 root (contains train/ val/ test/)")
    ap.add_argument("--out",  type=str, required=True, help="Output root for YOLO OBB layout")
    ap.add_argument("--filter-difficult", type=int, default=0, help="1=drop difficult==1 objects")
    ap.add_argument("--symlink", type=int, default=0, help="1=use symlinks for images instead of copying")
    args = ap.parse_args()

    base = Path(args.base)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    img_index = find_all_images(base)
    print(f"[INFO] Found {len(img_index)} unique images by stem across train/val/test")

    train_label_dir = base / "train" / "labelTxt-v1.5" / "DOTA-v1.5_train"
    val_label_dir   = base / "val"   / "labelTxt-v1.5" / "DOTA-v1.5_val"
    if not train_label_dir.exists():
        raise FileNotFoundError(f"Train label dir not found: {train_label_dir}")
    if not val_label_dir.exists():
        raise FileNotFoundError(f"Val label dir not found: {val_label_dir}")

    tri, tro, trs = convert_split("train", base, out_root, img_index, train_label_dir, bool(args.filter_difficult), bool(args.symlink))
    vai, vao, vas = convert_split("val",   base, out_root, img_index, val_label_dir,   bool(args.filter_difficult), bool(args.symlink))
    tec = copy_test_images(base, out_root, symlink=bool(args.symlink))
    write_yaml(out_root)

    print("\n========== SUMMARY ==========")
    print(f"Train: images={tri}, objects={tro}, skipped={trs}")
    print(f"Val:   images={vai}, objects={vao}, skipped={vas}")
    print(f"Test:  images={tec}")
    print(f"Output root: {out_root}")
    print("Label format: <cls> x1 y1 x2 y2 x3 y3 x4 y4  (normalized)")
    print("Filenames prefixed with 'split___' for Ultralytics compatibility")
    print("=============================")

if __name__ == "__main__":
    main()
