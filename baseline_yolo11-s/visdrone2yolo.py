import os
from pathlib import Path
import shutil


def visdrone2yolo(root_dir, target_labels_dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    target_labels_dir.mkdir(parents=True, exist_ok=True)  # make labels directory
    ann_dir = root_dir / 'annotations'
    pbar = tqdm(ann_dir.glob('*.txt'), desc=f'Converting {root_dir}')
    for f in pbar:
        img_path = (root_dir / 'images' / f.with_suffix('.jpg').name)
        img_size = Image.open(img_path).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone ignored regions flag
                    continue
                cls = int(row[5]) - 1  # class id -1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
        # Write all lines at once after processing the file
        out_path = target_labels_dir / f.name
        with open(out_path, 'w') as fl:
            fl.writelines(lines)


if __name__ == '__main__':
    # Hardcoded paths based on the provided locations
    base_path = r"F:\笔记、重要资料等存档\本科学习资料\大三\科研\学术裁缝\科研思路\VisDrone_DataSet"
    output_base = Path(base_path) / 'yolo11_datasets'
    output_base.mkdir(parents=True, exist_ok=True)

    datasets = {
        'VisDrone2019-DET-train': base_path + r'\VisDrone2019-DET-train',
        'VisDrone2019-DET-val': base_path + r'\VisDrone2019-DET-val',
        'VisDrone2019-DET-test-dev': base_path + r'\VisDrone2019-DET-test-dev'
    }

    # Process each dataset
    for name, input_path_str in datasets.items():
        input_dir = Path(input_path_str)
        output_dir = output_base / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to output directory
        src_images = input_dir / 'images'
        dst_images = output_dir / 'images'
        if src_images.exists():
            shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
            print(f"Copied images for {name}")

        # Convert annotations to YOLO labels
        target_labels = output_dir / 'labels'
        visdrone2yolo(input_dir, target_labels)
        print(f"Converted labels for {name}")