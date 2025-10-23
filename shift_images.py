import os
import cv2
import numpy as np
from pathlib import Path
from sys import path

# Директории
RAW_DIR = Path(path[0] + "\\raw")
ANGLES_DIR = Path(path[0] + "\\angles")

# Параметры
CROP_SIZE = 256
MAX_OFFSET = 15.0  # пикселей
NUM_CROPS = 100

# Поддерживаемые расширения
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def get_image_files(directory):
    files = []
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in SUPPORTED_EXTS:
            files.append(Path(directory) / f)
    return files

def shift_center_crop(img, dx, dy, crop_size):
    h, w = img.shape[:2]
    # центр изображения
    cy, cx = h // 2 + dy, w // 2 + dx
    # начальные координаты кропа
    y1 = max(0, cy - crop_size // 2)
    x1 = max(0, cx - crop_size // 2)
    # конечные координаты
    y2 = y1 + crop_size
    x2 = x1 + crop_size
    # при необходимости обрезаем за границы
    y2 = min(h, y2)
    x2 = min(w, x2)

    return img[y1:y2, x1:x2]

def crop_center(img, crop_size):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    left_i = cx - half
    top_i = cy - half

    if left_i < 0 or top_i < 0 or left_i + crop_size > w or top_i + crop_size > h:
        return None

    crop = img[top_i:top_i + crop_size, left_i:left_i + crop_size]
    return crop

def main():
    ANGLES_DIR.mkdir(parents=True, exist_ok=True)
    image_files = get_image_files(RAW_DIR)

    if not image_files:
        print(f"Нет изображений в {RAW_DIR}")
        return

    for img_path in image_files:
        print(f"Обработка: {img_path.name}")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Не удалось загрузить: {img_path}")
            continue

        h, w = img.shape[:2]
        if w < CROP_SIZE or h < CROP_SIZE:
            print(f"  Изображение слишком маленькое: {w}x{h}")
            continue


        out_dir = ANGLES_DIR / ("shift_" + img_path.stem)
        out_dir.mkdir(exist_ok=True)

        saved = 0
        all_shifts = []
        while saved < NUM_CROPS:  
            dx = int(np.random.uniform(-MAX_OFFSET+1, MAX_OFFSET))
            dy = int(np.random.uniform(-MAX_OFFSET+1, MAX_OFFSET))
            
            if (dx, dy) in all_shifts or (dx, dy) == (0, 0):
                continue

            crop = shift_center_crop(img, dx, dy, CROP_SIZE)
            if crop is None:
                continue

            dx_str = f"{dx:.1f}".replace(".", "p").replace("-", "m")
            dy_str = f"{dy:.1f}".replace(".", "p").replace("-", "m")
            
            filename = f"{dx_str.replace('.', 'p')}_{dy_str.replace('.', 'p')}_0p0.jpg"

            out_path = out_dir / filename
            cv2.imwrite(str(out_path), crop)

            all_shifts.append((dx, dy))
            saved += 1

        cv2.imwrite((out_dir / "0p0_0p0_0p0.jpg"), crop_center(img, CROP_SIZE))
        print(f"  Сохранено {saved} кропов в {out_dir}")

    print("Готово!")

if __name__ == "__main__":
    main()