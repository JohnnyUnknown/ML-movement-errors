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
MAX_OFFSET = 15.0      # пикселей
MAX_ANGLE = 5.0        # градусов
NUM_CROPS = 100

# Поддерживаемые расширения
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def get_image_files(directory):
    files = []
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in SUPPORTED_EXTS:
            files.append(Path(directory) / f)
    return files

def safe_format(val):
    """Форматирует число: заменяет '.' → 'p', '-' → 'm'"""
    s = f"{val:+.1f}"  # например: "+3.2" или "-1.0"
    s = s.replace('+', '').replace('.', 'p').replace('-', 'm')
    return s

def generate_transformed_crop(img, dx, dy, angle_deg, crop_size):
    height, width = img.shape[:2]
    center_x, center_y = (width // 2 + dx, height // 2 + dy)
    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
    out_image = cv2.warpAffine(img, matrix, (width, height))

    height, width = out_image.shape[:2]
    y1 = max(0, center_y - crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    # конечные координаты
    y2 = y1 + crop_size
    x2 = x1 + crop_size
    # при необходимости обрезаем за границы
    y2 = min(height, y2)
    x2 = min(width, x2)

    return out_image[y1:y2, x1:x2]

def crop_center(img, crop_size):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    left = cx - half
    top = cy - half

    if left < 0 or top < 0 or left + crop_size > w or top + crop_size > h:
        return None

    return img[top:top + crop_size, left:left + crop_size]

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

        out_dir = ANGLES_DIR / ("shift_rotate_" + img_path.stem)
        out_dir.mkdir(exist_ok=True)

        saved = 0
        all_shifts = []
        while saved < NUM_CROPS:
            dx = int(np.random.uniform(-MAX_OFFSET+1, MAX_OFFSET))
            dy = int(np.random.uniform(-MAX_OFFSET+1, MAX_OFFSET))
            angle = round(np.random.uniform(-MAX_ANGLE, MAX_ANGLE), 1)

            if (dx, dy) in all_shifts or (dx, dy) == (0, 0):
                continue

            crop = generate_transformed_crop(img, dx, dy, angle, CROP_SIZE)
            if crop is None:
                continue

            # Форматируем имена
            dx_str = safe_format(dx)
            dy_str = safe_format(dy)
            angle_str = safe_format(angle)

            filename = f"{dx_str}_{dy_str}_{angle_str}.jpg"
            out_path = out_dir / filename
            cv2.imwrite(str(out_path), crop)
            all_shifts.append((dx, dy))
            saved += 1

        # Сохраняем центральный кроп без сдвига и поворота
        center_crop = crop_center(img, CROP_SIZE)
        if center_crop is not None:
            cv2.imwrite(str(out_dir / "0p0_0p0_0p0.jpg"), center_crop)

        print(f"  Сохранено {saved} кропов в {out_dir}")

    print("Готово!")

if __name__ == "__main__":
    main()