import os
import numpy as np
import cv2 as cv
from pathlib import Path
from sys import path

# Конфигурация
RAW_DIR = Path(path[0] + "\\raw")
ANGLES_DIR = Path(path[0] + "\\angles")
CROP_SIZE = 256
ANGLE_START = -5.0
ANGLE_END = 5.0
ANGLE_STEP = 0.1
ANGLES = int((abs(ANGLE_START) + ANGLE_END) / ANGLE_STEP + 1)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def center_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    target = 256
    # центр изображения
    cy, cx = h // 2, w // 2
    # начальные координаты кропа
    y1 = max(0, cy - target // 2)
    x1 = max(0, cx - target // 2)
    # конечные координаты
    y2 = y1 + target
    x2 = x1 + target
    # при необходимости обрезаем за границы
    y2 = min(h, y2)
    x2 = min(w, x2)

    return img[y1:y2, x1:x2]


def rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    """! Поворачивает изображение на заданный угол.
        @param img: Исходное изображение.
        @param degrees: Угол поворота в градусах.
        @return: Повёрнутое изображение. """
    height, width = img.shape[:2]
    center_x, center_y = (width / 2, height / 2)
    matrix = cv.getRotationMatrix2D((center_x, center_y), degrees, 1.0)
    out_image = cv.warpAffine(img, matrix, (width, height))
    return out_image


def main():
    os.makedirs(ANGLES_DIR, exist_ok=True)
    angles = np.linspace(ANGLE_START, ANGLE_END, ANGLES)

    for filename in os.listdir(RAW_DIR):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            continue

        img_path = os.path.join(RAW_DIR, filename)
        try:
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"❌ Не удалось открыть {img_path}: {e}")
            continue

        w, h = img.shape[:2]
        if w < CROP_SIZE or h < CROP_SIZE:
            print(f"⚠️ Пропущено {filename}: размер ({w}x{h}) меньше {CROP_SIZE}x{CROP_SIZE}")
            continue

        output_subdir = os.path.join(ANGLES_DIR, ("rotate_" + name))
        os.makedirs(output_subdir, exist_ok=True)

        for angle in angles:
            rotated = rotate_image(img, angle)

            cropped = center_crop(rotated)

            # Формат имени: угол с заменой точки и знака (например: -2.5 → m2_5.jpg)
            angle_str = f"{angle:.1f}"
            safe_angle = angle_str.replace('-', 'm').replace('.', 'p')
            out_filename = f"0p0_0p0_{safe_angle}.jpg"
            out_path = os.path.join(output_subdir, out_filename)

            cv.imwrite(out_path, cropped)

        print(f"✅ Готово: {filename}")

    print("🎉 Все изображения обработаны!")

if __name__ == "__main__":
    main()