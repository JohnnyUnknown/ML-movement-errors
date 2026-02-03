"""
Скрипт для генерации набора обучающих изображений с имитацией смещений и поворотов.

Назначение:
- Загружает исходные изображения из директории 'raw'
- Для каждого изображения создаёт серию кропов 300x300 пикселей:
  * Случайными смещениями в пределах ±10 пикселей по X/Y
  * С поворотами в диапазоне от -3.0° до +3.0° с шагом 0.05°
- Сохраняет кропы в структурированную директорию 'images' с именами,
  кодирующими параметры трансформации (смещение и угол поворота)
- Дополнительно сохраняет центральный кроп без трансформации (эталон)

Формат имён файлов:
  dx_dy_angle.jpg
  Пример: "3p5_m2p0_+1p50.jpg" → смещение +3.5 по X, -2.0 по Y, поворот +1.50°
  Особенности кодировки:
    '.' → 'p' (point)
    '-' → 'm' (minus)
    '+' → опускается (положительные значения без знака)
"""
import os
import cv2
import numpy as np
from pathlib import Path
import config


# Директории для входных и выходных данных
RAW_DIR = config.RAW_DIR          # Исходные изображения (входная директория)
IMG_DIR = config.IMG_DIR          # Выходные кропы с поворотами

# Параметры генерации
CROP_SIZE = config.CROP_SIZE      # Размер квадратного кропа (пиксели)
MAX_ANGLE = config.MAX_ANGLE      # Максимальный угол поворота (градусы)
ANGLE_STEP = config.ANGLE_STEP    # Шаг изменения угла (градусы)
MAX_OFFSET = 10.0                 # Максимальное смещение центра кропа (пиксели)
ANGLES = int(MAX_ANGLE * 2 / ANGLE_STEP + 1)  # Количество шагов поворота: от -3.0° до +3.0° с шагом 0.05°
angles = np.linspace(-MAX_ANGLE, MAX_ANGLE, ANGLES)  # Массив углов поворота

# Поддерживаемые форматы изображений
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def get_image_files(directory):
    """
    Сканирует директорию и возвращает список путей к изображениям с поддерживаемыми расширениями.
    
    :param directory: Путь к директории (объект Path или строка)
    :return: Список объектов Path, указывающих на найденные изображения
    """
    files = []
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in SUPPORTED_EXTS:
            files.append(Path(directory) / f)
    return files


def safe_format(val, angle_flag=False):
    """
    Форматирует числовое значение для безопасного использования в имени файла.
    
    Правила преобразования:
      - Точка '.' заменяется на 'p' (например, 3.5 → 3p5)
      - Минус '-' заменяется на 'm' (например, -2.0 → m2p0)
      - Плюс '+' удаляется (положительные значения без знака)
      - Для углов используется формат с двумя знаками после запятой (+1.50)
      - Для смещений — один знак после запятой (+3.5)
    
    :param val: Числовое значение для форматирования
    :param angle_flag: Флаг режима угла (точность 2 знака) или смещения (1 знак)
    :return: Строка в безопасном для файловой системы формате
    """
    if angle_flag:
        s = f"{val:+.2f}"  # Формат для углов: "+1.50"
    else:
        s = f"{val:+.1f}"  # Формат для смещений: "+3.5"
    
    # Замена символов для совместимости с именами файлов
    s = s.replace('+', '').replace('.', 'p').replace('-', 'm')
    return s


def generate_transformed_crop(img, dx, dy, angle_deg):
    """
    Создаёт кроп изображения с заданным смещением центра и поворотом.
    
    Алгоритм:
      1. Вычисляется новый центр кропа с учётом смещения (dx, dy)
      2. Строится матрица аффинного преобразования для поворота вокруг нового центра
      3. Применяется поворот ко всему изображению
      4. Из повёрнутого изображения извлекается квадратный кроп заданного размера
    
    :param img: Исходное изображение (2D массив в градациях серого)
    :param dx: Смещение центра по оси X (пиксели)
    :param dy: Смещение центра по оси Y (пиксели)
    :param angle_deg: Угол поворота в градусах (положительное — против часовой стрелки)
    :return: Кропированное изображение (массив) или None при выходе за границы
    """
    height, width = img.shape[:2]
    # Новый центр кропа с учётом смещения
    center_x, center_y = (width // 2 + dx, height // 2 + dy)
    
    # Матрица поворота вокруг смещённого центра
    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
    out_image = cv2.warpAffine(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    
    # Координаты кропа с центрированием вокруг смещённой точки
    y1 = max(0, center_y - CROP_SIZE // 2)
    x1 = max(0, center_x - CROP_SIZE // 2)
    y2 = min(height, y1 + CROP_SIZE)
    x2 = min(width, x1 + CROP_SIZE)
    
    # Проверка корректности кропа (должен быть полного размера)
    if (y2 - y1) < CROP_SIZE or (x2 - x1) < CROP_SIZE:
        return None
    
    return out_image[y1:y2, x1:x2]


def crop_center(img):
    """
    Извлекает центральный квадратный кроп заданного размера без трансформаций.
    
    :param img: Исходное изображение
    :return: Центральный кроп или None, если изображение слишком мало
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = CROP_SIZE // 2
    left = cx - half
    top = cy - half

    # Проверка выхода за границы изображения
    if left < 0 or top < 0 or left + CROP_SIZE > w or top + CROP_SIZE > h:
        return None

    return img[top:top + CROP_SIZE, left:left + CROP_SIZE]


def main():
    """
    Основной цикл генерации обучающих кропов.
    
    Этапы обработки:
      1. Создание выходной директории (если не существует)
      2. Поиск всех изображений в RAW_DIR
      3. Для каждого изображения:
          - Загрузка в градациях серого
          - Проверка минимального размера
          - Генерация кропов для каждого угла поворота:
              * Случайное смещение в пределах ±MAX_OFFSET
              * Применение поворота и кропирование
              * Сохранение с кодированием параметров в имени файла
          - Сохранение эталонного центрального кропа (без смещения/поворота)
    """
    # Создание структуры выходных директорий
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Получение списка исходных изображений
    image_files = get_image_files(RAW_DIR)

    if not image_files:
        print(f"Нет изображений в {RAW_DIR} с поддерживаемыми расширениями: {SUPPORTED_EXTS}")
        return

    # Обработка каждого изображения
    for img_path in image_files:
        print(f"Обработка: {img_path.name}")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Не удалось загрузить: {img_path}")
            continue

        h, w = img.shape[:2]
        if w < CROP_SIZE or h < CROP_SIZE:
            print(f"Изображение слишком маленькое ({w}x{h}), пропускаем")
            continue

        # Создание поддиректории для текущего изображения
        out_dir = IMG_DIR / ("shift_rotate_" + img_path.stem)
        out_dir.mkdir(exist_ok=True)

        saved = 0
        all_shifts = []

        # Генерация кропов для каждого угла поворота
        for angle in angles:
            # Случайное смещение в пределах допустимого диапазона
            dx = int(np.random.uniform(-MAX_OFFSET + 1, MAX_OFFSET))
            dy = int(np.random.uniform(-MAX_OFFSET + 1, MAX_OFFSET))
            
            # Получение трансформированного кропа
            crop = generate_transformed_crop(img, dx, dy, angle)
            if crop is None:
                continue  # Пропуск некорректных кропов
            
            # Формирование имени файла с кодированием параметров
            dx_str = safe_format(dx)
            dy_str = safe_format(dy)
            angle_str = safe_format(angle, angle_flag=True)
            filename = f"{dx_str}_{dy_str}_{angle_str}.jpg"
            out_path = out_dir / filename
            
            # Сохранение кропа
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            all_shifts.append((dx, dy))
            saved += 1

        # Сохранение эталонного кропа (центр без смещения и поворота)
        center_crop = crop_center(img)
        if center_crop is not None:
            cv2.imwrite(str(out_dir / "0p0_0p0_0p00.jpg"), center_crop, 
                       [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1

        print(f"Сохранено {saved} кропов в {out_dir.name}")

    print("\nГотово! Все изображения обработаны.")


if __name__ == "__main__":
    main()