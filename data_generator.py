"""
Скрипт для извлечения комплексного набора признаков качества изображений и сопоставления.

ОСНОВНАЯ ЗАДАЧА:
Генерация обучающего датасета для задачи коррекции измеренных смещений между изображениями.
Для каждой пары изображений (эталонное vs трансформированное) извлекаются:
  1. Измеренные смещения (dx, dy) методом фазовой корреляции
  2. Истинные смещения (из имени файла)
  3. Метрики качества изображения (PSNR, SSIM, MS-SSIM, VIF, FSIM)
  4. Статистические признаки (контраст, энтропия, градиентная энергия, резкость, SNR)

ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ:
  - deviation_dx: разница между истинным и измеренным смещением по X
  - deviation_dy: разница между истинным и измеренным смещением по Y 
  Эти величины используются для обучения модели коррекции систематических ошибок фазовой корреляции.
"""

import os
import cv2
import numpy as np
import pandas as pd
import re
from sys import path
from skimage import measure
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import piq
import config


IMG_DIR = config.IMG_DIR      # Директория с входными изображениями
PARAMS_DIR = path[0] + "\\parameters"         # Директория для промежуточных результатов 
img_size = (config.CROP_SIZE, config.CROP_SIZE)  # Ожидаемый размер изображений (пиксели)


def compute_quality_metrics(ref_img, curr_img):
    """
    Вычисляет комплекс метрик качества изображения между эталоном и текущим изображением.
    
    ПОДДЕРЖИВАЕМЫЕ МЕТРИКИ:
      - PSNR (Peak Signal-to-Noise Ratio): отношение пиковой мощности сигнала к мощности шума
        * Чем выше → тем лучше качество (типичные значения: 30-50 дБ)
      
      - SSIM (Structural Similarity Index): мера структурного сходства
        * Диапазон [0, 1], где 1 = идеальное совпадение
      
      - MS-SSIM (Multi-Scale SSIM): мультимасштабная версия SSIM
        * Учитывает восприятие на разных пространственных частотах
      
      - VIF (Visual Information Fidelity): мера сохранения визуальной информации
        * Основана на модели восприятия человеческого зрения
      
      - FSIM (Feature Similarity Index): сходство на основе фазовых конгруэнций и градиентов
        * Устойчива к геометрическим трансформациям
    
    ВАЖНЫЕ ЗАМЕЧАНИЯ:
      - Для метрик на основе PyTorch (MS-SSIM, VIF, FSIM) требуется преобразование в 3-канальное RGB
      - Все изображения нормализуются в диапазон [0, 1] перед вычислением
      - Для градаций серого выполняется дублирование канала для совместимости с 3-канальными метриками
    
    :param ref_img: Эталонное изображение (uint8, 2D для grayscale или 3D для цветного)
    :param curr_img: Текущее изображение (того же формата и размера)
    :return: Словарь с вычисленными метриками
    """
    metrics = {}
    
    # Преобразование в RGB для совместимости с метриками, требующими 3 канала
    if len(ref_img.shape) == 2:  # Grayscale → RGB
        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
        curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
    else:
        ref_rgb = ref_img
        curr_rgb = curr_img

    # --- PSNR (Peak Signal-to-Noise Ratio) ---
    # Классическая метрика качества, основанная на среднеквадратичной ошибке
    mse = np.mean((ref_rgb - curr_rgb) ** 2)
    if mse == 0:
        metrics["psnr"] = 100  # Идеальное качество
    else:
        metrics['psnr'] = compare_psnr(ref_img, curr_img, data_range=255)

    # --- SSIM (Structural Similarity Index) ---
    # Учитывает яркость, контраст и структуру (более коррелирует с восприятием человека)
    metrics['ssim'] = compare_ssim(ref_img, curr_img, multichannel=True, data_range=255)

    # --- Подготовка тензоров для PyTorch-метрик ---
    # Преобразование: H×W×C (numpy) → [1, C, H, W] (PyTorch)
    def to_tensor(img):
        img_float = img.astype(np.float32) / 255.0  # Нормализация в [0, 1]
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)  # [H,W,C] → [1,C,H,W]
        return tensor

    ref_tensor = to_tensor(ref_rgb)
    curr_tensor = to_tensor(curr_rgb)

    # --- MS-SSIM (Multi-Scale SSIM) ---
    # Вычисляет SSIM на нескольких масштабах для лучшей корреляции с восприятием
    metrics['ms_ssim'] = piq.multi_scale_ssim(ref_tensor, curr_tensor, data_range=1.0).item()

    # --- VIF (Visual Information Fidelity) ---
    # Оценивает, сколько визуальной информации сохранено в искажённом изображении
    metrics['vif'] = piq.vif_p(ref_tensor, curr_tensor, data_range=1.0).item()

    # --- FSIM (Feature Similarity Index) ---
    # Использует фазовые конгруэнции (инвариантны к яркости) и градиенты как признаки
    metrics['fsim'] = piq.fsim(ref_tensor, curr_tensor, data_range=1.0).item()

    return metrics


def parse_shift_angle_from_filename(filename):
    """
    Извлекает истинные параметры трансформации из имени файла.
    
    Формат имени: "dx_dy_angle.jpg"
    Примеры: 
        "3p5_m2p0_1p50.jpg" → dx=+3.5, dy=-2.0, angle=1.50°
        "0p0_0p0_0p00.jpg"   → dx=0, dy=0, angle=0.00°
    
    Кодировка:
        'p' заменяет десятичную точку ('.')
        'm' заменяет минус ('-')
        Положительные значения без знака '+'
    
    :param filename: Имя файла (например, "3p5_m2p0_1p50.jpg")
    :return: Кортеж (dx, dy, angle) в пикселях/градусах или (None, None, None) при ошибке
    """
    name, _ = os.path.splitext(filename)
    parts = name.split('_')
    
    # Проверка структуры имени (должно быть ровно 3 компонента: dx, dy, angle)
    if len(parts) != 3:
        return None, None, None

    dx_str, dy_str, angle_str = parts

    def parse_component(s):
        # Допустимые шаблоны: "m123p45", "123p45", "m123", "123"
        if not re.fullmatch(r'm?\d+(p\d+)?', s):
            return None
        try:
            s_clean = s.replace('p', '.')
            if s_clean.startswith('m'):
                s_clean = '-' + s_clean[1:]
            return float(s_clean)
        except Exception:
            return None

    angle = parse_component(angle_str)
    dx = parse_component(dx_str)
    dy = parse_component(dy_str)

    if angle is None or dx is None or dy is None:
        return None, None, None

    return dx, dy, angle


def compute_snr(img_float):
    """Вычисляет SNR (отношение сигнал/шум) в dB"""
    mean = np.mean(img_float)
    var = np.var(img_float)
    if var == 0:
        return 0.0
    snr = 10 * np.log10(mean**2 / var)
    return float(snr)


def main():
    """
    Основной цикл обработки изображений для извлечения признаков.
    
    АЛГОРИТМ:
      1. Обход всех поддиректорий в IMG_DIR (каждая = один сценарий с трансформациями)
      2. Для каждой поддиректории:
          a. Загрузка эталонного изображения (0p0_0p0_0p00.jpg)
          b. Создание окна Ханна для фазовой корреляции
          c. Обработка всех трансформированных изображений:
               - Загрузка и нормализация
               - Вычисление смещения методом фазовой корреляции (cv2.phaseCorrelate)
               - Извлечение статистических признаков (контраст, энтропия, градиенты и т.д.)
               - Расчёт метрик качества относительно эталона
               - Парсинг истинных параметров из имени файла
          e. Сортировка результатов по углу поворота
      3. Сохранение результатов в csv-файлы в директории parameters
    
    ВАЖНЫЕ ЗАМЕЧАНИЯ:
      - Используется окно Ханна для подавления граничных эффектов в фазовой корреляции
      - Нормализация яркости до [0, 255] обеспечивает стабильность вычислений
      - Инверсия знака истинных смещений (кроме нулевых) для соответствия системе координат:
          * В изображениях: +Y → вниз
          * При повороте против часовой стрелки объект смещается вверх → отрицательное смещение в изображении
    """
    os.makedirs(PARAMS_DIR, exist_ok=True)
    len_dirs = None

    for root, dirs, files in os.walk(IMG_DIR):
        if len_dirs is None:
            len_dirs = len(dirs) 
            
        if os.path.normpath(root) == os.path.normpath(IMG_DIR):
            continue

        ref_path = os.path.join(root, "0p0_0p0_0p00.jpg")
        if not os.path.isfile(ref_path):
            print(f"Пропущена папка {root}: не найдено 0p0_0p0_0p00.jpg")
            continue

        # Создание 2D окна Ханна для фазовой корреляции (подавление спектральной утечки)
        hann = cv2.createHanningWindow(img_size, cv2.CV_32F)

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX)

        if ref_img is None:
            print(f"Не удалось загрузить {ref_path}")
            continue

        results = []

        for file in files:
            img_path = os.path.join(root, file)

            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            curr_img = cv2.normalize(curr_img, None, 0, 255, cv2.NORM_MINMAX)

            if curr_img is None:
                print(f"Не удалось загрузить {img_path}")
                continue

            # Убедимся, что размеры совпадают
            if curr_img.shape != ref_img.shape:
                print(f"Размеры не совпадают: {ref_path} vs {img_path}")
                continue

            # Вычисляем смещение с помощью phase correlation
            try:
                (dx, dy), response = cv2.phaseCorrelate(
                            np.float32(ref_img),
                            np.float32(curr_img),
                            hann
                            )
                
                true_dx, true_dy, angle = parse_shift_angle_from_filename(file)
                    
                # --- Дополнительные метрики изображения ---
                img_float = curr_img.astype(np.float32) / 255.0

                # Средняя яркость (математическое ожидание интенсивности)
                mean_brightness = float(np.mean(img_float))
                
                # Медианная яркость (устойчивая к выбросам)
                median_brightness = float(np.median(img_float))

                # Контраст (стандартное отклонение яркости)
                contrast = float(np.std(img_float))

                # Энтропия (информационная сложность по Шеннону)
                # Высокая энтропия = больше деталей/шума, низкая = однородные области
                entropy = float(measure.shannon_entropy(curr_img))

                # Градиентная энергия (интенсивность перепадов яркости)
                # Характеризует наличие границ и текстур
                gx, gy = np.gradient(img_float)
                gradient_energy = float(np.mean(gx**2 + gy**2))

                # Резкость (через дисперсию лапласиана)
                # Лапласиан усиливает высокочастотные компоненты (границы)
                lap_var = float(cv2.Laplacian(curr_img, cv2.CV_64F).var())
                
                # Отношение сигнал/шум (в децибелах)
                snr = compute_snr(img_float)
                
            except Exception as e:
                print(f"Ошибка phaseCorrelate для {file}: {e}")
                continue
            
            # Расчёт комплекса метрик качества изображения
            quality_metrics = compute_quality_metrics(ref_img, curr_img)

            results.append({
                    'angle': angle,
                    'dx': dx,
                    'dy': dy,
                    "true_dx": true_dx if true_dx == 0 else -true_dx,  # Инверсия знака для ненулевых смещений
                    "true_dy": true_dy if true_dy == 0 else -true_dy,
                    'response': response,  # Максимальное значение корреляции (мера уверенности)
                    'contrast': contrast,
                    'entropy': entropy,
                    'gradient_energy': gradient_energy,
                    'mean_brightness': mean_brightness,
                    'sharpness': lap_var,
                    'snr': snr,
                    'median_brightness': median_brightness,
                    'psnr': quality_metrics['psnr'],
                    'ssim': quality_metrics['ssim'],
                    'ms_ssim': quality_metrics['ms_ssim'],
                    'vif': quality_metrics['vif'],
                    'fsim': quality_metrics['fsim'],
                })

        results.sort(key=lambda x: x['angle'])

        header = (
            "angle,dx,dy,true_dx,true_dy,response,contrast,entropy,gradient_energy,"
            "mean_brightness,sharpness,snr,median_brightness,psnr,ssim,ms_ssim,vif,fsim\n"
        )

        # Сохраняем в CSV
        output_csv = os.path.join(PARAMS_DIR, f"{root.split('\\')[-1]}.csv")
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            for r in results:
                f.write(
                    f"{r['angle']:.2f},{r['dx']:.3f},{r['dy']:.3f},{r['true_dx']:.3f},{r['true_dy']:.3f},"
                    f"{r['response']:.3f},{r['contrast']:.3f},{r['entropy']:.3f},{r['gradient_energy']:.3f},"
                    f"{r['mean_brightness']:.3f},{r['sharpness']:.3f},{r['snr']:.3f},{r['median_brightness']:.3f},"
                    f"{r['psnr']:.3f},{r['ssim']:.3f},{r['ms_ssim']:.3f},{r['vif']:.3f},{r['fsim']:.3f}\n"
                )
                
        print(f"Обработано: {os.path.basename(root)} → {len(results)} изображений, сохранено в {output_csv}")

    print("Все папки обработаны!")


if __name__ == "__main__":
    main()