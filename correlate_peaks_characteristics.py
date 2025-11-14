import os
import cv2
import numpy as np
import pandas as pd
import re
from sys import path


ANGLES_DIR = path[0] + "\\angles\\images"
PARAMS_DIR = path[0] + "\\angles\\parameters"
img_size = (600, 600)
grid_size = (3, 3)



def parse_shift_angle_from_filename(filename):
    name, _ = os.path.splitext(filename)
    parts = name.split('_')
    
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


def calculate_phase_correlation(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Same shape for obj")
    
    hann_window = np.outer(
        np.hanning(img1.shape[0]),
        np.hanning(img1.shape[1])
    )
    img1_float = img1.astype(np.float32)*hann_window
    img2_float = img2.astype(np.float32)*hann_window
    
    fft1 = np.fft.fft2(img1_float)
    fft2 = np.fft.fft2(img2_float)
    
    cross_spectrum = np.conjugate(fft2) * fft1
    
    cross_spectrum_normalized = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)
    bfft = np.fft.ifft2(cross_spectrum_normalized)
    correlation = np.abs(np.fft.ifftshift(bfft))
    return correlation

def split_images_into_tiles(img1, img2, grid_size=(3, 3)):
    h, w = img1.shape[:2]
    
    tile_h = h // grid_size[0]
    tile_w = w // grid_size[1]
    
    tiles1 = []
    tiles2 = []
    
    for i in range(grid_size[0]):
        row_tiles1 = []
        row_tiles2 = []
        for j in range(grid_size[1]):
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < grid_size[0] - 1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j < grid_size[1] - 1 else w
            
            tile1 = img1[y_start:y_end, x_start:x_end]
            tile2 = img2[y_start:y_end, x_start:x_end]
            
            row_tiles1.append(tile1)
            row_tiles2.append(tile2)
        
        tiles1.append(row_tiles1)
        tiles2.append(row_tiles2)
    
    return tiles1, tiles2, (tile_h, tile_w)


def find_shift(h):
    ind = np.unravel_index(np.argmax(h, axis=None), h.shape)
    return ind

def weighted_centroid(correlation, peak_loc, window_size=5): #ret x,y
    rows, cols = correlation.shape
    y_peak, x_peak = peak_loc
    
    half_window = window_size // 2
    y_start = max(0, y_peak - half_window)
    y_end = min(rows, y_peak + half_window + 1)
    x_start = max(0, x_peak - half_window)
    x_end = min(cols, x_peak + half_window + 1)
    
    window = correlation[y_start:y_end, x_start:x_end]
    
    y_coords, x_coords = np.mgrid[y_start:y_end, x_start:x_end]
    
    total_weight = np.sum(window)
    
    if total_weight > 0:
        refined_y = np.sum(y_coords * window) / total_weight
        refined_x = np.sum(x_coords * window) / total_weight
    else:
        refined_y, refined_x = y_peak, x_peak
    
    return refined_x, refined_y

def calculate_correlation_for_tiles(tiles1, tiles2):
    grid_rows = len(tiles1)
    grid_cols = len(tiles1[0])
    
    # Сохраняем все сдвиги для каждого tile
    all_shifts = []
    sample_tile1 = tiles1[0][0]
    sample_tile2 = tiles2[0][0]
    sample_correlation = calculate_phase_correlation(sample_tile1, sample_tile2)
    # print("Тип:", sample_correlation.dtype)
    # print("Минимум:", sample_correlation.min())
    # print("Максимум:", sample_correlation.max())
    # print("Среднее:", sample_correlation.mean())
    # print("Медиана фона (без пика):", np.median(sample_correlation))
    tile_corr_h, tile_corr_w = sample_correlation.shape
    combined_correlation = np.zeros((grid_rows * tile_corr_h, grid_cols * tile_corr_w))
    
    for i in range(grid_rows):
        row_shifts = []
        for j in range(grid_cols):
            tile1 = tiles1[i][j]
            tile2 = tiles2[i][j]
            
            correlation = calculate_phase_correlation(tile1, tile2)
            phase_corelation_shift = find_shift(correlation)
            refined_peak = weighted_centroid(correlation, phase_corelation_shift, 5)
            
            # Сохраняем сдвиги для текущего tile
            tile_shifts = {
                'tile_coords': (i, j),
                'custom_shift': refined_peak,
                'peak_location': phase_corelation_shift,
                'response': np.max(correlation),
                'PCE': calculate_pce(correlation)
            }
            row_shifts.append(tile_shifts)
            y_start = i * tile_corr_h
            y_end = (i + 1) * tile_corr_h
            x_start = j * tile_corr_w
            x_end = (j + 1) * tile_corr_w
            
            combined_correlation[y_start:y_end, x_start:x_end] = correlation
        
        all_shifts.append(row_shifts)
    return combined_correlation, all_shifts

def calculate_pce(correlation):
    # Находим пик
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_value = correlation[peak_idx]
    
    # Создаём маску без пика (например, убираем окно 5x5)
    mask = np.ones_like(correlation, dtype=bool)
    h, w = correlation.shape
    r, c = peak_idx
    # Обрезаем окно вокруг пика
    r_min, r_max = max(0, r - 2), min(h, r + 3)
    c_min, c_max = max(0, c - 2), min(w, c + 3)
    mask[r_min:r_max, c_min:c_max] = False
    
    # Фон — всё кроме окрестности пика
    background = correlation[mask]
    
    # PCE = (peak^2) / (средняя энергия фона)
    pce = (peak_value ** 2) / (np.mean(background ** 2) + 1e-10)    
    # print(f"Пик: {peak_value:.6f}")
    # print(f"Энергия фона: {np.mean(background ** 2):.2e}")
    # print(f"PCE: {pce:.2e}")
    return pce

def extract_tails_info(shifts):
    tails_info = {}
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            num = shifts[i][j]['tile_coords']
            tail_name = f'{num[0]}_{num[1]}'
            tails_info[tail_name + "_dx"] = shifts[i][j]['custom_shift'][0]
            tails_info[tail_name + "_dy"] = shifts[i][j]['custom_shift'][1]
            tails_info[tail_name + "_PCE"] = shifts[i][j]['PCE']
            tails_info[tail_name + "_resp"] = shifts[i][j]['response']
    return tails_info


def main():
    os.makedirs(PARAMS_DIR, exist_ok=True)
    for root, dirs, files in os.walk(ANGLES_DIR):
        if root == ANGLES_DIR:
            continue

        ref_path = os.path.join(root, "0p0_0p0_0p0.jpg")
        if not os.path.isfile(ref_path):
            print(f"⚠️ Пропущена папка {root}: не найдено 0p0_0p0_0p0.jpg")
            continue

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX)
        # ref_img_float = ref_img.astype(np.float64) + np.random.normal(0, 1.0, ref_img.shape)
        # ref_img = np.clip(ref_img_float, 0, 255).astype(np.uint8)

        # if ref_img is None:
        #     print(f"❌ Не удалось загрузить {ref_path}")
        #     continue

        results = []

        for file in files:
            img_path = os.path.join(root, file)
            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if curr_img is None:
                print(f"❌ Не удалось загрузить {img_path}")
                continue

            curr_img = cv2.normalize(curr_img, None, 0, 255, cv2.NORM_MINMAX)
            # curr_img_float = curr_img.astype(np.float64) + np.random.normal(0, 1.0, curr_img.shape)
            # curr_img = np.clip(curr_img_float, 0, 255).astype(np.uint8)

            try:
                tile1, tile2, _ = split_images_into_tiles(ref_img, curr_img, grid_size)
                _, shifts = calculate_correlation_for_tiles(tile1, tile2)
                
                true_dx, true_dy, angle = parse_shift_angle_from_filename(file)
                    
            except Exception as e:
                print(f"❌ Ошибка phaseCorrelate для {file}: {e}")
                continue

            res = {
                    'angle': angle,
                    "true_dx": true_dx if true_dx == 0 else -true_dx,
                    "true_dy": true_dy if true_dy == 0 else -true_dy,
                }
            res |= extract_tails_info(shifts)

            results.append(res)
            # print(results)

        results.sort(key=lambda x: x['angle'])

        header = (
            "angle,true_dx,true_dy,"
            "0_0_dx,0_0_dy,0_0_PCE,0_0_resp,0_1_dx,0_1_dy,0_1_PCE,0_1_resp,0_2_dx,0_2_dy,0_2_PCE,0_2_resp,"
            "1_0_dx,1_0_dy,1_0_PCE,1_0_resp,1_1_dx,1_1_dy,1_1_PCE,1_1_resp,1_2_dx,1_2_dy,1_2_PCE,1_2_resp,"
            "2_0_dx,2_0_dy,2_0_PCE,2_0_resp,2_1_dx,2_1_dy,2_1_PCE,2_1_resp,2_2_dx,2_2_dy,2_2_PCE,2_2_resp\n"
        )

        # Сохраняем в CSV
        output_csv = os.path.join(PARAMS_DIR, f"{root.split('\\')[-1]}.csv")
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            for r in results:
                f.write(
                    f"{r['angle']:.2f},{r['true_dx']:.3f},{r['true_dy']:.3f},"
                    f"{r['0_0_dx']:.3f},{r['0_0_dy']:.3f},{r['0_0_PCE']:.3f},{r['0_0_resp']:.6f},"
                    f"{r['0_1_dx']:.3f},{r['0_1_dy']:.3f},{r['0_1_PCE']:.3f},{r['0_1_resp']:.6f},"
                    f"{r['0_2_dx']:.3f},{r['0_2_dy']:.3f},{r['0_2_PCE']:.3f},{r['0_2_resp']:.6f},"
                    f"{r['1_0_dx']:.3f},{r['1_0_dy']:.3f},{r['1_0_PCE']:.3f},{r['1_0_resp']:.6f},"
                    f"{r['1_1_dx']:.3f},{r['1_1_dy']:.3f},{r['1_1_PCE']:.3f},{r['1_1_resp']:.6f},"
                    f"{r['1_2_dx']:.3f},{r['1_2_dy']:.3f},{r['1_2_PCE']:.3f},{r['1_2_resp']:.6f},"
                    f"{r['2_0_dx']:.3f},{r['2_0_dy']:.3f},{r['2_0_PCE']:.3f},{r['2_0_resp']:.6f},"
                    f"{r['2_1_dx']:.3f},{r['2_1_dy']:.3f},{r['2_1_PCE']:.3f},{r['2_1_resp']:.6f},"
                    f"{r['2_2_dx']:.3f},{r['2_2_dy']:.3f},{r['2_2_PCE']:.3f},{r['2_2_resp']:.6f}\n"
                )
                
        print(f"✅ Обработано: {os.path.basename(root)} → {len(results)} изображений, сохранено в {output_csv}")

    print("🎉 Все папки обработаны!")

if __name__ == "__main__":
    main()