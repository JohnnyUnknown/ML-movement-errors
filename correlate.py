import os
import cv2
import numpy as np
import re
from sys import path
from skimage import measure
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import piq



ANGLES_DIR = path[0] + "\\angles_2deg\\images"
PARAMS_DIR = path[0] + "\\angles_2deg\\parameters"
img_size = (256, 256)


def compute_quality_metrics(ref_img, curr_img):
    """
    Вычисляет метрики качества изображения между ref_img и curr_img.
    Все изображения в формате uint8, BGR или grayscale.
    Возвращает словарь с PSNR, SSIM, MS-SSIM, LPIPS, VIF, FSIM
    """
    metrics = {}
    
    # Если grayscale, преобразуем к 3-канальному для LPIPS и MS-SSIM
    if len(ref_img.shape) == 2:
        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
        curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
    else:
        ref_rgb = ref_img
        curr_rgb = curr_img

    # --- PSNR ---
    metrics['psnr'] = compare_psnr(ref_img, curr_img, data_range=255)

    # --- SSIM ---
    metrics['ssim'] = compare_ssim(ref_img, curr_img, multichannel=True, data_range=255)

    # --- Подготовка тензоров для PyTorch ---
    def to_tensor(img):
        img_float = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
        return tensor

    ref_tensor = to_tensor(ref_rgb)
    curr_tensor = to_tensor(curr_rgb)

    # --- MS-SSIM ---
    metrics['ms_ssim'] = piq.multi_scale_ssim(ref_tensor, curr_tensor, data_range=1.0).item()

    # --- VIF ---
    metrics['vif'] = piq.vif_p(ref_tensor, curr_tensor, data_range=1.0).item()

    # --- FSIM ---
    metrics['fsim'] = piq.fsim(ref_tensor, curr_tensor, data_range=1.0).item()

    return metrics

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

def compute_snr(img_float):
    """Вычисляет SNR (отношение сигнал/шум) в dB"""
    mean = np.mean(img_float)
    var = np.var(img_float)
    if var == 0:
        return 0.0
    snr = 10 * np.log10(mean**2 / var)
    return float(snr)

def compute_derivatives(results):
    """Вычисляет производные метрики между соседними кадрами"""
    if len(results) < 2:
        return results

    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]

        curr["delta_dx"] = curr["dx"] - prev["dx"]
        curr["delta_dy"] = curr["dy"] - prev["dy"]
        curr["delta_response"] = curr["response"] - prev["response"]

        curr["delta_entropy"] = curr["entropy"] - prev["entropy"]
        curr["delta_gradient_energy"] = curr["gradient_energy"] - prev["gradient_energy"]
        curr["delta_sharpness"] = curr["sharpness"] - prev["sharpness"]

        curr["motion_magnitude"] = np.sqrt(curr["dx"]**2 + curr["dy"]**2)
        prev_motion_mag = np.sqrt(prev["dx"]**2 + prev["dy"]**2)
        curr["delta_motion_mag"] = curr["motion_magnitude"] - prev_motion_mag

    # Для первого кадра ставим нули
    for field in [
        "delta_dx", "delta_dy", "delta_response",
        "delta_entropy", "delta_gradient_energy",
        "delta_sharpness", "motion_magnitude",
        "delta_motion_mag"
    ]:
        results[0][field] = 0.0

    return results


def main():
    for root, dirs, files in os.walk(ANGLES_DIR):
        if root == ANGLES_DIR:
            continue

        ref_path = os.path.join(root, "0p0_0p0_0p0.jpg")
        if not os.path.isfile(ref_path):
            print(f"⚠️ Пропущена папка {root}: не найдено 0p0_0p0_0p0.jpg")
            continue

        hann = cv2.createHanningWindow(img_size, cv2.CV_32F)

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX)

        if ref_img is None:
            print(f"❌ Не удалось загрузить {ref_path}")
            continue

        results = []

        for file in files:
            img_path = os.path.join(root, file)

            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            curr_img = cv2.normalize(curr_img, None, 0, 255, cv2.NORM_MINMAX)

            if curr_img is None:
                print(f"❌ Не удалось загрузить {img_path}")
                continue

            # Убедимся, что размеры совпадают
            if curr_img.shape != ref_img.shape:
                print(f"⚠️ Размеры не совпадают: {ref_path} vs {img_path}")
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

                # Средняя яркость
                mean_brightness = float(np.mean(img_float))
                
                # Медианная яркость
                median_brightness = float(np.median(img_float))

                # Контраст (стандартное отклонение яркости)
                contrast = float(np.std(img_float))

                # Энтропия (информационная сложность)
                entropy = float(measure.shannon_entropy(curr_img))

                # Градиентная энергия (интенсивность перепадов)
                gx, gy = np.gradient(img_float)
                gradient_energy = float(np.mean(gx**2 + gy**2))

                # Резкость (через дисперсию лапласиана)
                lap_var = float(cv2.Laplacian(curr_img, cv2.CV_64F).var())
                
                snr = compute_snr(img_float)
                
            except Exception as e:
                print(f"❌ Ошибка phaseCorrelate для {file}: {e}")
                continue
            
            quality_metrics = compute_quality_metrics(ref_img, curr_img)

            results.append({
                    'filename': file,
                    'angle': angle,
                    'dx': dx,
                    'dy': dy,
                    "true_dx": true_dx if true_dx == 0 else -true_dx,
                    "true_dy": true_dy if true_dy == 0 else -true_dy,
                    'response': response,
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

        results = compute_derivatives(results)

        results.sort(key=lambda x: x['angle'])

        header = (
            "angle,dx,dy,true_dx,true_dy,response,contrast,entropy,gradient_energy,mean_brightness,median_brightness,"
            "sharpness,snr,motion_magnitude,delta_dx,delta_dy,delta_response,delta_entropy,"
            "delta_gradient_energy,delta_sharpness,delta_motion_mag,psnr,ssim,ms_ssim,vif,fsim\n"
        )

        # Сохраняем в CSV
        output_csv = os.path.join(PARAMS_DIR, f"{root.split('\\')[-1]}.csv")
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            for r in results:
                f.write(
                    f"{r['angle']:.2f},{r['dx']:.3f},{r['dy']:.3f},{r['true_dx']:.3f},{r['true_dy']:.3f},{r['response']:.3f},"
                    f"{r['contrast']:.3f},{r['entropy']:.3f},{r['gradient_energy']:.3f},"
                    f"{r['mean_brightness']:.3f},{r['median_brightness']:.3f},{r['sharpness']:.3f},{r['snr']:.3f},"
                    f"{r['motion_magnitude']:.3f},{r['delta_dx']:.3f},{r['delta_dy']:.3f},{r['delta_response']:.3f},"
                    f"{r['delta_entropy']:.3f},{r['delta_gradient_energy']:.3f},{r['delta_sharpness']:.3f},{r['delta_motion_mag']:.3f},"
                    f"{r['psnr']:.3f},{r['ssim']:.3f},{r['ms_ssim']:.3f},{r['vif']:.3f},{r['fsim']:.3f}\n"
                )
                
        print(f"✅ Обработано: {os.path.basename(root)} → {len(results)} изображений, сохранено в {output_csv}")

    print("🎉 Все папки обработаны!")

if __name__ == "__main__":
    main()