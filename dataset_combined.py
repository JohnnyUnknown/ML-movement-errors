"""
    Формирует и сохраняет итоговый датасет с признаками и целевыми переменными.

    СТРУКТУРА ДАТАСЕТА:
        Признаки (18 столбцов):
        - Угол поворота ('angle')
        - Измеренные смещения ('dx', 'dy')
        - Истинные смещения ('true_dx', 'true_dy')
        - Отклик корреляции ('response')
        - Статистические признаки (контраст, энтропия, градиентная энергия и др.)
        - Метрики качества (PSNR, SSIM, MS-SSIM, VIF, FSIM)
        
        Целевые переменные (2 столбца):
        - 'deviation_dx': разница между истинным и измеренным смещением по X
        - 'deviation_dy': разница между истинным и измеренным смещением по Y

    ЛОГИКА ВЫЧИСЛЕНИЯ ОТКЛОНЕНИЙ:
        1. Базовое отклонение = |истинное| - |измеренное|
        2. Коррекция знака:
            - Если истинное смещение отрицательное → отклонение тоже отрицательное
            - Если истинное = 0, но измеренное отрицательное → отклонение положительное
            (компенсация ложного отрицательного смещения при нулевом истинном)
        3. Цель: обучить модель предсказывать поправку к измеренному смещению

    ПРИМЕР:
        Истинное смещение: -2.0px, Измеренное: -1.7px
        Базовое отклонение: |−2.0| − |−1.7| = 2.0 − 1.7 = +0.3
        После коррекции знака (истинное < 0): −0.3
        Интерпретация: измеренное смещение занижено на 0.3px → нужно добавить −0.3 к измеренному
"""
import pandas as pd
import os
from pathlib import Path
from sys import path


# pd.options.mode.use_inf_as_na = True

PARAMS_DIR = Path(path[0] + "\\parameters")
columns = [
            'angle','dx','dy','true_dx','true_dy','response','contrast','entropy','gradient_energy','mean_brightness',
            'sharpness','snr','median_brightness', 'psnr','ssim','ms_ssim','vif','fsim'
        ]

all_data = pd.DataFrame()

for file in os.listdir(PARAMS_DIR):
    if file.endswith('.csv'):  
        file_path = PARAMS_DIR / file
        df_features = pd.read_csv(file_path)
        
        # Проверяем, что все нужные столбцы присутствуют
        if not set(columns).issubset(df_features.columns):
            continue
        
        # Нахождение отклонений измеренных значений от истинных
        deviation_dx = round(df_features["true_dx"].abs() - df_features["dx"].abs(), 5)
        deviation_dy = round(df_features["true_dy"].abs() - df_features["dy"].abs(), 5)

        # Формирование таргетных столбцов
        df_features.insert(len(df_features.columns), "deviation_dx", deviation_dx)
        df_features.insert(len(df_features.columns), "deviation_dy", deviation_dy)

        df_features.loc[df_features['true_dx'] < 0, 'deviation_dx'] *= -1
        df_features.loc[df_features['true_dy'] < 0, 'deviation_dy'] *= -1
        df_features.loc[((df_features['true_dx'] == 0) & 
                         (df_features['dx'] < 0)), 'deviation_dx'] = df_features.loc[((df_features['true_dx'] == 0) & 
                                                                                      (df_features['dx'] < 0)), 'deviation_dx'].abs()
        df_features.loc[((df_features['true_dy'] == 0) & 
                         (df_features['dy'] < 0)), 'deviation_dy'] = df_features.loc[((df_features['true_dy'] == 0) & 
                                                                                      (df_features['dy'] < 0)), 'deviation_dy'].abs()
        
        all_data = pd.concat([all_data, df_features], ignore_index=True)

    
all_data.fillna(0, inplace=True)

print(all_data)

csv_path = Path(path[0] + "\\combined_data.csv")
all_data.to_csv(csv_path, index=False, encoding='utf8')