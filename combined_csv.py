import pandas as pd
import numpy as np
import os
from pathlib import Path
from sys import path


pd.options.mode.use_inf_as_na = True

ANGLES_DIR = Path(path[0] + "\\angles")
columns = [
            'angle','dx','dy','true_dx','true_dy','response','contrast','entropy','gradient_energy','mean_brightness','median_brightness',
            'sharpness','dynamic_range','snr','motion_magnitude','delta_dx','delta_dy','delta_response','delta_entropy',
            'delta_gradient_energy','delta_sharpness','delta_motion_mag','psnr','ssim','ms_ssim','vif','fsim'
        ]

# Признаки для модели (всё, кроме dx, dy)
feature_columns = [col for col in columns if col not in {'delta_dx', 'delta_dy'}]

dataframes = []

for file in os.listdir(ANGLES_DIR):
    if file.endswith('.csv'):  
        file_path = ANGLES_DIR / file
        df = pd.read_csv(file_path)
        
        # Проверяем, что все нужные столбцы присутствуют
        if not set(columns).issubset(df.columns):
            continue
            
        df_features = df[feature_columns].copy()

        # Перенос целевых столбцов в конец таблицы с отрицательным знаком
        dx = df_features.pop("true_dx")
        dy = df_features.pop("true_dy")
        df_features.insert(len(df_features.columns), "true_dx", -dx)
        df_features.insert(len(df_features.columns), "true_dy", -dy)
        
        dataframes.append(df_features)

if dataframes:
    all_data = pd.concat(dataframes, ignore_index=True)
else:
    all_data = pd.DataFrame(columns=columns)
    
all_data.fillna(0, inplace=True)

csv_path = Path(path[0] + "\\combined_data.csv")
all_data.to_csv(csv_path, index=False, encoding='utf8')