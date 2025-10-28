import pandas as pd
import numpy as np
import os
from pathlib import Path
from sys import path


pd.options.mode.use_inf_as_na = True

ANGLES_DIR = Path(path[0] + "\\angles\\parameters")
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

        # # Изменение знака из-за особенностей метода cv2.phaseCorrelate
        # df_features["true_dx"] = df_features["true_dx"] * (-1)
        # df_features["true_dy"] = df_features["true_dy"] * (-1)
        
        # Нахождение отклонений измеренных значений от истинных
        deviation_dx = round(df_features["true_dx"].abs() - df_features["dx"].abs(), 3)
        deviation_dy = round(df_features["true_dy"].abs() - df_features["dy"].abs(), 3)

        # Формирование таргетных столбцов
        df_features.insert(len(df_features.columns), "deviation_dx", deviation_dx)
        df_features.insert(len(df_features.columns), "deviation_dy", deviation_dy)

        df_features.loc[df_features['true_dx'] < 0, 'deviation_dx'] *= -1
        df_features.loc[df_features['true_dy'] < 0, 'deviation_dy'] *= -1
        df_features.loc[((df_features['true_dx'] == 0) & (df_features['dx'] < 0)), 
                       'deviation_dx'] = df_features.loc[((df_features['true_dx'] == 0) & (df_features['dx'] < 0)), 
                                                            'deviation_dx'].abs()
        df_features.loc[((df_features['true_dy'] == 0) & (df_features['dy'] < 0)), 
                        'deviation_dy'] = df_features.loc[((df_features['true_dy'] == 0) & (df_features['dy'] < 0)), 
                                                            'deviation_dy'].abs()
        
        dataframes.append(df_features)

if dataframes:
    all_data = pd.concat(dataframes, ignore_index=True)
else:
    all_data = pd.DataFrame(columns=columns)
    
all_data.fillna(0, inplace=True)

print(all_data)

csv_path = Path(path[0] + "\\combined_data_shift.csv")
all_data.to_csv(csv_path, index=False, encoding='utf8')