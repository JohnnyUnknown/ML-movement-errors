import pandas as pd
import numpy as np
import os
from pathlib import Path
from sys import path


pd.options.mode.use_inf_as_na = True

ANGLES_DIR = Path(path[0] + "\\angles\\parameters")
columns = [
            'angle','true_dx','true_dy',
            "0_0_dx","0_0_dy","0_0_resp","0_1_dx","0_1_dy","0_1_resp","0_2_dx","0_2_dy","0_2_resp",
            "1_0_dx","1_0_dy","1_0_resp","1_1_dx","1_1_dy","1_1_resp","1_2_dx","1_2_dy","1_2_resp",
            "2_0_dx","2_0_dy","2_0_resp","2_1_dx","2_1_dy","2_1_resp","2_2_dx","2_2_dy","2_2_resp"
        ]

# Признаки для модели (всё, кроме dx, dy)
# feature_columns = [col for col in columns if col not in {'delta_dx', 'delta_dy'}]

dataframes = []

for file in os.listdir(ANGLES_DIR):
    if file.endswith('.csv'):  
        file_path = ANGLES_DIR / file
        df = pd.read_csv(file_path)
        
        # Проверяем, что все нужные столбцы присутствуют
        if not set(columns).issubset(df.columns):
            continue
            
        df_features = df[columns].copy()
        
        # Нахождение отклонений измеренных значений от истинных
        deviation_dx = round(df_features["true_dx"].abs() - df_features["dx"].abs(), 5)
        deviation_dy = round(df_features["true_dy"].abs() - df_features["dy"].abs(), 5)

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

csv_path = Path(path[0] + "\\angles\\combined_data.csv")
all_data.to_csv(csv_path, index=False, encoding='utf8')