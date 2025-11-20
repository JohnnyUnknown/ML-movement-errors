import pandas as pd
import numpy as np
import os
from pathlib import Path
from sys import path


pd.options.mode.use_inf_as_na = True

ANGLES_DIR = Path(path[0] + "\\angles\\parameters")
columns = [
            'angle','true_dx','true_dy',
            "0_0_dx","0_0_dy","0_0_PCE","0_0_resp","0_1_dx","0_1_dy","0_1_PCE","0_1_resp","0_2_dx","0_2_dy","0_2_PCE","0_2_resp",
            "1_0_dx","1_0_dy","1_0_PCE","1_0_resp","1_1_dx","1_1_dy","1_1_PCE","1_1_resp","1_2_dx","1_2_dy","1_2_PCE","1_2_resp",
            "2_0_dx","2_0_dy","2_0_PCE","2_0_resp","2_1_dx","2_1_dy","2_1_PCE","2_1_resp","2_2_dx","2_2_dy","2_2_PCE","2_2_resp"
        ]


dataframes = []
dx, dy = [], []


for file in os.listdir(ANGLES_DIR):
    if file.endswith('.csv'):  
        file_path = ANGLES_DIR / file
        df = pd.read_csv(file_path)
            
        df_features = df.copy()

        for i in range(len(df)):

            dx_list = df.loc[i:, ["0_0_dx","0_1_dx","0_2_dx","1_0_dx","1_1_dx","1_2_dx","2_0_dx","2_1_dx","2_2_dx"]]
            dy_list = df.loc[i:, ["0_0_dy","0_1_dy","0_2_dy","1_0_dy","1_1_dy","1_2_dy","2_0_dy","2_1_dy","2_2_dy"]]
            pce_list = df.loc[i:, ["0_0_PCE","0_1_PCE","0_2_PCE","1_0_PCE","1_1_PCE","1_2_PCE","2_0_PCE","2_1_PCE","2_2_PCE"]]
            resp_list = df.loc[i:, ["0_0_resp","0_1_resp","0_2_resp","1_0_resp","1_1_resp","1_2_resp","2_0_resp","2_1_resp","2_2_resp"]]
            
            weights = np.array(pce_list)
            # dx_est = np.average(dx_list, weights=weights)
            # dy_est = np.average(dy_list, weights=weights)
            dx_est = np.mean(dx_list)
            dy_est = np.mean(dy_list)

            dx.append(round(dx_est-100, 3))
            dy.append(round(dy_est-100, 3))

        dx, dy = np.array(dx), np.array(dy)

        # Нахождение отклонений измеренных значений от истинных
        deviation_dx = round(df_features["true_dx"].abs() - dx, 5)
        deviation_dy = round(df_features["true_dy"].abs() - dy, 5)

        # Формирование таргетных столбцов
        df_features.insert(1, "dx", dx)
        df_features.insert(2, "dy", dy)
        df_features.insert(len(df_features.columns), "deviation_dx", deviation_dx)
        df_features.insert(len(df_features.columns), "deviation_dy", deviation_dy)

        df_features.loc[df_features['true_dx'] < 0, 'deviation_dx'] *= -1
        df_features.loc[df_features['true_dy'] < 0, 'deviation_dy'] *= -1
        df_features.loc[((df_features['true_dx'] == 0) & (dx < 0)), 
                        'deviation_dx'] = df_features.loc[((df_features['true_dx'] == 0) & (dx < 0)), 
                                                            'deviation_dx'].abs()
        df_features.loc[((df_features['true_dy'] == 0) & (dy < 0)), 
                        'deviation_dy'] = df_features.loc[((df_features['true_dy'] == 0) & (dy < 0)), 
                                                            'deviation_dy'].abs()
        
        dx, dy = [], []
        dataframes.append(df_features)

if dataframes:
    all_data = pd.concat(dataframes, ignore_index=True)
else:
    all_data = pd.DataFrame(columns=columns)
    
all_data.fillna(0, inplace=True)

print(all_data)

csv_path = Path(path[0] + "\\angles\\combined_data.csv")
all_data.to_csv(csv_path, index=False, encoding='utf8')