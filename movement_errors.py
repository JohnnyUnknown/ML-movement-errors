import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.neural_network import MLPRegressor
from sys import path
import optuna
from xgboost import XGBRegressor
from EDA import get_selected_params


def measurement_deviations(all_data, delta = 1):
    """Функция отсеивает выбросы, сравнивая найденные значения (dx, dy) с (true_dx, true_dy).
        Возвращает новый DataFrame состоящий только из полей с разницей найденных и истинных
        значений в диапазоне [true_dx +/- delta] (для dy аналогично)."""
    df = all_data.copy()
    
    low_dx = df['true_dx'] - delta
    high_dx = df['true_dx'] + delta

    low_dy = df['true_dy'] - delta
    high_dy = df['true_dy'] + delta

    mask = (
        (df['dx'] >= low_dx) & (df['dx'] <= high_dx) &
        (df['dy'] >= low_dy) & (df['dy'] <= high_dy)
    )
    error_percent = round((1 - df[mask].shape[0] / len(all_data)) * 100, 2)
    print("Процент отклонений > 1px корреляц. метода по осям:", error_percent, "%")

    return df[mask], error_percent


def get_deviation_data(all_data, clear_data):
    """Функция получает полный и отфильтрованный наборы данных. Возвращает датафрейм с ошибочными измерениями."""
    all_data_dev = all_data.merge(clear_data, how='left', indicator=True)
    emis_corr_data = all_data_dev[all_data_dev['_merge'] == 'left_only'].drop(columns=['_merge'])
    return emis_corr_data


def bayes_opt(X_train, y_train):
    def multioutput_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    mae_scorer = make_scorer(multioutput_mae, greater_is_better=False)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        model = MultiOutputRegressor(XGBRegressor(**params))
        scores = -cross_val_score(model, X_train, y_train, cv=3, scoring=mae_scorer, n_jobs=-1)
        return scores.mean()

    print("\nЗапуск Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Лучшие параметры (Optuna):")
    print(study.best_params)
    print(f"Лучший MAE: {study.best_value:.3f} px")



path_dir = Path(path[0])
all_data = pd.read_csv((path_dir / "angles_2deg\\combined_data_2deg.csv"))
delta = 0.5

X, y = get_selected_params(method="AVG", num_of_params=7, show_img=False, save_img=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
test_index = list(y_test.index)


# # Оптимизация гиперпараметров моделей
# bayes_opt(X.loc[:, params], y)

# for num in range(50, 150, 10):
#     model = RandomForestRegressor(n_estimators=num)
#     multi_model = MultiOutputRegressor(model)
#     scores_mae = cross_val_score(multi_model, X.loc[:, params], y, cv=3, scoring='neg_mean_absolute_error')
#     scores_mse = cross_val_score(multi_model, X.loc[:, params], y, cv=3, scoring='neg_mean_squared_error')
#     print(f"{num}: Avg MAE (dx & dy) = {-scores_mae.mean():.4f} ± {scores_mae.std():.4f}")
#     print(f"{num}: Avg MSE (dx & dy) = {-scores_mse.mean():.4f} ± {scores_mse.std():.4f}")


model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# model = XGBRegressor(n_estimators=342, max_depth=7, learning_rate=0.097, 
#                       random_state=42, subsample=0.94, colsample_bytree=0.64)

multi_model = MultiOutputRegressor(model)
# multi_model = load("model_2.joblib")


multi_model.fit(X_train, y_train)
y_pred = multi_model.predict(X_test)

# # Сохранение модели
# multi_model.fit(np.array(X), y)
# dump(multi_model, 'model_2.joblib')


# Сравнение истииных отклонений с измеренными до правки и после 
corr = X_test.loc[:, ['dx', 'dy']]
true_shift = all_data.loc[test_index, ['true_dx', 'true_dy']]
mae_corr = mean_absolute_error(true_shift, corr)
print(f"MAE до коррекции: {mae_corr:.5f} пикселей")
mae = mean_absolute_error(true_shift, corr + y_pred, multioutput='uniform_average')
print(f"Средняя MAE после коррекции: {mae:.3f} пикселей")


# Добавление к тестовым данным столбцов с истинными смещениями для анализа отклонений
data = X_test.copy()
data.insert(3, "true_dx", all_data.loc[test_index, "true_dx"])
data.insert(4, "true_dy", all_data.loc[test_index, "true_dy"])


# Получение всех отфильтрованных измерений для анализа без ML
emis_data, err_emis = measurement_deviations(data, delta)
emiss_err = get_deviation_data(all_data=data, clear_data=emis_data)


# Добавление поправки к измеренным смещениям
data["dx"] = data["dx"] + y_pred[:, 0]
data["dy"] = data["dy"] + y_pred[:, 1]

# Получение всех отфильтрованных измерений для анализа с поправками ML
emis_data_ML, err_emis_ML = measurement_deviations(data, delta)
emiss_err_ML = get_deviation_data(all_data=data, clear_data=emis_data_ML)


# Вывод предсказаний ухудшающих значения смещений
y_pred = pd.DataFrame(y_pred, index=test_index, columns=['dev_dx', 'dev_dy'])
out_errors = []
coef = 2
for i in test_index:
    pred_dev_x, true_dev_x = round(y_pred.loc[i, 'dev_dx'], 1), round(y_test.loc[i, 'deviation_dx'], 1)
    pred_dev_y, true_dev_y = round(y_pred.loc[i, 'dev_dy'], 1), round(y_test.loc[i, 'deviation_dy'], 1)

    if ((true_dev_x < 0 and pred_dev_x > 0) or 
        (true_dev_x > 0 and pred_dev_x < 0) or
        (true_dev_y < 0 and pred_dev_y > 0) or 
        (true_dev_y > 0 and pred_dev_y < 0) or
        (true_dev_x < 0 and pred_dev_x < true_dev_x * coef) or 
        (true_dev_x > 0 and pred_dev_x > true_dev_x * coef) or
        (true_dev_y < 0 and pred_dev_y < true_dev_y * coef) or 
        (true_dev_y > 0 and pred_dev_y > true_dev_y * coef) or 
        (true_dev_x == 0 and pred_dev_x >= delta) or 
        (true_dev_y == 0 and pred_dev_y >= delta) or 
        (true_dev_x == 0 and pred_dev_x <= -delta) or 
        (true_dev_y == 0 and pred_dev_y <= -delta)
        ):
        out_errors.append([i, round(y_test.loc[i, :], 1).values, round(y_pred.loc[i, :], 1).values])
# print(*out_errors, sep="\n")
print("\nКоличество предсказаний, ухудшающих ошибку смещения:", len(out_errors), 
      "\nПроцент плохих предсказаний:", len(out_errors) / len(test_index))


# Отрисовка смещений с ML и без  
fig = plt.figure(figsize=(14, 6))
axs = fig.subplots(1, 5)
axs[0].scatter(emis_data["dx"].values, emis_data["dy"].values)
axs[0].set(title=f"Corr emission", xlim=[-20, 20], ylim=[20, -20])
axs[1].scatter(emiss_err["dx"].values, emiss_err["dy"].values)
axs[1].set(title=f"Errors {err_emis}%", xlim=[-20, 20], ylim=[20, -20])
axs[2].scatter(emis_data_ML["dx"].values, emis_data_ML["dy"].values)
axs[2].set(title=f"Corr with ML", xlim=[-20, 20], ylim=[20, -20])
axs[3].scatter(emiss_err_ML["dx"].values, emiss_err_ML["dy"].values)
axs[3].set(title=f"Errors with ML {err_emis_ML}%", xlim=[-20, 20], ylim=[20, -20])  # , xlim=[-50, 50], ylim=[50, -50]
axs[4].scatter(all_data.loc[test_index, "true_dx"].values, all_data.loc[test_index, "true_dy"].values)
axs[4].set(title=f"True", xlim=[-20, 20], ylim=[-20, 20])
# plt.savefig((path_dir / f"graphics\\ML_SBS_{len(SBS_analysis())}.jpg"), dpi=800)
plt.show()

