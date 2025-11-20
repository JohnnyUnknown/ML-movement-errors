import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sys import path
import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
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
    print(f"Процент отклонений > {delta}px корреляц. метода по осям:", error_percent, "%")

    return df[mask], error_percent


def get_deviation_data(all_data, clear_data):
    """Функция получает полный и отфильтрованный наборы данных. Возвращает датафрейм с ошибочными измерениями."""
    all_data_dev = all_data.merge(clear_data, how='left', indicator=True)
    emis_corr_data = all_data_dev[all_data_dev['_merge'] == 'left_only'].drop(columns=['_merge'])
    return emis_corr_data


def prediction_analysis(y_test, y_pred, test_index):
    """ Вывод количества и процента предсказаний ухудшающих значения смещений. 
        Функция считает за ошибку предсказания если:
        - у истинного и предсказанного значения разные знаки, и их сумма по модулю больше 0.5 (delta);
        - предсказанное значение в два и более раза больше истинного;
        - истинное значение == 0, а предсказанное не в диапазоне [-0.5; 0.5] (delta)."""
    
    y_pred = pd.DataFrame(y_pred, index=test_index, columns=['dev_dx', 'dev_dy'])
    out_errors = []
    coef = 2
    for i in test_index:
        pred_dev_x, true_dev_x = round(y_pred.loc[i, 'dev_dx'], 1), round(y_test.loc[i, 'deviation_dx'], 1)
        pred_dev_y, true_dev_y = round(y_pred.loc[i, 'dev_dy'], 1), round(y_test.loc[i, 'deviation_dy'], 1)

        if ((true_dev_x < 0 and pred_dev_x > 0 and (abs(true_dev_x) + abs(pred_dev_x)) > delta) or 
            (true_dev_x > 0 and pred_dev_x < 0 and (abs(true_dev_x) + abs(pred_dev_x)) > delta) or
            (true_dev_y < 0 and pred_dev_y > 0 and (abs(true_dev_y) + abs(pred_dev_y)) > delta) or 
            (true_dev_y > 0 and pred_dev_y < 0 and (abs(true_dev_y) + abs(pred_dev_y)) > delta) or
            (true_dev_x < 0 and pred_dev_x < (true_dev_x * coef)) or 
            (true_dev_x > 0 and pred_dev_x > (true_dev_x * coef)) or
            (true_dev_y < 0 and pred_dev_y < (true_dev_y * coef)) or 
            (true_dev_y > 0 and pred_dev_y > (true_dev_y * coef)) or 
            (true_dev_x == 0 and pred_dev_x > delta) or 
            (true_dev_x == 0 and pred_dev_x < -delta) or 
            (true_dev_y == 0 and pred_dev_y > delta) or 
            (true_dev_y == 0 and pred_dev_y < -delta)
            ):
            out_errors.append([i, round(y_test.loc[i, :], 2).values, round(y_pred.loc[i, :], 2).values])
    print(f"\nКоличество предсказаний, ухудшающих ошибку смещения: {len(out_errors)} из {len(test_index)}" 
          f" ({round(len(out_errors) / len(test_index) * 100, 2)} %)\n")
    print(*out_errors, sep="\n")


def bayes_opt(X_train, y_train):
    def multioutput_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    mae_scorer = make_scorer(multioutput_mae, greater_is_better=False)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 40, 300),
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


def plot_displacement_scatter(
    true_dx, true_dy,
    raw_dx, raw_dy,
    corrected_dx, corrected_dy,
    figsize=(10, 8),
    alpha=0.6,
    s=25,
    title_prefix=""
):
    """
    Визуализирует разброс dx и dy отдельно до и после коррекции.

    Создаёт 4 графика:
        [0,0] — dx до коррекции
        [0,1] — dx после коррекции
        [1,0] — dy до коррекции
        [1,1] — dy после коррекции

    Параметры:
        true_dx, true_dy: массивы истинных значений
        raw_dx, raw_dy: исходные (до ML) смещения
        corrected_dx, corrected_dy: смещения после коррекции моделью
        figsize: размер фигуры
        alpha: прозрачность точек
        s: размер маркеров
        title_prefix: (опционально) префикс для заголовков (например, "Block 5")
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex='col', sharey='row')

    # Объединяем данные для одинакового масштаба
    all_dx = np.concatenate([true_dx, raw_dx, corrected_dx])
    all_dy = np.concatenate([true_dy, raw_dy, corrected_dy])
    dx_lims = [all_dx.min() - 1, all_dx.max() + 1]
    dy_lims = [all_dy.min() - 1, all_dy.max() + 1]

    # --- dx: до и после ---
    axes[0, 0].scatter(true_dx, raw_dx, c='lightcoral', alpha=alpha, s=s, label='Raw')
    axes[0, 0].plot(dx_lims, dx_lims, 'k--', linewidth=1, alpha=0.7)
    axes[0, 0].set_xlim(dx_lims)
    axes[0, 0].set_ylim(dx_lims)
    axes[0, 0].set_xlabel('True dx')
    axes[0, 0].set_ylabel('Predicted dx')
    axes[0, 0].set_title(f"{title_prefix}dx — Before correction".strip())
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].scatter(true_dx, corrected_dx, c='mediumseagreen', alpha=alpha, s=s, label='Corrected')
    axes[0, 1].plot(dx_lims, dx_lims, 'k--', linewidth=1, alpha=0.7)
    axes[0, 1].set_xlim(dx_lims)
    axes[0, 1].set_ylim(dx_lims)
    axes[0, 1].set_xlabel('True dx')
    axes[0, 1].set_ylabel('Predicted dx')
    axes[0, 1].set_title(f"{title_prefix}dx — After correction".strip())
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()

    # --- dy: до и после ---
    axes[1, 0].scatter(true_dy, raw_dy, c='lightcoral', alpha=alpha, s=s, label='Raw')
    axes[1, 0].plot(dy_lims, dy_lims, 'k--', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlim(dy_lims)
    axes[1, 0].set_ylim(dy_lims)
    axes[1, 0].set_xlabel('True dy')
    axes[1, 0].set_ylabel('Predicted dy')
    axes[1, 0].set_title(f"{title_prefix}dy — Before correction".strip())
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend()

    axes[1, 1].scatter(true_dy, corrected_dy, c='mediumseagreen', alpha=alpha, s=s, label='Corrected')
    axes[1, 1].plot(dy_lims, dy_lims, 'k--', linewidth=1, alpha=0.7)
    axes[1, 1].set_xlim(dy_lims)
    axes[1, 1].set_ylim(dy_lims)
    axes[1, 1].set_xlabel('True dy')
    axes[1, 1].set_ylabel('Predicted dy')
    axes[1, 1].set_title(f"{title_prefix}dy — After correction".strip())
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_error_vs_angle(
    angle,
    true_dx, true_dy,
    raw_dx, raw_dy,
    corrected_dx, corrected_dy,
    figsize=(10, 8),
    alpha=0.6,
    s=25,
    angle_unit='degrees'  # или 'radians'
):
    """
    Строит ошибку (predicted - true) по dx и dy в зависимости от угла поворота.
    
    4 графика:
        [0,0] — ошибка dx до коррекции
        [0,1] — ошибка dx после коррекции
        [1,0] — ошибка dy до коррекции
        [1,1] — ошибка dy после коррекции

    Параметры:
        angle: массив углов (длина n)
        true_dx, true_dy: истинные смещения
        raw_dx, raw_dy: смещения до коррекции
        corrected_dx, corrected_dy: смещения после коррекции
        figsize: размер фигуры
        alpha: прозрачность точек
        s: размер маркеров
        angle_unit: 'degrees' или 'radians' — влияет на метки (опционально)
    """
    # Вычисляем ошибки
    error_raw_dx = raw_dx - true_dx
    error_raw_dy = raw_dy - true_dy
    error_corr_dx = corrected_dx - true_dx
    error_corr_dy = corrected_dy - true_dy

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Объединяем все ошибки для одинакового масштаба по Y
    all_errors_dx = np.concatenate([error_raw_dx, error_corr_dx])
    all_errors_dy = np.concatenate([error_raw_dy, error_corr_dy])
    
    y_lim_dx = np.max(np.abs(all_errors_dx)) * 1.1
    y_lim_dy = np.max(np.abs(all_errors_dy)) * 1.1

    # --- dx: до и после ---
    axes[0, 0].scatter(angle, error_raw_dx, c='lightcoral', alpha=alpha, s=s, label='Raw error')
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    axes[0, 0].set_ylabel('Error in dx\n(predicted – true)')
    axes[0, 0].set_title('dx error — Before correction')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].set_ylim(-y_lim_dx, y_lim_dx)
    axes[0, 0].legend()

    axes[0, 1].scatter(angle, error_corr_dx, c='mediumseagreen', alpha=alpha, s=s, label='Corrected error')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    axes[0, 1].set_ylabel('Error in dx\n(predicted – true)')
    axes[0, 1].set_title('dx error — After correction')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].set_ylim(-y_lim_dx, y_lim_dx)
    axes[0, 1].legend()

    # --- dy: до и после ---
    axes[1, 0].scatter(angle, error_raw_dy, c='lightcoral', alpha=alpha, s=s, label='Raw error')
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    axes[1, 0].set_xlabel(f'Angle ({angle_unit})')
    axes[1, 0].set_ylabel('Error in dy\n(predicted – true)')
    axes[1, 0].set_title('dy error — Before correction')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].set_ylim(-y_lim_dy, y_lim_dy)
    axes[1, 0].legend()

    axes[1, 1].scatter(angle, error_corr_dy, c='mediumseagreen', alpha=alpha, s=s, label='Corrected error')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    axes[1, 1].set_xlabel(f'Angle ({angle_unit})')
    axes[1, 1].set_ylabel('Error in dy\n(predicted – true)')
    axes[1, 1].set_title('dy error — After correction')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].set_ylim(-y_lim_dy, y_lim_dy)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()




path_dir = Path(path[0])
all_data = pd.read_csv((path_dir / "angles_2\\combined_data.csv"))

# # Данные для работы с тайлами кросс-корреляции
# from EDA_peaks import get_selected_params
# all_data = pd.read_csv((path_dir / "angles\\combined_data.csv"))

delta = 0.5

X, y = get_selected_params(method=None, num_of_params=8, show_img=False, save_img=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_index = list(y_test.index)


# # Оптимизация гиперпараметров моделей
# bayes_opt(X, y)

# for num in range(50, 150, 10):
#     model = RandomForestRegressor(n_estimators=num)
#     multi_model = MultiOutputRegressor(model)
#     scores_mae = cross_val_score(multi_model, X, y, cv=5, scoring='neg_mean_absolute_error')
#     scores_mse = cross_val_score(multi_model, X, y, cv=5, scoring='neg_mean_squared_error')
#     print(f"{num}: Avg MAE (dx & dy) = {-scores_mae.mean():.4f} ± {scores_mae.std():.4f}")
#     print(f"{num}: Avg MSE (dx & dy) = {-scores_mse.mean():.4f} ± {scores_mse.std():.4f}")


model = RandomForestRegressor(n_estimators=80, min_samples_split=3, random_state=42)
# model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='MAE')
# model = XGBRegressor(n_estimators=70, random_state=42, n_jobs=-1)
# model = XGBRegressor(n_estimators=105, max_depth=8, learning_rate=0.0156, 
#                       random_state=42, subsample=0.9937, colsample_bytree=0.6544)


multi_model = MultiOutputRegressor(model)
# multi_model = load("model_2.joblib")


multi_model.fit(X_train, y_train)
y_pred = multi_model.predict(X_test)
# print(cross_val_score(multi_model, X, y, scoring='neg_mean_absolute_error'))


# # Сохранение модели (обучение на полном наборе данных)
# multi_model.fit(np.array(X), y)
# dump(multi_model, 'model_2.joblib')


# Сравнение истинных отклонений с измеренными до правки и после 
corr = X_test.loc[:, ['dx', 'dy']]
true_shift = all_data.loc[test_index, ['true_dx', 'true_dy']]
mae_corr = mean_absolute_error(true_shift, corr)
print(round(y_test.describe().transpose(), 4))
print(f"MAE до коррекции: {mae_corr:.5f} пикселей\n")

y_test_ML = y_test - y_pred
print(round(y_test_ML.describe().transpose(), 4))
corr_ML = corr + y_pred
mae = mean_absolute_error(true_shift, corr_ML)
print(f"Средняя MAE после коррекции: {mae:.5f} пикселей\n")


# Добавление к тестовым данным столбцов с истинными смещениями для анализа отклонений
data = X_test.copy()
data.insert(3, "true_dx", all_data.loc[test_index, "true_dx"])
data.insert(4, "true_dy", all_data.loc[test_index, "true_dy"])


# Получение всех отфильтрованных измерений для анализа без ML
print("Исходные данные:")
emis_data, err_emis = measurement_deviations(data, delta)
emiss_err = get_deviation_data(all_data=data, clear_data=emis_data)


# Добавление поправки к измеренным смещениям
data["dx"] = data["dx"] + y_pred[:, 0]
data["dy"] = data["dy"] + y_pred[:, 1]

# Получение всех отфильтрованных измерений для анализа с поправками ML
print("Данные после правки ML:")
emis_data_ML, err_emis_ML = measurement_deviations(data, delta)
emiss_err_ML = get_deviation_data(all_data=data, clear_data=emis_data_ML)


prediction_analysis(y_test, y_pred, test_index)


plot_displacement_scatter(true_shift["true_dx"], true_shift['true_dy'], 
                          corr["dx"], corr["dy"],
                          corr_ML["dx"], corr_ML["dy"])
plot_error_vs_angle(all_data.loc[test_index, "angle"],
                    true_shift["true_dx"], true_shift['true_dy'], 
                    corr["dx"], corr["dy"],
                    corr_ML["dx"], corr_ML["dy"])


# Отрисовка смещений с ML и без  
limit = [-25, 25]
fig = plt.figure(figsize=(14, 6))
axs = fig.subplots(1, 5)
axs[0].scatter(emis_data["dx"].values, emis_data["dy"].values)
axs[0].set(title=f"Corr emission", xlim=limit, ylim=limit)
axs[1].scatter(emiss_err["dx"].values, emiss_err["dy"].values)
axs[1].set(title=f"Errors {err_emis}%", xlim=limit, ylim=limit)
axs[2].scatter(emis_data_ML["dx"].values, emis_data_ML["dy"].values)
axs[2].set(title=f"Corr with ML", xlim=limit, ylim=limit)
axs[3].scatter(emiss_err_ML["dx"].values, emiss_err_ML["dy"].values)
axs[3].set(title=f"Errors with ML {err_emis_ML}%", xlim=limit, ylim=limit) 
axs[4].scatter(all_data.loc[test_index, "true_dx"].values, all_data.loc[test_index, "true_dy"].values)
axs[4].set(title=f"True", xlim=limit, ylim=limit)
# plt.savefig((path_dir / f"graphics\\ML_SBS_{len(SBS_analysis())}.jpg"), dpi=800)
plt.show()

