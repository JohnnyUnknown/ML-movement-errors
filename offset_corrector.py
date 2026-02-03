"""
Скрипт для коррекции систематических ошибок измерения смещений методом фазовой корреляции
с помощью машинного обучения.

Фазовая корреляция отлично работает для небольших смещений, но при поворотах изображения
возникает систематическая ошибка — измеренное смещение отличается от истинного.
Модель учится предсказывать эту ошибку и компенсировать её, повышая точность измерений.

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ:
1. Загружает датасет с измеренными смещениями (dx, dy) и истинными значениями
2. Обучает модель предсказывать поправку к измеренным смещениям
3. Оценивает, насколько коррекция улучшает точность измерений
4. Визуализирует результаты до и после применения модели
"""

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from joblib import dump, load
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from selection_of_params import get_selected_params
from sys import path
from time import perf_counter
from xgboost import XGBRegressor


def measurement_deviations(all_data, delta=1):
    """
    Отфильтровывает "плохие" измерения, оставляем только те измерения, где отклонение ≤ delta 
    пикселей по обеим осям.
    
    return:
      df[mask]: Отфильтрованный датафрейм с "хорошими" измерениями
      error_percent: Процент отброшенных измерений (чем меньше — тем лучше работает фазовая корреляция)
    """
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
    print(f"Процент найденных смещений с отклонением > {delta}px от истины (по x или y):", error_percent, "%")

    return df[mask], error_percent


def get_deviation_data(all_data, clear_data):
    """
    Находит именно те измерения, которые были отброшены функцией measurement_deviations().
    """
    all_data_dev = all_data.merge(clear_data, how='left', indicator=True)
    emis_corr_data = all_data_dev[all_data_dev['_merge'] == 'left_only'].drop(columns=['_merge'])
    return emis_corr_data


def prediction_analysis(y_test, y_pred, test_index, delta=0.5, print_err=True):
    """
    Выявляет предсказания модели, которые потенциально ухудшают точность измерения смещения.
    
    Логика определения "плохого" предсказания:
      • Смена знака + значительная амплитуда (> 0.5px суммарно)
      • Перекорректировка в 2+ раза относительно истинной ошибки
      • Значительная поправка при нулевой истинной ошибке (> ±0.5px)
    """
    # Преобразуем предсказания в удобный формат для поэлементного доступа
    y_pred = pd.DataFrame(y_pred, index=test_index, columns=['dev_dx', 'dev_dy'])
    out_errors = []
    
    # Параметры чувствительности анализа
    coef = 2    # порог "перекорректировки": поправка > 2× истинной ошибки
    
    for i in test_index:
        # Округляем до 0.1px для устойчивости к мелким флуктуациям
        pred_dev_x, true_dev_x = round(y_pred.loc[i, 'dev_dx'], 1), round(y_test.loc[i, 'deviation_dx'], 1)
        pred_dev_y, true_dev_y = round(y_pred.loc[i, 'dev_dy'], 1), round(y_test.loc[i, 'deviation_dy'], 1)

        # Проверяем 3 типа проблемных предсказаний:
        
        # 1. СМЕНА ЗНАКА с существенной амплитудой
        # Пример: истинная ошибка -0.7px (нужно вычесть), модель предложила +0.6px (прибавить)
        # Суммарный эффект: 0.7 + 0.6 = 1.3px — почти гарантированное ухудшение
        if ((true_dev_x < 0 and pred_dev_x > 0 and (abs(true_dev_x) + abs(pred_dev_x)) > delta) or 
            (true_dev_x > 0 and pred_dev_x < 0 and (abs(true_dev_x) + abs(pred_dev_x)) > delta) or
            (true_dev_y < 0 and pred_dev_y > 0 and (abs(true_dev_y) + abs(pred_dev_y)) > delta) or 
            (true_dev_y > 0 and pred_dev_y < 0 and (abs(true_dev_y) + abs(pred_dev_y)) > delta) or
            
            # 2. ПЕРЕКОРРЕКТИРОВКА (более чем в 2 раза)
            # Пример: истинная ошибка +0.4px, модель предложила +0.9px → избыточная поправка
            (true_dev_x < 0 and pred_dev_x < (true_dev_x * coef)) or 
            (true_dev_x > 0 and pred_dev_x > (true_dev_x * coef)) or
            (true_dev_y < 0 and pred_dev_y < (true_dev_y * coef)) or 
            (true_dev_y > 0 and pred_dev_y > (true_dev_y * coef)) or 
            
            # 3. ЛОЖНОЕ СМЕЩЕНИЕ при нулевой истинной ошибке
            # Пример: объект не двигался (ошибка=0), но модель "нашла" смещение ±0.6px
            (true_dev_x == 0 and pred_dev_x > delta) or 
            (true_dev_x == 0 and pred_dev_x < -delta) or 
            (true_dev_y == 0 and pred_dev_y > delta) or 
            (true_dev_y == 0 and pred_dev_y < -delta)
            ):
            
            # Сохраняем проблемный случай для анализа
            out_errors.append([
                i,
                (f"Истинные ошибки: {round(y_test.loc[i, :], 3).values}, "
                 f"Предсказанные поправки: {round(y_pred.loc[i, :], 3).values}")
            ])
    
    # Выводим статистику по проблемным предсказаниям
    print(f"\nПроблемных предсказаний: {len(out_errors)} из {len(test_index)}" 
          f" ({len(out_errors) / len(test_index) * 100:.2f} %)\n")
    
    # Детали первых 10 проблемных случаев
    if print_err and out_errors:
        for err in out_errors[:10]:
            print(f"  [{err[0]}] {err[1]}")
        if len(out_errors) > 10:
            print(f"  ... ещё {len(out_errors) - 10} случаев")
    
    return out_errors


def bayes_opt(X_train, y_train):
    """
    Автоматический подбор лучших параметров для XGBoost через Optuna.
    
    Подбираемые параметры:
      - n_estimators: сколько деревьев строить (больше = точнее, но дольше)
      - max_depth: глубина каждого дерева (глубже = сложнее зависимости, но риск переобучения)
      - learning_rate: скорость обучения (меньше = стабильнее, но медленнее сходится)
      - subsample/colsample: доля данных/признаков для каждого дерева (регуляризация)
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 40, 300),
            "eval_metric": "mae",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        
        model = MultiOutputRegressor(XGBRegressor(**params))
        scores = -cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        return scores.mean()

    print("\nЗапуск Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Лучшие параметры (Optuna):")
    print(study.best_params)
    print(f"Лучший MAE: {study.best_value:.3f} px")

    return study.best_params


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
    Визуализация точности измерений до и после коррекции модели.
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
    Показывает, как ошибка измерения зависит от угла поворота.
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


def searching_incorrect_offsets(X_test, y_pred, delta=0.5, show=False):
    """ 
    Сравнивает процент "плохих" измерений до и после применения модели.
    
    Как работает:
      1. Берём тестовые данные и считаем, сколько измерений имеют ошибку > delta пикселей
      2. Добавляем к измеренным смещениям поправку от модели
      3. Снова считаем процент "плохих" измерений
      4. Сравниваем: если процент уменьшился — модель помогла
    
    Дополнительно (при show=True):
      - Рисуем 5 графиков в ряд:
        * Корректные измерения без ML
        * Ошибочные измерения без ML (с процентом ошибок)
        * Корректные измерения с ML
        * Ошибочные измерения с ML (с новым процентом)
        * Истинные смещения (эталон)
    """
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

    # Отрисовка смещений с ML и без  
    if show:
        limit = [-25, 25]
        fig = plt.figure(figsize=(14, 6))
        axs = fig.subplots(1, 5)
        axs[0].scatter(emis_data["dx"].values, emis_data["dy"].values)
        axs[0].set(title=f"Corr emission", xlabel="dx", xlim=limit, ylabel="dy", ylim=limit)
        axs[1].scatter(emiss_err["dx"].values, emiss_err["dy"].values)
        axs[1].set(title=f"Errors {err_emis}%", xlabel="dx", xlim=limit, ylim=limit)
        axs[2].scatter(emis_data_ML["dx"].values, emis_data_ML["dy"].values)
        axs[2].set(title=f"Corr with ML", xlabel="dx", xlim=limit, ylim=limit)
        axs[3].scatter(emiss_err_ML["dx"].values, emiss_err_ML["dy"].values)
        axs[3].set(title=f"Errors with ML {err_emis_ML}%", xlabel="dx", xlim=limit, ylim=limit) 
        axs[4].scatter(all_data.loc[test_index, "true_dx"].values, all_data.loc[test_index, "true_dy"].values)
        axs[4].set(title=f"True", xlabel="dx", xlim=limit, ylim=limit)
        # plt.savefig((path_dir / f"graphics\\ML_SBS_{len(SBS_analysis())}.jpg"), dpi=800)
        plt.show()



path_dir = Path(path[0])
all_data = pd.read_csv((path_dir / "combined_data.csv"))

# Порог ошибки в пикселях для анализа "плохих" измерений
delta = 1

# Получаем оптимальный набор признаков 
X, y = get_selected_params(method=None, num_of_params=7, show_img=False, save_img=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_index = list(y_test.index)


# ==================== ВЫБОР И ОБУЧЕНИЕ МОДЕЛИ ====================

model = RandomForestRegressor(n_estimators=80, 
                              criterion="squared_error",  
                              min_samples_split=3, 
                              random_state=1)

# Альтернативные модели (раскомментировать для экспериментов):
# model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='MAE')
# model = XGBRegressor(n_estimators=70, random_state=1, eval_metric=mean_absolute_error, n_jobs=-1)
# model = XGBRegressor(**bayes_opt(X, y))  # с автоматическим подбором параметров

# Оборачиваем в MultiOutputRegressor, так как у нас две целевые переменные (dx и dy)
multi_model = MultiOutputRegressor(model)

# Загрузка ранее сохранённой модели (раскомментировать если нужно продолжить работу с готовой моделью)
# multi_model = load("model_2.joblib")


# ==================== ОЦЕНКА КАЧЕСТВА МОДЕЛИ ====================

# Кросс-валидация на полном датасете (5 фолдов)
# Раскомментировать для объективной оценки, но занимает время
# print("Значение кросс-валидации модели (MAE):", 
#       np.mean(cross_val_score(multi_model, X, y, cv=5, scoring='neg_mean_absolute_error') * -1), "\n")
# print("Значение кросс-валидации модели (MSE):", 
#       np.mean(cross_val_score(multi_model, X, y, cv=5, scoring='neg_mean_squared_error') * -1), "\n")


# Обучение модели на обучающей выборке
multi_model.fit(X_train, y_train)

# Замер скорости работы модели (важно для реального времени)
start = perf_counter()
y_pred = multi_model.predict(X_test)
finish = perf_counter()
print("Время инференса модели:", round((finish - start) / X_test.shape[0] * 1000000, 5), "мкс на изображение\n")


# ==================== СОХРАНЕНИЕ МОДЕЛИ (ОПЦИОНАЛЬНО) ====================

# Сохранение обученной модели для последующего использования
multi_model.fit(np.array(X), y)  # обучение на всех данных перед сохранением
# dump(multi_model, 'image_offset_corrector.joblib')


# ==================== АНАЛИЗ РЕЗУЛЬТАТОВ КОРРЕКЦИИ ====================

# Сравниваем точность ДО коррекции (просто измеренные смещения) и ПОСЛЕ (с поправкой от модели)
print("Сводка по тестовому набору данных:")
corr = X_test.loc[:, ['dx', 'dy']]  # измеренные смещения без коррекции
true_shift = all_data.loc[test_index, ['true_dx', 'true_dy']]  # истинные смещения
mae_corr = mean_absolute_error(true_shift, corr)  # ошибка без коррекции
print(round(y_test.describe().transpose(), 3))
print(f"MAE до коррекции: {mae_corr:.5f} пикселей\n")

print("Сводка по тестовому с поправками набору данных:")
y_test_ML = y_test - y_pred  # остаточная ошибка после коррекции
print(round(y_test_ML.describe().transpose(), 3))
corr_ML = corr + y_pred  # измеренные смещения + поправка от модели
mae = mean_absolute_error(true_shift, corr_ML)  # ошибка после коррекции
print(f"Средняя MAE после коррекции: {mae:.5f} пикселей\n")


# ==================== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================

# График 1: точность измерений до и после коррекции
plot_displacement_scatter(true_shift["true_dx"], true_shift['true_dy'], 
                          corr["dx"], corr["dy"],
                          corr_ML["dx"], corr_ML["dy"])

# График 2: ошибка в зависимости от угла поворота
plot_error_vs_angle(all_data.loc[test_index, "angle"],
                    true_shift["true_dx"], true_shift['true_dy'], 
                    corr["dx"], corr["dy"],
                    corr_ML["dx"], corr_ML["dy"])


# # ==================== ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК ====================

# Сравнение процентов "плохих" измерений до и после коррекции
searching_incorrect_offsets(X_test, y_pred, delta, show=True)

# Анализ "вредных" предсказаний — тех, что ухудшили результат
prediction_analysis(y_test, y_pred, test_index, delta, print_err=0)