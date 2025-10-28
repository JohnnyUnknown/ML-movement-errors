import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sys import path
import optuna
from xgboost import XGBRegressor


def PCA_analysis(quantity=15):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_std)
    # 3. Получаем нагрузки (loadings): компоненты × признаки
    loadings = pca.components_.T  # shape: (n_features, n_components)

    # 4. Вклад признаков в PC1 (можно заменить на PC1 + PC2 и т.д.)
    pc1_loadings = loadings[:, 0]  # первая главная компонента

    # Абсолютные значения — потому что знак показывает направление, а не важность
    pc1_importance = np.abs(pc1_loadings)

    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'pc1_loading': pc1_loadings,
        'pc1_abs_loading': pc1_importance
    }).sort_values(by='pc1_abs_loading', ascending=False)

    # plt.figure(figsize=(10, 6))
    # plt.barh(importance_df['feature'], importance_df['pc1_abs_loading'])
    # plt.xlabel('Абсолютная нагрузка на PC1')
    # plt.title('Вклад признаков в первую главную компоненту (PCA)')
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # # plt.savefig((path_dir / "graphics\\importance_PCA.jpg"), dpi=500)
    # plt.show()
    print("PCA:", importance_df['feature'].values[:quantity])
    return importance_df['feature'].values[:quantity]


def AVG_analysis(quantity=15):
    # Обучим две модели (или одну с multi-output — но важность будет усреднена)
    model_x = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_y = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    # Обучение
    model_x.fit(X, y["deviation_dx"])
    model_y.fit(X, y["deviation_dy"])

    # Получаем важности
    importance_x = model_x.feature_importances_
    importance_y = model_y.feature_importances_

    # Средняя важность по осям (опционально)
    importance_avg = (importance_x + importance_y) / 2

    # Создаём DataFrame для удобства
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_x': importance_x,
        'importance_y': importance_y,
        'importance_avg': importance_avg
    }).sort_values(by='importance_avg', ascending=False)

    top_features = feature_importance_df

    # plt.figure(figsize=(10, 6))
    # plt.barh(top_features['feature'], top_features['importance_avg'])
    # plt.xlabel('Средняя важность признака')
    # plt.title('Влияние признаков на ошибку измерения смещения')
    # plt.gca().invert_yaxis()  # Самый важный — сверху
    # plt.tight_layout()
    # # plt.savefig((path_dir / "graphics\\importance_avg.jpg"), dpi=500)
    # plt.show()
    print("AVG:", top_features['feature'].values[:quantity])
    return top_features['feature'].values[:quantity]


def SBS_analysis():
    # importances = np.array([est.feature_importances_ for est in multi_model.estimators_])
    # # Усредняем важность по осям x и y
    # mean_importances = np.mean(importances, axis=0)
    # print(mean_importances.shape, X_train.shape)
    
    # # Создаём DataFrame для удобства
    # feature_importance_df = pd.DataFrame({
    #     'feature': X_train.columns,
    #     'importance': mean_importances
    # }).sort_values(by='importance', ascending=False)

    # print("\nВажность признаков (топ-15):")
    # print(feature_importance_df.head(15))

    # # Создаём "усреднённую" модель для SelectFromModel
    # # (sklearn не поддерживает MultiOutput напрямую в SelectFromModel)
    # avg_importance_model = RandomForestRegressor(n_estimators=150, random_state=42)
    # avg_importance_model.fit(X_train, np.linalg.norm(y_train, axis=1))  # обучаем на модуле ошибки

    # selector = SelectFromModel(avg_importance_model, prefit=True)
    # X_selected = selector.transform(X_train)

    # selected_features = X_train.columns[selector.get_support()].tolist()

    # print(f"\nОтобрано признаков: {len(selected_features)} из {X_train.shape[1]}")
    # print("Отобранные признаки:")
    # for i, feat in enumerate(selected_features, 1):
    #     print(f"  {i}. {feat}")

    # plt.figure(figsize=(10, 6))
    # top_n = min(15, len(feature_importance_df))
    # top_features = feature_importance_df.head(top_n)

    # plt.barh(top_features['feature'], top_features['importance'])
    # plt.xlabel('Средняя важность признака')
    # plt.title('Важность признаков для предсказания ошибки смещения')
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # # plt.savefig((path_dir / "graphics\\importance_SBS.jpg"), dpi=500)
    # plt.show()
    sbs = SelectFromModel(model, threshold=0.03)
    sbs.fit(X, y)
    print("SBS:", sbs.get_feature_names_out())
    return sbs.get_feature_names_out()



def tolerance_window(all_data, coef = 1.5):
    """Функция отсеивает выбросы, которые выходят за рамки окна диапазонов истинных значений 
        и возвращает новый DataFrame."""
    min_shift = all_data["true_dx"].min() - coef
    max_shift = all_data["true_dx"].max() + coef
    new_data = all_data.loc[(all_data["dx"] < max_shift)
                            & (all_data["dy"] < max_shift)
                            & (all_data["dx"] > min_shift)
                            & (all_data["dy"] > min_shift)]
    error_percent = round((1 - new_data.shape[0] / len(all_data)) * 100, 2)
    print("Процент выбросов корреляционного метода общ.:", error_percent, "%")
    return new_data, error_percent


def measurement_deviations(all_data, coef = 1.5):
    """Функция отсеивает выбросы индивидуально, сравнивая предсказанные значения (dx, dy) с (true_dx, true_dy).
        Возвращает новый DataFrame состоящий только из полей с разницей предсказанных и истинных
        значений в диапазоне [true_dx +/- coef] (для dy аналогично)."""
    df = all_data.copy()
    
    low_dx = df['true_dx'] - coef
    high_dx = df['true_dx'] + coef

    low_dy = df['true_dy'] - coef
    high_dy = df['true_dy'] + coef

    mask = (
        (df['dx'] >= low_dx) & (df['dx'] <= high_dx) &
        (df['dy'] >= low_dy) & (df['dy'] <= high_dy)
    )
    error_percent = round((1 - df[mask].shape[0] / len(all_data)) * 100, 2)
    print("Процент выбросов корреляционного метода индивид.:", error_percent, "%")

    return df[mask].copy(), error_percent


def get_deviation_data(all_data, clear_data):
    """Функция получает полный набор данных и отфильтрованный. Возвращает датафрейм с ошибочными измерениями."""
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

all_data = pd.read_csv((path_dir / "combined_data_2.csv"))
coef = 0.5

feature_columns = [col for col in all_data.columns if col not in {'true_dx', 'true_dy'}]
y = all_data.loc[:, feature_columns[-2:]]
X = all_data.loc[:, feature_columns[:-2]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
test_index = list(y_test.index)


# scaler_X = StandardScaler()
# X_train = pd.DataFrame(scaler_X.fit_transform(X_train), columns=X.columns, index=list(y_train.index))
# X_test = pd.DataFrame(scaler_X.transform(X_test), columns=X.columns, index=test_index)



# # Проверка на дисбаланс примеров с нулевыми смещениями
# df_y_train = pd.DataFrame(y_train, columns=y.columns)
# print("dx=0, dy=0:", len(df_y_train.loc[(df_y_train["true_dx"] == 0) & (df_y_train["true_dy"] == 0)]), "раз из", len(df_y_train), "\n")
# print(pd.DataFrame(X_train, columns=X.columns).describe().transpose().loc[:, ["mean", "std", "min", "50%", "max"]])
# print(X.describe().transpose().loc[:, ["mean", "std", "min", "50%", "max"]])



model = RandomForestRegressor(n_estimators=40, random_state=42)
# model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# model = XGBRegressor(n_estimators=342, max_depth=7, learning_rate=0.097, 
#                       random_state=42, subsample=0.94, colsample_bytree=0.64)
# multi_model = load("model_2.joblib")


# Отбор наиболее значимых параметров
# params = SBS_analysis()
params = ['angle', 'dx', 'dy', 'response', 'snr']
X_train_clear = pd.DataFrame(X_train, columns=X.columns).loc[:, params]
X_test_clear = pd.DataFrame(X_test, columns=X.columns).loc[:, params]



# # Оптимизация гиперпараметров моделей
# bayes_opt(X.loc[:, params], y)

# for num in range(50, 150, 10):
#     model = RandomForestRegressor(n_estimators=num)
#     multi_model = MultiOutputRegressor(model)
#     scores_mae = cross_val_score(multi_model, X.loc[:, params], y, cv=3, scoring='neg_mean_absolute_error')
#     scores_mse = cross_val_score(multi_model, X.loc[:, params], y, cv=3, scoring='neg_mean_squared_error')
#     print(f"{num}: Avg MAE (dx & dy) = {-scores_mae.mean():.4f} ± {scores_mae.std():.4f}")
#     print(f"{num}: Avg MSE (dx & dy) = {-scores_mse.mean():.4f} ± {scores_mse.std():.4f}")



multi_model = MultiOutputRegressor(model)
multi_model.fit(X_train_clear, y_train)
y_pred = multi_model.predict(X_test_clear)

# # Сохранение модели
# multi_model.fit(np.array(X.loc[:, params]), y)
# dump(multi_model, 'model_2.joblib')


# Сравнение истииных отклонений с измеренными до правки и после 
corr = X_test.loc[:, ['dx', 'dy']]
true_shift = all_data.loc[test_index, ['true_dx', 'true_dy']]
mae_corr = mean_absolute_error(true_shift, corr)
print(f"MAE до коррекции: {mae_corr:.5f} пикселей")
mae = mean_absolute_error(true_shift, corr + y_pred, multioutput='uniform_average')
print(f"Средняя MAE после коррекции: {mae:.3f} пикселей")


# PCA_analysis(15)
# AVG_analysis(15)
# SBS_analysis()


# Добавление к тестовым данным столбцов с истинными смещениями для анализа отклонений
data = X_test.copy()
data.insert(3, "true_dx", all_data.loc[test_index, "true_dx"])
data.insert(4, "true_dy", all_data.loc[test_index, "true_dy"])



# Получение всех отфильтрованных измерений для анализа без ML
emis_data, err_emis = measurement_deviations(data, coef)
emiss_err = get_deviation_data(all_data=data, clear_data=emis_data)



# Добавление поправки к измеренным смещениям
data["dx"] = data["dx"] + y_pred[:, 0]
data["dy"] = data["dy"] + y_pred[:, 1]

# Получение всех отфильтрованных измерений для анализа с поправками ML
emis_data_ML, err_emis_ML = measurement_deviations(data, coef)
emiss_err_ML = get_deviation_data(all_data=data, clear_data=emis_data_ML)


# # # Вывод предсказаний с противоположным знаком от истинного
# # y_pred = pd.DataFrame(y_pred, index=test_index, columns=['dev_dx', 'dev_dy'])
# # out_errors = []
# # for i in test_index:
# #     pred_dev_x, true_dev_x = round(y_pred.loc[i, 'dev_dx'], 1), round(y_test.loc[i, 'deviation_dx'], 1)
# #     pred_dev_y, true_dev_y = round(y_pred.loc[i, 'dev_dy'], 1), round(y_test.loc[i, 'deviation_dy'], 1)

# #     if ((true_dev_x < 0 and pred_dev_x > 0) or (true_dev_x > 0 and pred_dev_x < 0) or
# #         (true_dev_y < 0 and pred_dev_y > 0) or (true_dev_y > 0 and pred_dev_y < 0)):
# #         # pred_dev_x > true_dev_x or pred_dev_x <= true_dev_x - coef or 
# #         # pred_dev_y > true_dev_y + coef or pred_dev_y <= true_dev_y - coef):
# #         out_errors.append([i, y_test.loc[i, :].values, y_pred.loc[i, :].values])
# # print(*out_errors, len(out_errors), sep="\n")


# # # Сохранение всех смещений в таблицу
# # shifts = all_data.loc[test_index, ["true_dx", "true_dy", "dx", "dy"]].values
# # result = round(pd.DataFrame(np.concat((shifts, y_test, y_pred), axis=1),
# #                         index=X_test.index, 
# #                         columns=["true_dx", "true_dy", "corr_dx", "corr_dy", "true_dev_dx", 
# #                                  "true_dev_dy", "pred_dev_dx", "pred_dev_dy"]), 3)
# # csv_path = Path(path[0] + "\\result_data.csv")
# # result.to_csv(csv_path, index=False, encoding='utf8')



# Отрисовка данных корреляции
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



# --------------------------------------------------------------------------------------------------

# # Модель
# model = MLPRegressor(
#     hidden_layer_sizes=(64, 32),
#     activation='relu',
#     solver='adam',
#     max_iter=500,
#     random_state=42
# )
# model.fit(X_train, y_train)

# # Оценка
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# print(f"MAE MLP после коррекции: {mae:.5f} пикселей")
