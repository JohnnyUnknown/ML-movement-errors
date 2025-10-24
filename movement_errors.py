import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sys import path


def PCA_analysis(quantity=5):
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


def AVG_analysis(quantity=5):
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
    # importances = np.array([est.feature_importances_ for est in model_forest.estimators_])
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
    # avg_importance_model = RandomForestRegressor(n_estimators=200, random_state=42)
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
    sbs = SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=42), threshold=0.01)
    sbs.fit(X_train, y_train)
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
        значений в диапазоне [true_dx - coef, true_dx + coef] (для dy аналогично)."""
    if coef <= 0:
        raise ValueError("coef должен быть положительным")
    if coef < 1:
        raise ValueError("Рекомендуется использовать coef >= 1. При coef < 1 границы инвертируются.")

    df = all_data.copy()

    # --- Обработка dx ---
    # Границы по умолчанию (для ненулевых true_dx)
    low_dx = df['true_dx'] - coef
    high_dx = df['true_dx'] + coef

    # # Упорядочиваем границы (на случай отрицательных значений)
    # dx_min = np.minimum(low_dx, high_dx)
    # dx_max = np.maximum(low_dx, high_dx)

    # # Замена границ для строк, где true_dx == 0
    # zero_dx_mask = (df['true_dx'] == 0)
    # dx_min = np.where(zero_dx_mask, -1.0, dx_min)
    # dx_max = np.where(zero_dx_mask,  1.0, dx_max)

    # --- Обработка dy ---
    low_dy = df['true_dy'] - coef
    high_dy = df['true_dy'] + coef

    # dy_min = np.minimum(low_dy, high_dy)
    # dy_max = np.maximum(low_dy, high_dy)

    # zero_dy_mask = (df['true_dy'] == 0)
    # dy_min = np.where(zero_dy_mask, -1.0, dy_min)
    # dy_max = np.where(zero_dy_mask,  1.0, dy_max)

    # --- Фильтрация ---
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


path_dir = Path(path[0])

all_data = pd.read_csv((path_dir / "combined_data.csv"))
coef = 1


# Получение всех отфильтрованных измерений для анализа
# all_data_window, err_win = tolerance_window(all_data, coef)
# emis_corr_data, err_emis = measurement_deviations(all_data, coef)

# Получение всех ошибочных измерений для анализа
# window_err = get_deviation_data(all_data=all_data, clear_data=all_data_window)
# emiss_err = get_deviation_data(all_data=all_data, clear_data=emis_corr_data)

# all_data, _ = tolerance_window(all_data, coef)

feature_columns = [col for col in all_data.columns if col not in {'true_dx', 'true_dy'}]
y = all_data.loc[:, feature_columns[-2:]]
X = all_data.loc[:, feature_columns[:-2]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

test_index = list(y_test.index)

# scaler_X = StandardScaler()
# X_train = scaler_X.fit_transform(X_train)
# X_test = scaler_X.transform(X_test)



# Проверка на дисбаланс примеров с нулевыми смещениями
# df_y_train = pd.DataFrame(y_train, columns=y.columns)
# print("dx=0, dy=0:", len(df_y_train.loc[(df_y_train["true_dx"] == 0) & (df_y_train["true_dy"] == 0)]), "раз из", len(df_y_train), "\n")
# print(pd.DataFrame(X_train, columns=X.columns).describe().transpose().loc[:, ["mean", "std", "min", "50%", "max"]])
# print(X.describe().transpose().loc[:, ["mean", "std", "min", "50%", "max"]])



# X_train_clear = pd.DataFrame(X_train, columns=X.columns).loc[:, SBS_analysis()]
# X_test_clear = pd.DataFrame(X_test, columns=X.columns).loc[:, SBS_analysis()]
model_forest = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model_forest.fit(X_train, y_train)
y_pred = model_forest.predict(X_test)
# print(np.expand_dims(y_pred[:, 1], axis=0))

# all_pred_data = pd.concat([X_test, pd.DataFrame(y_pred, columns=["deviation_dx", "deviation_dy"], index=X_test.index)], axis=1)

data = X_test.copy()
data.insert(3, "true_dx", all_data.loc[test_index, "true_dx"])
data.insert(4, "true_dy", all_data.loc[test_index, "true_dy"])

# Добавление поправки к смещениям
data["dx"] = data["dx"] + y_pred[:, 0]
data["dy"] = data["dy"] + y_pred[:, 1]

# Получение всех отфильтрованных измерений для анализа
all_data_window, err_win = tolerance_window(data, coef)
emis_data, err_emis = measurement_deviations(data, coef)

# Получение всех ошибочных измерений для анализа
window_err = get_deviation_data(all_data=data, clear_data=all_data_window)
emiss_err = get_deviation_data(all_data=data, clear_data=emis_data)

# # print(window_err.loc[:, ["dx", "dy", "true_dx", "true_dy"]])



# PCA_analysis(10)
AVG_analysis(10)
# SBS_analysis()


# -----------------------------Анализ векторов смещения--------------------------------------------

# true_vec = np.linalg.norm([all_data.loc[test_index, ["true_dx"]].values, 
#                        all_data.loc[test_index, ["true_dy"]].values], axis=0)
# e_corr_vec = np.linalg.norm([X_test["dx"].values, X_test["dy"].values], axis=0)
# e_true_vec = np.linalg.norm(y_test, axis=1)
# e_pred_vec = np.linalg.norm(y_pred, axis=1)

# print(np.round(e_true[:20], 2), "\n")
# print(np.round(e_corr[:20], 2), "\n")
# print(np.round(true[:20].T, 2), "\n")
# print(np.round(e_pred[:20], 2))

# print(np.expand_dims(true.flatten(), axis=0).T.shape, 
#       np.expand_dims(e_corr, axis=0).T.shape, 
#       np.expand_dims(e_pred, axis=0).T.shape, 
#       np.expand_dims(e_true, axis=0).T.shape)

# result_vectors = pd.DataFrame(np.concat((
#                         np.expand_dims(true_vec.flatten(), axis=0).T, 
#                         np.expand_dims(e_corr_vec, axis=0).T, 
#                         np.expand_dims(e_true_vec, axis=0).T, 
#                         np.expand_dims(e_pred_vec, axis=0).T), axis=1),
#                         index=X_test.index, 
#                         columns=["true_vector", "corr_vector", "true_dev", "pred_dev"])
# print(result_vectors)

#----------------------------------------------------------------------------------------------------------

shifts = all_data.loc[test_index, ["true_dx", "true_dy", "dx", "dy"]].values
result = round(pd.DataFrame(np.concat((shifts, y_test, y_pred), axis=1),
                        index=X_test.index, 
                        columns=["true_dx", "true_dy", "corr_dx", "corr_dy", "true_dev_dx", 
                                 "true_dev_dy", "pred_dev_dx", "pred_dev_dy"]), 3)
# csv_path = Path(path[0] + "\\result_data.csv")
# result.to_csv(csv_path, index=False, encoding='utf8')


# print(f"MAE фактическая:       {e_true.mean():.3f} px")
# print(f"MAE корреляционное:    {e_corr.mean():.3f} px")
# print(f"MAE предсказанное:     {e_pred.mean():.3f} px")
# print(f"Отклонение предсказ.:  {(e_true.mean() - e_pred.mean()):.3f} px ({(1 - e_pred.mean()/(e_true.mean() + 1e-8))*100:.1f}%)")
# print(f"Отклонение корреляц.:  {(e_true.mean() - e_corr.mean()):.3f} px ({(1 - e_corr.mean()/(e_true.mean() + 1e-8))*100:.1f}%)")
# print(f"Медианное смещение:    {np.median(e_true):.3f} px")
# print(f"Медиан. коррел. смещ.: {np.median(e_corr):.3f} px")
# print(f"Медиан. предск. смещ.: {np.median(e_pred):.3f} px\n")

# print(pd.DataFrame(e_true, columns=["true"]).describe().transpose(), "\n")
# print(pd.DataFrame(e_corr, columns=["corr"]).describe().transpose(), "\n")
# print(pd.DataFrame(e_pred, columns=["pred"]).describe().transpose())



# # Отрисовка данных корреляции
# fig = plt.figure(figsize=(14, 6))
# axs = fig.subplots(1, 4)
# axs[0].scatter(all_data_window["dx"].values, all_data_window["dy"].values)
# axs[0].set(title=f"ML Corr with window", xlim=[-30, 30], ylim=[-30, 30])
# axs[1].scatter(window_err["dx"].values, window_err["dy"].values)
# axs[1].set(title=f"ML window. Ошибка {err_win}%", xlim=[-30, 30], ylim=[-30, 30])  # , xlim=[-30, 30], ylim=[-30, 30]
# axs[2].scatter(emis_data["dx"].values, emis_data["dy"].values)
# axs[2].set(title=f"ML Emission", xlim=[-30, 30], ylim=[-30, 30])
# axs[3].scatter(emiss_err["dx"].values, emiss_err["dy"].values)
# axs[3].set(title=f"ML Emisson error. Ошибка {err_emis}%", xlim=[-30, 30], ylim=[-30, 30])
# # axs[4].scatter(all_data["true_dx"].values, all_data["true_dy"].values)
# # axs[4].set(title=f"True", xlim=[-30, 30], ylim=[-30, 30])
# # plt.savefig((path_dir / "graphics\\ML_SBS.jpg"), dpi=800)
# plt.show()




# print(all_data.loc[list(X_test.index)[:-1], ["true_dx"]])

# fig = plt.figure()
# axs = fig.subplots(1, 3)
# axs[0].scatter(all_data.loc[list(X_test.index)[:-1], ["true_dx"]].values, 
#                all_data.loc[list(X_test.index)[:-1], ["true_dy"]].values)
# axs[0].set(title=f"True", xlim=[-30, 30], ylim=[-30, 30])   # , xlim=[-30, 30], ylim=[-30, 30]
# axs[1].scatter(X_test["dx"].values, X_test["dy"].values)
# axs[1].set(title=f"Corr", xlim=[-30, 30], ylim=[-30, 30])
# axs[2].scatter(X_test["dx"].values + y_pred[:, 0], X_test["dy"].values + y_pred[:, 0])
# axs[2].set(title=f"Pred", xlim=[-30, 30], ylim=[-30, 30])
# plt.show()

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





# # from sklearn.linear_model import LinearRegression

# # model_LinearRegression = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
# # print(model_LinearRegression.score(X_train, y_train))

# # y_pred_LinearRegression = model_LinearRegression.predict(X_test)
# # mae_LinearRegression = mean_absolute_error(y_test, y_pred_LinearRegression)
# # print(f"MAE LinearRegression после коррекции: {mae_LinearRegression:.5f} пикселей")



# new = np.zeros(y_test.shape)
# mae_forest = mean_absolute_error(y_test, new)
# print(f"MAE до коррекции: {mae_forest:.5f} пикселей")