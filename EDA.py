import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sys import path
import seaborn as sns


def corr_matrix(all_data):    
    corr_matrix = all_data.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица')
    plt.show()


def PCA_analysis(X, quantity=10, show_img=True, save_img=False):
    pca = PCA()
    pca.fit(X)
    # Получаем нагрузки (loadings): компоненты × признаки
    loadings = pca.components_.T 

    # Вклад признаков в PC1 (можно заменить на PC1 + PC2 и т.д.)
    pc1_loadings = loadings[:, 0]  # первая главная компонента

    # Абсолютные значения — потому что знак показывает направление, а не важность
    pc1_importance = np.abs(pc1_loadings)

    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'pc1_loading': pc1_loadings,
        'pc1_abs_loading': pc1_importance
    }).sort_values(by='pc1_abs_loading', ascending=False)

    if show_img:
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['pc1_abs_loading'])
        plt.xlabel('Абсолютная нагрузка на PC1')
        plt.title('Вклад признаков в первую главную компоненту (PCA)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        if save_img: 
            plt.savefig((path_dir / "graphics\\importance_PCA.jpg"), dpi=500)

    if quantity > len(X.columns): 
        quantity = len(X.columns)
    top_features = importance_df['feature'].values[:quantity]
    print("PCA:", top_features)
    return top_features


def AVG_analysis(X, y, quantity=10, show_img=True, save_img=False):
    # Обучим две модели (или одну с multi-output — но важность будет усреднена)
    model_x = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_y = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Обучение
    model_x.fit(X, y["deviation_dx"])
    model_y.fit(X, y["deviation_dy"])

    # Получаем важности
    importance_x = model_x.feature_importances_
    importance_y = model_y.feature_importances_

    # Средняя важность по осям (опционально)
    importance_avg = (importance_x + importance_y) / 2

    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_x': importance_x,
        'importance_y': importance_y,
        'importance_avg': importance_avg
    }).sort_values(by='importance_avg', ascending=False)

    if show_img:
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance_avg'])
        plt.xlabel('Средняя важность признака')
        plt.title('Влияние признаков на ошибку измерения смещения усредненное по dx и dy.')
        plt.gca().invert_yaxis()  # Самый важный — сверху
        plt.tight_layout()
        plt.show()
        if save_img:
            plt.savefig((path_dir / "graphics\\importance_avg.jpg"), dpi=500)

    if quantity > len(X.columns): 
        quantity = len(X.columns)
    top_features = feature_importance_df['feature'].values[:quantity]
    print("AVG:", top_features)
    return top_features


def SBS_analysis(X, y, quantity=10, show_img=True, save_img=False):
    model_x = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y["deviation_dx"])
    model_y = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y["deviation_dy"])

    if quantity > len(X.columns): 
        quantity = len(X.columns)
    selector_x = SelectFromModel(model_x, max_features=quantity, threshold=0.001).fit(X, y["deviation_dx"])
    selector_y = SelectFromModel(model_y, max_features=quantity, threshold=0.001).fit(X, y["deviation_dy"])

    # selected_features_x = X.columns[selector_x.get_support()].tolist()
    # selected_features_y = X.columns[selector_y.get_support()].tolist()

    feature_importance_x = pd.DataFrame({
        'feature': X.columns,
        'importance': selector_x.estimator.feature_importances_
    }).sort_values(by='importance', ascending=False)

    feature_importance_y = pd.DataFrame({
        'feature': X.columns,
        'importance': selector_y.estimator.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # Объединяем DataFrame по признакам, сохраняя максимальное значение importance
    merged_df = pd.merge(feature_importance_x, feature_importance_y, on='feature', how='outer', suffixes=('_x', '_y'))
    # Создаем новый столбец, где для каждого признака берется максимум из importance_x и importance_y
    merged_df['importance'] = merged_df[['importance_x', 'importance_y']].max(axis=1)
    # Выбираем только нужные столбцы и сортируем по важности
    result = merged_df[['feature', 'importance']].sort_values(by='importance', ascending=False)

    # # Вывод диаграмм влияния признаков на x и y по отдельности
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    # # Диаграмма важности для model_x
    # axes[0].barh(feature_importance_x['feature'], feature_importance_x['importance'], color='b')
    # axes[0].set_title('Важность признаков по X')
    # axes[0].invert_yaxis()  # чтобы важные фичи были сверху
    # axes[0].set_xlabel('Важность признака')
    # # Диаграмма важности для model_y
    # axes[1].barh(feature_importance_y['feature'], feature_importance_y['importance'], color='r')
    # axes[1].set_title('Важность признаков по Y')
    # axes[1].invert_yaxis()
    # axes[1].set_xlabel('Важность признака')
    # plt.tight_layout()
    # plt.show()

    if show_img:
        plt.figure(figsize=(10, 6))
        plt.barh(result['feature'], result['importance'])
        plt.xlabel('Средняя важность признака')
        plt.title('Влияние признаков на ошибку измерения смещения по dx и dy.')
        plt.gca().invert_yaxis()  # Самый важный — сверху
        plt.tight_layout()
        plt.show()
        if save_img:
            plt.savefig((path_dir / "graphics\\importance_SBS.jpg"), dpi=500)

    params = result["feature"].values[:quantity]
    print("SBS:", params)
    return params


def analysis_data(all_data):    
    print(all_data.describe().transpose())

    # Поиск нулевых значений
    print(all_data.isnull().sum())

    # Проверка на дисбаланс примеров с нулевыми смещениями
    print("dx=0, dy=0:", len(all_data.loc[(all_data["true_dx"] == 0) & (all_data["true_dy"] == 0)]), 
          "раз из", len(all_data), "\n")

    corr_matrix(all_data)


def get_selected_params(method="AVG", num_of_params=10, show_img=False, save_img=False):
    all_data = pd.read_csv((path_dir / "angles_2deg\\combined_data_2deg.csv"))

    # analysis_data(all_data)

    # Убираем 'true_dx' и 'true_dy', они нужны только для анализа
    feature_columns = [col for col in all_data.columns if col not in {'true_dx', 'true_dy'}]
    y = all_data.loc[:, feature_columns[-2:]]
    X = all_data.loc[:, feature_columns[:-2]]

    if method in ['AVG', 'PCA', 'SBS']:
        if method == 'AVG':
            params = AVG_analysis(X, y, quantity=num_of_params, show_img=show_img, save_img=save_img)
        if method == 'PCA':
            params = PCA_analysis(X, quantity=num_of_params, show_img=show_img, save_img=save_img)
        if method == 'SBS':
            params = SBS_analysis(X, y, quantity=num_of_params, show_img=show_img, save_img=save_img)
    else:
        print("Ошибка. Выбран метод отбора признаков по умолчанию!")
        params = AVG_analysis(X, y, quantity=num_of_params, show_img=show_img, save_img=save_img)

    # Отбор и возврат наиболее значимых параметров
    return X.loc[:, params], y


path_dir = Path(path[0])