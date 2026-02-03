"""
Скрипт для комплексного анализа признаков и отбора наиболее информативных переменных
для задачи коррекции систематических ошибок измерения смещений методом фазовой корреляции.

ОСНОВНЫЕ ЗАДАЧИ:
1. Разведочный анализ данных (EDA):
   - Статистические характеристики признаков
   - Проверка на пропущенные значения и дисбаланс классов
   - Визуализация корреляционных зависимостей

2. Отбор признаков двумя независимыми методами:
   а) PCA (Principal Component Analysis) — анализ структуры данных через главные компоненты
   б) Случайный лес (Random Forest) — оценка важности признаков через деревья решений

3. Формирование оптимального подмножества признаков для обучения регрессионных моделей.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sys import path
import seaborn as sns


def analysis_data(all_data):    
    """
    Проводит комплексный разведочный анализ данных (Exploratory Data Analysis).
    
    ВЫПОЛНЯЕТ:
      1. Вывод основных статистик для всех признаков:
         - Среднее, стандартное отклонение, мин/макс, квартили
      
      2. Проверку на пропущенные значения (NaN)
      
      3. Анализ дисбаланса классов:
         - Особое внимание к примерам с нулевыми истинными смещениями (dx=0, dy=0)
         - Такие примеры могут доминировать в синтетических данных и исказить обучение
      
      4. Построение корреляционной матрицы:
         - Выявление мультиколлинеарности (высокая корреляция между признаками)
         - Обнаружение признаков, сильно коррелирующих с целевыми переменными
    
    ПРАКТИЧЕСКАЯ ЦЕННОСТЬ:
      - Определение необходимости нормализации/стандартизации
      - Выявление избыточных признаков для последующего отбора
      - Понимание структуры данных перед построением моделей
    """
    print(all_data.describe().transpose())

    # Поиск нулевых значений (пропущенных данных)
    print("\nПропущенные значения (NaN) по столбцам:")
    print(all_data.isnull().sum())

    # Проверка на дисбаланс примеров с нулевыми смещениями
    # Важно: избыток нулевых смещений может привести к смещению предсказаний модели к нулю
    zero_count = len(all_data.loc[(all_data["true_dx"] == 0) & (all_data["true_dy"] == 0)])
    total_count = len(all_data)
    print(f"\nПримеры с нулевыми истинными смещениями (dx=0, dy=0): {zero_count} из {total_count} "
          f"({zero_count/total_count*100:.1f}%)")

    # Визуализация корреляционных зависимостей
    corr_matrix(all_data)


def corr_matrix(all_data):    
    """
    Строит тепловую карту корреляционной матрицы для визуального анализа зависимостей.
    """
    corr_matrix = all_data.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                vmin=-1, vmax=1, center=0, square=True)
    plt.title('Корреляционная матрица признаков и целевых переменных', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def PCA_analysis(X, n_components=0.97, quantity=None, show_img=True, save_img=False):
    """
    Анализ важности признаков через метод главных компонент (PCA).
    
    ПРИНЦИП РАБОТЫ:
      PCA преобразует исходное признаковое пространство в новое, где оси (главные компоненты)
      упорядочены по объяснённой дисперсии. Признаки, сильно влияющие на главные компоненты,
      считаются более информативными. Улавливает только линейные зависимости!!!
    
    :param X: DataFrame с признаками
    :param n_components: доля объяснённой дисперсии (0.0–1.0) или фиксированное число компонент
    :param quantity: количество топ-признаков для визуализации (по умолчанию все компоненты)
    :param show_img: показывать ли график важности признаков
    :param save_img: сохранять ли график в файл
    :return: массив названий топ-признаков по важности
    """
    # 1. Стандартизация данных (обязательно для PCA)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(X_std, columns=X.columns)

    # 2. Применение PCA
    # n_components=0.97 означает: сохранить минимальное число компонент, объясняющих 97% дисперсии
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    
    num_params = quantity if quantity else pca.n_components_

    # 3. Анализ объяснённой дисперсии
    print(f"Объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Дисперсия по компонентам: {[f'{v:.1%}' for v in pca.explained_variance_ratio_]}")
    
    # 4. Важность признаков через анализ нагрузок (loadings)
    # Нагрузка = вклад признака в главную компоненту
    # Сумма абсолютных нагрузок по всем компонентам → общая важность признака
    loadings = pca.components_.T  # shape: (n_features, n_components)
    feature_importance = np.abs(loadings).sum(axis=1)
    
    # 5. Формирование рейтинга признаков
    importance_df = pd.DataFrame({
        'feature': X_std.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # 6. Визуализация важности признаков
    if show_img:
        plt.figure(figsize=(12, 8))
        bars = plt.barh(importance_df['feature'][:num_params], 
                       importance_df['importance'][:num_params],
                       color=plt.cm.viridis(np.linspace(0.3, 0.9, num_params)))
        plt.xlabel('Суммарная абсолютная нагрузка на главные компоненты', fontsize=11)
        plt.ylabel('Признак', fontsize=11)
        plt.title(f'Важность признаков по анализу главных компонент (PCA)\n'
                 f'Объяснено {pca.explained_variance_ratio_.sum():.1%} дисперсии данными', 
                 fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()  # Самый важный признак — сверху
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        if save_img:
            # Создание директории для графиков при необходимости
            (path_dir / "graphics").mkdir(parents=True, exist_ok=True)
            plt.savefig(path_dir / "graphics/importance_PCA.jpg", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 7. Возврат топ-признаков
    selected = importance_df['feature'].values[:num_params]
    print(f"\nТоп-{num_params} признаков по анализу PCA:")
    for i, feat in enumerate(selected, 1):
        print(f"  {i:2d}. {feat}")
    
    return selected


def AVG_analysis(X, y, quantity=10, show_img=True, save_img=False):
    """
    Анализ важности признаков через ансамбль деревьев решений (Random Forest).
    
    ПРИНЦИП РАБОТЫ:
      Случайный лес оценивает важность признака по уменьшению критерия неоднородности
      (MSE для регрессии) при использовании признака для разделения в деревьях.
    
    :param X: DataFrame с признаками
    :param y: DataFrame с двумя целевыми переменными ['deviation_dx', 'deviation_dy']
    :param quantity: количество топ-признаков для визуализации и возврата
    :param show_img: показывать ли график важности признаков
    :param save_img: сохранять ли график в файл
    :return: массив названий топ-признаков по усреднённой важности
    """
    # Обучение двух независимых моделей для каждой целевой переменной
    # Это необходимо, так как зависимости для dx и dy могут отличаться
    model_x = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_y = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Обучение моделей на соответствующих целевых переменных
    model_x.fit(X, y["deviation_dx"])
    model_y.fit(X, y["deviation_dy"])

    # Получение важности признаков из каждой модели
    importance_x = model_x.feature_importances_
    importance_y = model_y.feature_importances_

    # Усреднение важности для балансированного отбора признаков
    # Признак считается важным, если он полезен для предсказания ОБОИХ смещений
    importance_avg = (importance_x + importance_y) / 2

    # Формирование структурированного рейтинга
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_x': importance_x,
        'importance_y': importance_y,
        'importance_avg': importance_avg
    }).sort_values(by='importance_avg', ascending=False)

    # Визуализация усреднённой важности признаков
    if show_img:
        plt.figure(figsize=(11, 7))
        colors = plt.cm.plasma(feature_importance_df['importance_avg'][:quantity] / 
                              feature_importance_df['importance_avg'].max())
        bars = plt.barh(feature_importance_df['feature'][:quantity], 
                       feature_importance_df['importance_avg'][:quantity],
                       color=colors)
        plt.xlabel('Усреднённая важность признака', fontsize=11)
        plt.ylabel('Признак', fontsize=11)
        plt.title('Влияние признаков на ошибку измерения смещения\n'
                 '(усреднённая важность по осям X и Y)', 
                 fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()  # Самый важный признак — сверху
        plt.grid(axis='x', alpha=0.3)
        
        # Добавление значений важности на график
        for i, v in enumerate(feature_importance_df['importance_avg'][:quantity]):
            plt.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=9)
        
        plt.tight_layout()
        if save_img:
            (path_dir / "graphics").mkdir(parents=True, exist_ok=True)
            plt.savefig((path_dir / "graphics/importance_avg.jpg"), dpi=500, bbox_inches='tight')
        plt.show()

    # Ограничение количества возвращаемых признаков реальным числом столбцов
    if quantity > len(X.columns): 
        quantity = len(X.columns)
    
    selected = feature_importance_df['feature'].values[:quantity]
    print(f"\n{'='*60}")
    print(f"ТОП-{quantity} ПРИЗНАКОВ ПО СЛУЧАЙНОМУ ЛЕСУ (усреднённая важность)")
    print(f"{'='*60}")
    for i, (feat, imp) in enumerate(zip(selected, feature_importance_df['importance_avg'].values[:quantity]), 1):
        print(f"  {i:2d}. {feat:25s} | важность: {imp:.4f}")
    
    return selected


def get_selected_params(method="AVG", num_of_params=10, show_img=False, save_img=False):
    """
    Унифицированный интерфейс для отбора оптимального набора признаков.
    
    ПОДДЕРЖИВАЕМЫЕ МЕТОДЫ:
      1. 'PCA' — отбор на основе анализа главных компонент
         * Преимущество: выявляет скрытую структуру данных
         * Рекомендация: использовать при наличии сильных линейных зависимостей
      
      2. 'AVG' — отбор на основе усреднённой важности из случайного леса
         * Преимущество: учитывает нелинейные зависимости с целевой переменной
         * Рекомендация: основной метод для регрессионных задач
      
      3. 'manual' (по умолчанию) — ручной выбор эмпирически оптимальных признаков
         * Набор: ['angle', 'dx', 'dy', 'sharpness', 'entropy', 'snr', 'mean_brightness']
    
    ВОЗВРАЩАЕТ:
      Кортеж (X_selected, y):
        - X_selected: DataFrame только с отобранными признаками
        - y: DataFrame с целевыми переменными ['deviation_dx', 'deviation_dy']
    """
    # Загрузка полного датасета
    all_data = pd.read_csv((path_dir / "combined_data.csv"))

    # Опциональный разведочный анализ (раскомментировать для диагностики)
    # analysis_data(all_data)

    # Формирование матрицы признаков и целевых переменных
    # Исключаем 'true_dx', 'true_dy' — они используются только для расчёта отклонений,
    # но не должны входить в признаки (утечка информации о целевой переменной!)
    feature_columns = [col for col in all_data.columns if col not in {'true_dx', 'true_dy'}]
    
    # Целевые переменные — последние два столбца (отклонения по осям)
    y = all_data.loc[:, ['deviation_dx', 'deviation_dy']]
    
    # Все остальные столбцы — признаки для обучения
    X = all_data.loc[:, [col for col in feature_columns if col not in ['deviation_dx', 'deviation_dy']]]

    # Выбор метода отбора признаков
    if method in ['AVG', 'PCA']:
        if method == 'AVG':
            print(f"\nЗапуск отбора признаков методом случайного леса (топ-{num_of_params})...")
            params = AVG_analysis(X, y, quantity=num_of_params, show_img=show_img, save_img=save_img)
        elif method == 'PCA':
            print(f"\nЗапуск отбора признаков методом PCA (объяснение 97% дисперсии, топ-{num_of_params})...")
            params = PCA_analysis(X, n_components=0.97, quantity=num_of_params, 
                                 show_img=show_img, save_img=save_img)
    else:
        # Эмпирически оптимальный набор признаков для задачи коррекции смещений
        # Выбран на основе многократных экспериментов и анализа важности
        params = ['angle', 'dx', 'dy', 'sharpness', 'entropy', 'snr', 'mean_brightness']
        print(f"\nИспользован эмпирически оптимальный набор признаков ({len(params)} шт.):")
        for i, p in enumerate(params, 1):
            print(f"  {i}. {p}")

    # Формирование итогового набора данных с отобранными признаками
    X_selected = X.loc[:, params]
    
    return X_selected, y



path_dir = Path(path[0])
# Пример использования: отбор признаков методом PCA с визуализацией
# get_selected_params(method="PCA", num_of_params=7, show_img=True)

# Пример использования: отбор признаков методом случайного леса
# get_selected_params(method="AVG", num_of_params=7, show_img=True, save_img=True)



# scaler = StandardScaler()
# data_std = scaler.fit_transform(all_data)
# data_std = pd.DataFrame(data_std, columns=all_data.columns)