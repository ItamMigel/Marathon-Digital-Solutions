#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel
import datetime
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('data/data.csv', sep=';')
# df.drop()

# Предварительная обработка данных
print("Обработка данных...")

# Преобразование столбца времени в datetime
df['Время'] = pd.to_datetime(df['Время'], format='%d.%m.%Y %H:%M')

# Добавление временных признаков
df['Hour'] = df['Время'].dt.hour
df['DayOfWeek'] = df['Время'].dt.dayofweek
df['Month'] = df['Время'].dt.month
df['Day'] = df['Время'].dt.day
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['TimeOfDay'] = df['Hour'].apply(lambda x: 'morning' if 5 <= x < 12 else 
                                ('afternoon' if 12 <= x < 17 else 
                                 ('evening' if 17 <= x < 22 else 'night')))
df = pd.get_dummies(df, columns=['TimeOfDay'], drop_first=True)

# Разделение данных на два участка (согласно description.txt)
df_area1 = df[[col for col in df.columns if ('2' not in col) or col == 'Время' or col.startswith('Hour') or 
              col.startswith('DayOfWeek') or col.startswith('Month') or col.startswith('Day') or 
              col.startswith('IsWeekend') or col.startswith('TimeOfDay')]]
df_area2 = df[[col for col in df.columns if ('1' not in col) or col == 'Время' or col.startswith('Hour') or 
              col.startswith('DayOfWeek') or col.startswith('Month') or col.startswith('Day') or 
              col.startswith('IsWeekend') or col.startswith('TimeOfDay')]]

# Функция для обнаружения и обработки выбросов
def handle_outliers(df, columns, method='iqr'):
    df_cleaned = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Замена выбросов на границы
            df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
            df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
        
        elif method == 'zscore':
            z_scores = stats.zscore(df_cleaned[col].dropna())
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3)
            
            # Создаем маску для всех строк датафрейма
            mask = df_cleaned[col].notna()
            mask[mask] = filtered_entries
            
            # Замена выбросов на NaN для последующей интерполяции
            df_cleaned.loc[~mask, col] = np.nan
            
    return df_cleaned

# Функция для анализа данных по участку
def analyze_area(df_area, area_name):
    print(f"\n--- Анализ данных участка {area_name} ---")
    
    # Базовая статистика
    print(f"\nРазмерность данных: {df_area.shape}")
    
    # Проверка на пропущенные значения
    missing_values = df_area.isnull().sum()
    print(f"\nКоличество пропущенных значений:")
    print(missing_values[missing_values > 0] if any(missing_values > 0) else "Пропущенных значений нет")
    
    # Статистика по гранулометрии (целевая переменная)
    granulometry_col = f'Гранулометрия {area_name[-1]}'
    print(f"\nСтатистика по гранулометрии:")
    print(df_area[granulometry_col].describe())
    
    # Временной ряд гранулометрии
    plt.figure(figsize=(12, 6))
    plt.plot(df_area['Время'], df_area[granulometry_col])
    plt.title(f'Временной ряд гранулометрии на участке {area_name}')
    plt.xlabel('Время')
    plt.ylabel('Значение гранулометрии')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'granulometry_time_series_{area_name}.png')
    
    # Распределение гранулометрии
    plt.figure(figsize=(10, 6))
    sns.histplot(df_area[granulometry_col].dropna(), kde=True)
    plt.title(f'Распределение значений гранулометрии на участке {area_name}')
    plt.xlabel('Значение гранулометрии')
    plt.savefig(f'granulometry_distribution_{area_name}.png')
    
    # Обработка выбросов
    numeric_columns = df_area.select_dtypes(include=['float64', 'int64']).columns
    df_area_cleaned = handle_outliers(df_area, numeric_columns, method='iqr')
    
    # Заполнение пропущенных значений с помощью KNN
    df_area_imputed = df_area_cleaned.copy()
    numeric_columns = df_area_imputed.select_dtypes(include=['float64', 'int64']).columns
    
    # Используем KNN импутер вместо простой медианы
    imputer = KNNImputer(n_neighbors=5)
    df_area_imputed[numeric_columns] = imputer.fit_transform(df_area_imputed[numeric_columns])
    
    # Корреляция между параметрами и гранулометрией
    corr_cols = [col for col in df_area_imputed.columns if col not in ['Время']]
    corr_matrix = df_area_imputed[corr_cols].corr()
    
    # Топ-15 параметров, наиболее влияющих на гранулометрию
    corr_with_target = corr_matrix[granulometry_col].drop(granulometry_col).sort_values(ascending=False)
    print(f"\nТоп-15 параметров, коррелирующих с гранулометрией:")
    print(corr_with_target.head(15))
    
    # Визуализация корреляции
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Корреляционная матрица для участка {area_name}')
    plt.tight_layout()
    plt.savefig(f'correlation_matrix_{area_name}.png')
    
    # Визуализация топ-8 коррелирующих с гранулометрией параметров
    top_features = corr_with_target.head(8).index.tolist()
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 4, i)
        plt.scatter(df_area_imputed[feature], df_area_imputed[granulometry_col], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel(granulometry_col)
        plt.title(f'Корреляция: {corr_with_target[feature]:.2f}')
    plt.tight_layout()
    plt.savefig(f'top_correlations_{area_name}.png')
    
    # Добавление лагированных признаков
    # Добавляем лаги по гранулометрии и основным признакам
    # lag_features = [granulometry_col] + [top_features[0]] + [top_features[1]]
    # for feature in lag_features:
    #     for lag in range(1, 4):  # Добавляем лаги от 1 до 3
    #         df_area_imputed[f'{feature}_lag{lag}'] = df_area_imputed[feature].shift(lag)
    
    # # Создаем скользящие средние
    # for feature in lag_features:
    #     for window in [3, 6, 12]:  # Окна в 3, 6 и 12 периодов
    #         df_area_imputed[f'{feature}_rolling{window}'] = df_area_imputed[feature].rolling(window=window).mean()
    
    # Удаляем строки с NaN после добавления лагов
    df_area_imputed = df_area_imputed.dropna()
    
    return df_area_imputed, granulometry_col, corr_with_target

# Расширенная функция для построения и оценки моделей
def build_model(df_area, target_col, top_features, area_name):
    print(f"\n--- Построение модели для участка {area_name} ---")
    
    # Выбор топ-15 признаков для модели + временные признаки
    time_features = ['Hour', 'DayOfWeek', 'IsWeekend', 'TimeOfDay_afternoon', 'TimeOfDay_evening', 'TimeOfDay_night']
    time_features = [col for col in time_features if col in df_area.columns]
    
    # Отбор признаков с учетом лагов и скользящих средних
    selected_features = top_features.head(15).index.tolist() + time_features
    
    # Добавляем лаговые признаки
    lag_features = [col for col in df_area.columns if '_lag' in col or '_rolling' in col]
    selected_features = selected_features + lag_features
    
    # Удалаем дубликаты признаков, если такие есть
    selected_features = list(set(selected_features))
    
    # Убеждаемся, что не включаем целевую переменную в признаки
    if target_col in selected_features:
        selected_features.remove(target_col)
        
    print(f"Выбранные признаки (первые 10): {selected_features[:10]}...")
    print(f"Всего признаков: {len(selected_features)}")
    
    # Подготовка данных
    X = df_area[selected_features].values
    y = df_area[target_col].values
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Расширенный список моделей
    models = {
        'Линейная регрессия': LinearRegression(),
    #     'Ridge': Ridge(alpha=1.0),
    #     'Lasso': Lasso(alpha=0.1),
    #     'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    #     'Случайный лес': RandomForestRegressor(n_estimators=100, random_state=42),
    #     'Градиентный бустинг': GradientBoostingRegressor(n_estimators=100, random_state=42),
    #     'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    #     'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    #     'CatBoost': cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
    #     'AdaBoost': AdaBoostRegressor(random_state=42),
    #     'SVR': SVR(kernel='rbf'),
    #     'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    best_models = {}
    
    # Создаем кроссвалидацию
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nОбучение модели: {name}")
        
        # Если название модели начинается с 'Linear', создаем полиномиальные признаки
        if name in ['Линейная регрессия', 'Ridge', 'Lasso', 'ElasticNet']:
            # Создаем pipeline с полиномиальными признаками для линейных моделей
            steps = [
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('model', model)
            ]
            pipeline = Pipeline(steps)
            
            # Обучаем модель
            pipeline.fit(X_train_scaled, y_train)
            
            # Предсказания
            y_pred = pipeline.predict(X_test_scaled)
            
            # Кросс-валидация
            cv_scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
            cv_rmse = -np.mean(cv_scores)
            
            best_models[name] = pipeline
        else:
            # Для не-линейных моделей просто обучаем модель
            model.fit(X_train_scaled, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test_scaled)
            
            # Кросс-валидация
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
            cv_rmse = -np.mean(cv_scores)
            
            best_models[name] = model
        
        # Оценка модели
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'CV_RMSE': cv_rmse, 'R2': r2}
        
        print(f"RMSE на тесте: {rmse:.4f}")
        print(f"RMSE кросс-валидации: {cv_rmse:.4f}")
        print(f"R² на тесте: {r2:.4f}")
        
        # Важность признаков для моделей, которые их поддерживают
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Признак': selected_features,
                'Важность': importances
            }).sort_values('Важность', ascending=False)
            print("Важность признаков (топ-10):")
            print(feature_importance.head(10))
        
        # Визуализация результатов
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Реальные значения')
        plt.ylabel('Предсказанные значения')
        plt.title(f'{name} - Реальные vs Предсказанные значения (RMSE: {rmse:.4f})')
        plt.savefig(f'{name.lower().replace(" ", "_")}_{area_name}.png')
    
    # Сортировка результатов по RMSE
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['RMSE'])}
    
    # Сравнение моделей
    plt.figure(figsize=(14, 8))
    models_names = list(sorted_results.keys())
    rmse_values = [results[name]['RMSE'] for name in models_names]
    cv_rmse_values = [results[name]['CV_RMSE'] for name in models_names]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, rmse_values, width, label='RMSE на тесте')
    plt.bar(x + width/2, cv_rmse_values, width, label='RMSE кросс-валидации')
    
    plt.ylabel('RMSE (меньше - лучше)')
    plt.title(f'Сравнение моделей для участка {area_name}')
    plt.xticks(x, models_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'model_comparison_{area_name}.png')
    
    # Стекинг моделей (ансамблирование)
    print("\nСоздание ансамбля моделей (стекинг)...")
    
    # Выбираем топ-4 модели по RMSE
    top_models_names = list(sorted_results.keys())[:4]
    base_models = [best_models[name] for name in top_models_names]
    
    # Делаем предсказания каждой базовой модели
    base_predictions = np.column_stack([
        model.predict(X_train_scaled) for model in base_models
    ])
    
    # Обучаем мета-модель
    meta_model = Ridge()
    meta_model.fit(base_predictions, y_train)
    
    # Предсказания базовых моделей на тестовых данных
    test_predictions = np.column_stack([
        model.predict(X_test_scaled) for model in base_models
    ])
    
    # Итоговое предсказание ансамбля
    ensemble_pred = meta_model.predict(test_predictions)
    
    # Оценка ансамбля
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print(f"Ансамбль моделей {', '.join(top_models_names)}")
    print(f"RMSE ансамбля: {ensemble_rmse:.4f}")
    print(f"R² ансамбля: {ensemble_r2:.4f}")
    
    # Добавляем результаты ансамбля
    results['Ансамбль'] = {'RMSE': ensemble_rmse, 'CV_RMSE': None, 'R2': ensemble_r2}
    
    # Визуализация результатов ансамбля
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, ensemble_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Ансамбль - Реальные vs Предсказанные значения (RMSE: {ensemble_rmse:.4f})')
    plt.savefig(f'ensemble_{area_name}.png')
    
    return results, best_models

# Анализ данных по участкам
print("\nАнализ участка 1...")
df_area1_processed, target_col1, corr1 = analyze_area(df_area1, 'участок 1')
print("\nАнализ участка 2...")
df_area2_processed, target_col2, corr2 = analyze_area(df_area2, 'участок 2')

# Построение моделей
print("\nПостроение моделей для участка 1...")
results1, models1 = build_model(df_area1_processed, target_col1, corr1, 'участок 1')

print("\nПостроение моделей для участка 2...")
results2, models2 = build_model(df_area2_processed, target_col2, corr2, 'участок 2')

# Анализ временных паттернов
print("\n--- Анализ временных паттернов ---")

# Анализ почасовых изменений гранулометрии
plt.figure(figsize=(12, 6))
hourly_gran1 = df_area1_processed.groupby('Hour')[target_col1].mean()
hourly_gran2 = df_area2_processed.groupby('Hour')[target_col2].mean()

plt.plot(hourly_gran1.index, hourly_gran1.values, 'b-', label='Участок 1')
plt.plot(hourly_gran2.index, hourly_gran2.values, 'r-', label='Участок 2')
plt.title('Средние значения гранулометрии по часам')
plt.xlabel('Час дня')
plt.ylabel('Среднее значение гранулометрии')
plt.legend()
plt.grid(True)
plt.savefig('hourly_granulometry.png')

# Анализ по дням недели
plt.figure(figsize=(12, 6))
daily_gran1 = df_area1_processed.groupby('DayOfWeek')[target_col1].mean()
daily_gran2 = df_area2_processed.groupby('DayOfWeek')[target_col2].mean()

days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
plt.plot(days, daily_gran1.values, 'b-', label='Участок 1')
plt.plot(days, daily_gran2.values, 'r-', label='Участок 2')
plt.title('Средние значения гранулометрии по дням недели')
plt.xlabel('День недели')
plt.ylabel('Среднее значение гранулометрии')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_granulometry.png')

# Вывод выводов и рекомендаций
print("\n--- Выводы и рекомендации ---")
print("\nТоп-15 параметров, влияющих на гранулометрию на участке 1:")
for i, (feature, corr) in enumerate(corr1.head(15).items(), 1):
    print(f"{i}. {feature}: {corr:.4f}")

print("\nТоп-15 параметров, влияющих на гранулометрию на участке 2:")
for i, (feature, corr) in enumerate(corr2.head(15).items(), 1):
    print(f"{i}. {feature}: {corr:.4f}")

# Сортировка результатов по RMSE
sorted_results1 = {k: v for k, v in sorted(results1.items(), key=lambda item: item[1]['RMSE'])}
sorted_results2 = {k: v for k, v in sorted(results2.items(), key=lambda item: item[1]['RMSE'])}

print("\nЛучшие модели для участка 1:")
for i, (name, metrics) in enumerate(sorted_results1.items(), 1):
    if i <= 3:  # показываем топ-3 модели
        print(f"{i}. {name}: RMSE = {metrics['RMSE']:.4f}, R² = {metrics['R2']:.4f}")

print("\nЛучшие модели для участка 2:")
for i, (name, metrics) in enumerate(sorted_results2.items(), 1):
    if i <= 3:  # показываем топ-3 модели
        print(f"{i}. {name}: RMSE = {metrics['RMSE']:.4f}, R² = {metrics['R2']:.4f}")

print("\nАнализ завершен. Результаты сохранены в виде графиков.") 