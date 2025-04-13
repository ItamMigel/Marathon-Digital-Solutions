import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import median_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import warnings
import gc
import time
import logging
import os
import psutil
import joblib
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
import datetime

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

warnings.filterwarnings('ignore')

# Функция для освобождения памяти
def free_memory():
    gc.collect()

# Функция для отслеживания времени выполнения этапов
def log_time(start_time, message):
    elapsed = time.time() - start_time
    logger.info(f"{message} - Время: {elapsed:.2f} сек")
    return time.time()

# Функция для отображения использования памяти
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # в МБ
    logger.info(f"Использование памяти: {mem:.2f} МБ")

# Функция для оценки оставшегося времени
def estimate_remaining_time(start_time, total_steps, completed_steps):
    elapsed = time.time() - start_time
    if completed_steps == 0:
        return "Оценка невозможна"
    
    time_per_step = elapsed / completed_steps
    remaining_steps = total_steps - completed_steps
    remaining_seconds = time_per_step * remaining_steps
    
    return str(datetime.timedelta(seconds=int(remaining_seconds)))

# Функция для сохранения результатов в отчет
def save_metrics_report(metrics, feature_importance, model_name="RandomForest"):
    """Сохраняет метрики и важность признаков в отчет"""
    with open(f"report_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"ОТЧЕТ ПО МОДЕЛИ: {model_name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("МЕТРИКИ КАЧЕСТВА:\n")
        f.write("-"*50 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
        
        f.write("\nВАЖНОСТЬ ТОП-20 ПРИЗНАКОВ:\n")
        f.write("-"*50 + "\n")
        for i, (name, importance) in enumerate(feature_importance[:20], 1):
            f.write(f"{i}. {name}: {importance:.4f}\n")
            
    logger.info(f"Отчет сохранен в файл report_{model_name}.txt")

# Засекаем общее время выполнения
total_start_time = time.time()

# Статус выполнения скрипта
def print_progress(current_step, total_steps=5):
    percent = (current_step / total_steps) * 100
    eta = estimate_remaining_time(total_start_time, total_steps, current_step)
    logger.info(f"ПРОГРЕСС: {current_step}/{total_steps} шагов ({percent:.1f}%) - Примерное время до завершения: {eta}")

try:
    # Шаг 1/5: Загрузка и анализ данных
    logger.info("======= ШАГ 1/5: ЗАГРУЗКА И АНАЛИЗ ДАННЫХ =======")
    print_progress(0, 5)
    start_time = time.time()
    
    # Загрузка данных
    data = pd.read_parquet('data/data_after_analys.parquet')
    # Используем больше данных для лучшего результата
    data = data.sample(frac=0.3, random_state=42)
    logger.info(f"Используем данные, размер: {len(data)} записей (30% от общего объема)")
    
    # Проверка целевых переменных
    granulometry_columns = [col for col in data.columns if 'Гранулометрия' in col]
    logger.info(f"Целевые переменные: {granulometry_columns}")
    
    # Выбор целевой переменной
    target_column = granulometry_columns[0]
    logger.info(f"Используем целевую переменную: {target_column}")
    
    # Очистка данных
    data = data.dropna(subset=[target_column])
    logger.info(f"Размер данных после очистки: {data.shape}")
    
    # Выбираем только числовые колонки
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    logger.info(f"Всего числовых колонок (без таргета): {len(numeric_cols)}")
    
    # Добавим проверку на наличие столбцов datetime в данных
    datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        logger.info(f"Обнаружены столбцы с типом datetime: {datetime_cols}")
        logger.info("Эти столбцы будут исключены из анализа")
    
    # Краткий анализ корреляции признаков с целевой переменной
    logger.info("Анализ корреляции признаков с целевой переменной")
    correlations = []
    for col in numeric_cols:
        if col != target_column:
            correlation, _ = spearmanr(data[col].fillna(0), data[target_column])
            if not np.isnan(correlation):
                correlations.append((col, correlation))
    
    # Сортируем признаки по модулю корреляции
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    logger.info(f"Топ-5 признаков по корреляции с целевой переменной:")
    for col, corr in correlations[:5]:
        logger.info(f"{col}: {corr:.4f}")
    
    X = data[numeric_cols]
    y = data[target_column]
    
    log_time(start_time, "Данные загружены и подготовлены")
    print_progress(1, 5)
    
    # Шаг 2/5: Подготовка данных для обучения
    logger.info("======= ШАГ 2/5: ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ =======")
    start_time = time.time()
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Размер обучающей выборки: {X_train.shape}")
    logger.info(f"Размер тестовой выборки: {X_test.shape}")
    
    # Масштабирование признаков
    logger.info("Масштабирование признаков")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Освобождаем память от ненужных данных
    del data
    free_memory()
    log_time(start_time, "Данные подготовлены для обучения")
    print_progress(2, 5)
    
    # Шаг 3/5: Оптимизация и обучение модели
    logger.info("======= ШАГ 3/5: ОПТИМИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ =======")
    start_time = time.time()
    
    # Определяем пространство поиска гиперпараметров - расширенное для лучшей точности
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse']
    }
    
    # Создаем базовую модель
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Поиск лучших параметров на небольшой выборке
    logger.info("Поиск оптимальных гиперпараметров...")
    sample_size = min(5000, X_train_scaled.shape[0])
    sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
    X_sample = X_train_scaled[sample_indices]
    y_sample = y_train.iloc[sample_indices]
    
    # Увеличиваем количество итераций поиска
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,  # Увеличиваем число итераций с 5 до 10
        scoring='neg_root_mean_squared_error',
        cv=3,  # Увеличиваем число фолдов с 2 до 3
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_sample, y_sample)
    best_params = random_search.best_params_
    logger.info(f"Лучшие параметры: {best_params}")
    logger.info(f"Лучший RMSE: {-random_search.best_score_:.4f}")
    
    # Обучаем модель с лучшими параметрами
    logger.info("Обучение модели с оптимальными параметрами...")
    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, verbose=1)
    best_model.fit(X_train_scaled, y_train)
    
    # Предсказания и оценка
    logger.info("Оценка модели на тестовых данных...")
    y_pred = best_model.predict(X_test_scaled)
    
    # Вычисление метрик
    metrics = {}
    metrics['mse'] = mean_squared_error(y_test, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_test, y_pred)
    metrics['mae'] = mean_absolute_error(y_test, y_pred)
    metrics['mdae'] = median_absolute_error(y_test, y_pred)
    metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
    
    logger.info("\nМетрики качества модели на тестовой выборке:")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"R²: {metrics['r2']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    
    # Кросс-валидация
    logger.info("Выполняем кросс-валидацию на 3 фолдах...")
    cv_scores = cross_val_score(
        best_model, 
        X_sample,
        y_sample, 
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    logger.info(f"Средний RMSE по кросс-валидации: {-np.mean(cv_scores):.4f} ± {np.std(-cv_scores):.4f}")
    
    # Извлекаем важность признаков
    importances = best_model.feature_importances_
    feature_importance = [(numeric_cols[i], importance) for i, importance in enumerate(importances)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    top_features = [feature[0] for feature in feature_importance[:50]]
    
    logger.info(f"Топ-10 признаков по важности:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        logger.info(f"{i+1}. {feature}: {importance:.4f}")
    
    # Создаем расширенные признаки
    logger.info("Создание расширенных признаков...")
    X_train_advanced = create_advanced_features(X_train, top_features, n_features=100)
    X_test_advanced = create_advanced_features(X_test, top_features, n_features=100)
    
    # Масштабирование новых признаков
    logger.info("Масштабирование расширенных признаков...")
    scaler_advanced = StandardScaler()
    X_train_advanced_scaled = scaler_advanced.fit_transform(X_train_advanced)
    X_test_advanced_scaled = scaler_advanced.transform(X_test_advanced)
    
    # Обучение модели на расширенных признаках
    logger.info("Обучение модели на расширенных признаках...")
    advanced_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, verbose=1)
    advanced_model.fit(X_train_advanced_scaled, y_train)
    
    # Оценка модели с расширенными признаками
    logger.info("Оценка модели с расширенными признаками...")
    y_pred_advanced = advanced_model.predict(X_test_advanced_scaled)
    
    # Вычисление метрик для модели с расширенными признаками
    metrics_advanced = {}
    metrics_advanced['mse'] = mean_squared_error(y_test, y_pred_advanced)
    metrics_advanced['rmse'] = np.sqrt(metrics_advanced['mse'])
    metrics_advanced['r2'] = r2_score(y_test, y_pred_advanced)
    metrics_advanced['mae'] = mean_absolute_error(y_test, y_pred_advanced)
    metrics_advanced['mdae'] = median_absolute_error(y_test, y_pred_advanced)
    metrics_advanced['explained_variance'] = explained_variance_score(y_test, y_pred_advanced)
    
    logger.info("\nМетрики качества модели с расширенными признаками:")
    logger.info(f"RMSE: {metrics_advanced['rmse']:.4f}")
    logger.info(f"R²: {metrics_advanced['r2']:.4f}")
    logger.info(f"MAE: {metrics_advanced['mae']:.4f}")
    
    # Сохраняем лучшую модель (базовую или расширенную)
    if metrics_advanced['rmse'] < metrics['rmse']:
        logger.info("Модель с расширенными признаками показала лучший результат!")
        best_overall_model = advanced_model
        best_overall_metrics = metrics_advanced
        joblib.dump(advanced_model, 'advanced_rf_model.pkl')
        logger.info("Сохранена модель с расширенными признаками: advanced_rf_model.pkl")
    else:
        logger.info("Базовая модель показала лучший результат.")
        best_overall_model = best_model
        best_overall_metrics = metrics
        joblib.dump(best_model, 'improved_rf_model.pkl')
        logger.info("Сохранена базовая модель: improved_rf_model.pkl")
    
    # Сохраняем информацию о важности признаков
    logger.info("\nВажность топ-10 признаков:")
    for name, importance in feature_importance[:10]:
        logger.info(f"{name}: {importance:.4f}")
    
    # Сохраняем отчеты
    save_metrics_report(metrics, feature_importance, "Full_Model")
    save_metrics_report(metrics_advanced, feature_importance, "Advanced_Model")
    
    log_time(start_time, "Модель обучена и оценена")
    print_progress(3, 5)
    
    # Шаг 4/5: Отбор признаков (упрощенный)
    logger.info("======= ШАГ 4/5: ОТБОР ПРИЗНАКОВ =======")
    start_time = time.time()
    
    # Выбор только важных признаков (верхний квартиль)
    threshold = np.percentile(best_model.feature_importances_, 75)
    logger.info(f"Порог важности признаков: {threshold:.4f}")
    
    # Фильтрация признаков
    important_indices = np.where(best_model.feature_importances_ > threshold)[0]
    important_features = [X_train.columns[i] for i in important_indices]
    logger.info(f"Отобрано {len(important_features)} важных признаков из {X_train.shape[1]}")
    
    # Используем только важные признаки
    X_train_important = X_train[important_features]
    X_test_important = X_test[important_features]
    
    # Масштабирование
    scaler_important = StandardScaler()
    X_train_important_scaled = scaler_important.fit_transform(X_train_important)
    X_test_important_scaled = scaler_important.transform(X_test_important)
    
    # Обучаем модель на важных признаках
    logger.info("Обучение модели на отобранных признаках...")
    important_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    important_model.fit(X_train_important_scaled, y_train)
    
    # Оценка модели с отобранными признаками
    y_pred_important = important_model.predict(X_test_important_scaled)
    
    # Метрики для модели с отобранными признаками
    metrics_important = {}
    metrics_important['mse'] = mean_squared_error(y_test, y_pred_important)
    metrics_important['rmse'] = np.sqrt(metrics_important['mse'])
    metrics_important['r2'] = r2_score(y_test, y_pred_important)
    metrics_important['mae'] = mean_absolute_error(y_test, y_pred_important)
    metrics_important['mdae'] = median_absolute_error(y_test, y_pred_important)
    metrics_important['explained_variance'] = explained_variance_score(y_test, y_pred_important)
    
    logger.info("\nМетрики модели с отобранными признаками:")
    logger.info(f"RMSE: {metrics_important['rmse']:.4f}")
    logger.info(f"R²: {metrics_important['r2']:.4f}")
    logger.info(f"MAE: {metrics_important['mae']:.4f}")
    
    # Сохраняем отчеты
    save_metrics_report(metrics_important, feature_importance, "Selected_Features")
    
    log_time(start_time, "Отбор признаков завершен")
    print_progress(4, 5)
    
    # Шаг 5/5: Создание визуализаций
    logger.info("======= ШАГ 5/5: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ =======")
    start_time = time.time()
    
    # 1. Визуализация фактических vs предсказанных значений
    logger.info("Создание визуализации 1/4: сравнение фактических и предсказанных значений")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Сравнение фактических и предсказанных значений')
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    
    # 2. Визуализация важности признаков (топ-15)
    logger.info("Создание визуализации 2/4: важность признаков")
    plt.figure(figsize=(12, 8))
    feature_names = [name for name, _ in feature_importance[:15]]
    feature_values = [importance for _, importance in feature_importance[:15]]
    
    plt.barh(range(len(feature_names)), feature_values, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.title('Топ-15 важных признаков')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # 3. Визуализация распределения ошибок
    logger.info("Создание визуализации 3/4: распределение ошибок")
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Ошибка')
    plt.ylabel('Частота')
    plt.title('Распределение ошибок модели')
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    
    # 4. Корреляционная матрица топ признаков
    logger.info("Создание визуализации 4/4: корреляционная матрица")
    top_feature_names = [name for name, _ in feature_importance[:10]]
    top_features_df = X_test[top_feature_names]
    plt.figure(figsize=(12, 8))
    corr_matrix = top_features_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Корреляционная матрица топ-10 признаков')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    log_time(start_time, "Визуализации сохранены в файлы")
    print_progress(5, 5)
    
    # Выводим общее время выполнения
    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("\n========== ОТЧЕТ О ВЫПОЛНЕНИИ ==========")
    logger.info(f"Общее время выполнения скрипта: {int(hours)}:{int(minutes):02}:{int(seconds):02}")
    logger.info("Базовая модель:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    
    logger.info("Модель с отобранными признаками:")
    logger.info(f"  RMSE: {metrics_important['rmse']:.4f}")
    logger.info(f"  R²: {metrics_important['r2']:.4f}")
    logger.info(f"  MAE: {metrics_important['mae']:.4f}")
    
    logger.info("Модель с расширенными признаками:")
    logger.info(f"  RMSE: {metrics_advanced['rmse']:.4f}")
    logger.info(f"  R²: {metrics_advanced['r2']:.4f}")
    logger.info(f"  MAE: {metrics_advanced['mae']:.4f}")
    
    logger.info("Отчеты сохранены в файлах")
    logger.info("Визуализации сохранены в папке проекта")
    logger.info("==========================================")

except Exception as e:
    logger.error(f"Произошла ошибка: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    logger.info("Выполнение скрипта прервано из-за ошибки")

# Функция для кросс-валидации модели
def cross_validate_model(X, y, model, cv=3):
    """Проводит кросс-валидацию модели и возвращает средние метрики"""
    logger.info(f"Начало кросс-валидации с {cv} фолдами")
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    rmse_scores = []
    r2_scores = []
    mae_scores = []
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"Обучение на фолде {i+1}/{cv}...")
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train_cv, y_train_cv)
        
        logger.info(f"Предсказание на фолде {i+1}/{cv}...")
        y_pred_cv = model.predict(X_val_cv)
        
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
        r2 = r2_score(y_val_cv, y_pred_cv)
        mae = mean_absolute_error(y_val_cv, y_pred_cv)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mae_scores.append(mae)
        
        logger.info(f"Фолд {i+1}/{cv}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")
    
    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_mae = np.mean(mae_scores)
    
    logger.info(f"Средние результаты кросс-валидации:")
    logger.info(f"RMSE: {mean_rmse:.4f} ± {np.std(rmse_scores):.4f}")
    logger.info(f"R²: {mean_r2:.4f} ± {np.std(r2_scores):.4f}")
    logger.info(f"MAE: {mean_mae:.4f} ± {np.std(mae_scores):.4f}")
    
    return {
        'mean_rmse': mean_rmse,
        'mean_r2': mean_r2,
        'mean_mae': mean_mae,
        'std_rmse': np.std(rmse_scores),
        'std_r2': np.std(r2_scores),
        'std_mae': np.std(mae_scores)
    }

# Функция для создания расширенных признаков
def create_advanced_features(X, top_features, n_features=100):
    """
    Создает расширенные признаки на основе исходных данных для повышения точности
    
    Параметры:
    ----------
    X : DataFrame
        Исходные признаки
    top_features : list
        Список наиболее важных признаков (по корреляции или важности)
    n_features : int
        Максимальное количество новых признаков для создания
    
    Возвращает:
    -----------
    DataFrame с исходными и новыми признаками
    """
    logger.info(f"Создание расширенных признаков на основе {len(top_features)} важных признаков")
    X_new = X.copy()
    feature_count = 0
    
    # Выбираем топ признаки для преобразований
    selected_features = top_features[:10]  # Увеличиваем до 10 лучших признаков
    
    # 1. Полиномиальные признаки разных степеней
    logger.info("Создание полиномиальных признаков")
    for feature in selected_features:
        if feature_count >= n_features:
            break
            
        # Квадратичные признаки
        X_new[f"{feature}_squared"] = X[feature] ** 2
        feature_count += 1
        
        # Кубические признаки
        X_new[f"{feature}_cubed"] = X[feature] ** 3
        feature_count += 1
        
        # Признаки 4-й степени
        X_new[f"{feature}_power4"] = X[feature] ** 4
        feature_count += 1
    
    # 2. Тригонометрические преобразования
    logger.info("Создание тригонометрических признаков")
    for feature in selected_features[:5]:
        if feature_count >= n_features:
            break
            
        # Нормализуем для тригонометрических функций
        norm_feature = (X[feature] - X[feature].mean()) / (X[feature].std() + 1e-8)
        
        # Синус признаки
        X_new[f"{feature}_sin"] = np.sin(norm_feature)
        feature_count += 1
        
        # Косинус признаки
        X_new[f"{feature}_cos"] = np.cos(norm_feature)
        feature_count += 1
    
    # 3. Взаимодействия между признаками
    logger.info("Создание взаимодействий между признаками")
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            if feature_count >= n_features:
                break
                
            feat1 = selected_features[i]
            feat2 = selected_features[j]
            
            # Произведение признаков
            X_new[f"inter_{feat1}_{feat2}"] = X[feat1] * X[feat2]
            feature_count += 1
            
            # Отношение признаков (с защитой от деления на ноль)
            X_new[f"ratio_{feat1}_{feat2}"] = X[feat1] / (X[feat2] + 1e-8)
            feature_count += 1
            
            # Сумма признаков
            X_new[f"sum_{feat1}_{feat2}"] = X[feat1] + X[feat2]
            feature_count += 1
            
            # Разность признаков
            X_new[f"diff_{feat1}_{feat2}"] = X[feat1] - X[feat2]
            feature_count += 1
    
    # 4. Логарифмические и экспоненциальные признаки
    logger.info("Создание логарифмических и экспоненциальных признаков")
    for feature in selected_features:
        if feature_count >= n_features:
            break
            
        # Для логарифма нужны положительные значения
        min_val = X[feature].min()
        if min_val <= 0:
            X_new[f"{feature}_log"] = np.log(X[feature] - min_val + 1)
        else:
            X_new[f"{feature}_log"] = np.log(X[feature] + 1)
        feature_count += 1
        
        # Экспонента с нормализацией
        norm_feature = (X[feature] - X[feature].min()) / (X[feature].max() - X[feature].min() + 1e-8)
        X_new[f"{feature}_exp"] = np.exp(norm_feature) - 1  # -1 чтобы избежать очень больших значений
        feature_count += 1
    
    # 5. Трехсторонние взаимодействия для топ-3 признаков
    if len(selected_features) >= 3:
        logger.info("Создание трехсторонних взаимодействий")
        top3 = selected_features[:3]
        for i in range(len(top3)):
            for j in range(i+1, len(top3)):
                for k in range(j+1, len(top3)):
                    if feature_count >= n_features:
                        break
                        
                    f1, f2, f3 = top3[i], top3[j], top3[k]
                    X_new[f"three_way_{f1}_{f2}_{f3}"] = X[f1] * X[f2] * X[f3]
                    feature_count += 1
    
    # Заполняем пропуски и обрабатываем бесконечности
    for col in X_new.columns:
        X_new[col] = X_new[col].replace([np.inf, -np.inf], np.nan)
        X_new[col] = X_new[col].fillna(X_new[col].median())
    
    logger.info(f"Создано {feature_count} дополнительных признаков. Итоговое количество: {X_new.shape[1]}")
    return X_new

# Функция main для запуска всего процесса
def main():
    try:
        # Шаг 1/5: Загрузка и анализ данных
        logger.info("======= ШАГ 1/5: ЗАГРУЗКА И АНАЛИЗ ДАННЫХ =======")
        print_progress(0, 5)
        start_time = time.time()
        
        # Загрузка данных
        data = pd.read_parquet('data/data_after_analys.parquet')
        # Используем больше данных для лучшего результата
        data = data.sample(frac=0.3, random_state=42)
        logger.info(f"Используем данные, размер: {len(data)} записей (30% от общего объема)")
        
        # Проверка целевых переменных
        granulometry_columns = [col for col in data.columns if 'Гранулометрия' in col]
        logger.info(f"Целевые переменные: {granulometry_columns}")
        
        # Выбор целевой переменной
        target_column = granulometry_columns[0]
        logger.info(f"Используем целевую переменную: {target_column}")
        
        # Очистка данных
        data = data.dropna(subset=[target_column])
        logger.info(f"Размер данных после очистки: {data.shape}")
        
        # Выбираем только числовые колонки
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_column]
        
        logger.info(f"Всего числовых колонок (без таргета): {len(numeric_cols)}")
        
        # Добавим проверку на наличие столбцов datetime в данных
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            logger.info(f"Обнаружены столбцы с типом datetime: {datetime_cols}")
            logger.info("Эти столбцы будут исключены из анализа")
        
        # Краткий анализ корреляции признаков с целевой переменной
        logger.info("Анализ корреляции признаков с целевой переменной")
        correlations = []
        for col in numeric_cols:
            if col != target_column:
                correlation, _ = spearmanr(data[col].fillna(0), data[target_column])
                if not np.isnan(correlation):
                    correlations.append((col, correlation))
        
        # Сортируем признаки по модулю корреляции
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        logger.info(f"Топ-5 признаков по корреляции с целевой переменной:")
        for col, corr in correlations[:5]:
            logger.info(f"{col}: {corr:.4f}")
        
        X = data[numeric_cols]
        y = data[target_column]
        
        log_time(start_time, "Данные загружены и подготовлены")
        print_progress(1, 5)
        
        # Шаг 2/5: Подготовка данных для обучения
        logger.info("======= ШАГ 2/5: ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ =======")
        start_time = time.time()
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Размер обучающей выборки: {X_train.shape}")
        logger.info(f"Размер тестовой выборки: {X_test.shape}")
        
        # Масштабирование признаков
        logger.info("Масштабирование признаков")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Освобождаем память от ненужных данных
        del data
        free_memory()
        log_time(start_time, "Данные подготовлены для обучения")
        print_progress(2, 5)
        
        # Шаг 3/5: Оптимизация и обучение модели
        logger.info("======= ШАГ 3/5: ОПТИМИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ =======")
        start_time = time.time()
        
        # Определяем пространство поиска гиперпараметров - расширенное для лучшей точности
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True, False],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse']
        }
        
        # Создаем базовую модель
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Поиск лучших параметров на небольшой выборке
        logger.info("Поиск оптимальных гиперпараметров...")
        sample_size = min(5000, X_train_scaled.shape[0])
        sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
        X_sample = X_train_scaled[sample_indices]
        y_sample = y_train.iloc[sample_indices]
        
        # Увеличиваем количество итераций поиска
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,  # Увеличиваем число итераций с 5 до 10
            scoring='neg_root_mean_squared_error',
            cv=3,  # Увеличиваем число фолдов с 2 до 3
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_sample, y_sample)
        best_params = random_search.best_params_
        logger.info(f"Лучшие параметры: {best_params}")
        logger.info(f"Лучший RMSE: {-random_search.best_score_:.4f}")
        
        # Обучаем модель с лучшими параметрами
        logger.info("Обучение модели с оптимальными параметрами...")
        best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, verbose=1)
        best_model.fit(X_train_scaled, y_train)
        
        # Предсказания и оценка
        logger.info("Оценка модели на тестовых данных...")
        y_pred = best_model.predict(X_test_scaled)
        
        # Вычисление метрик
        metrics = {}
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['mdae'] = median_absolute_error(y_test, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
        
        logger.info("\nМетрики качества модели на тестовой выборке:")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        
        # Кросс-валидация
        logger.info("Выполняем кросс-валидацию на 3 фолдах...")
        cv_scores = cross_val_score(
            best_model, 
            X_sample,
            y_sample, 
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        logger.info(f"Средний RMSE по кросс-валидации: {-np.mean(cv_scores):.4f} ± {np.std(-cv_scores):.4f}")
        
        # Извлекаем важность признаков
        importances = best_model.feature_importances_
        feature_importance = [(numeric_cols[i], importance) for i, importance in enumerate(importances)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        top_features = [feature[0] for feature in feature_importance[:50]]
        
        logger.info(f"Топ-10 признаков по важности:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        # Создаем расширенные признаки
        logger.info("Создание расширенных признаков...")
        X_train_advanced = create_advanced_features(X_train, top_features, n_features=100)
        X_test_advanced = create_advanced_features(X_test, top_features, n_features=100)
        
        # Масштабирование новых признаков
        logger.info("Масштабирование расширенных признаков...")
        scaler_advanced = StandardScaler()
        X_train_advanced_scaled = scaler_advanced.fit_transform(X_train_advanced)
        X_test_advanced_scaled = scaler_advanced.transform(X_test_advanced)
        
        # Обучение модели на расширенных признаках
        logger.info("Обучение модели на расширенных признаках...")
        advanced_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, verbose=1)
        advanced_model.fit(X_train_advanced_scaled, y_train)
        
        # Оценка модели с расширенными признаками
        logger.info("Оценка модели с расширенными признаками...")
        y_pred_advanced = advanced_model.predict(X_test_advanced_scaled)
        
        # Вычисление метрик для модели с расширенными признаками
        metrics_advanced = {}
        metrics_advanced['mse'] = mean_squared_error(y_test, y_pred_advanced)
        metrics_advanced['rmse'] = np.sqrt(metrics_advanced['mse'])
        metrics_advanced['r2'] = r2_score(y_test, y_pred_advanced)
        metrics_advanced['mae'] = mean_absolute_error(y_test, y_pred_advanced)
        metrics_advanced['mdae'] = median_absolute_error(y_test, y_pred_advanced)
        metrics_advanced['explained_variance'] = explained_variance_score(y_test, y_pred_advanced)
        
        logger.info("\nМетрики качества модели с расширенными признаками:")
        logger.info(f"RMSE: {metrics_advanced['rmse']:.4f}")
        logger.info(f"R²: {metrics_advanced['r2']:.4f}")
        logger.info(f"MAE: {metrics_advanced['mae']:.4f}")
        
        # Сохраняем лучшую модель (базовую или расширенную)
        if metrics_advanced['rmse'] < metrics['rmse']:
            logger.info("Модель с расширенными признаками показала лучший результат!")
            best_overall_model = advanced_model
            best_overall_metrics = metrics_advanced
            joblib.dump(advanced_model, 'advanced_rf_model.pkl')
            logger.info("Сохранена модель с расширенными признаками: advanced_rf_model.pkl")
        else:
            logger.info("Базовая модель показала лучший результат.")
            best_overall_model = best_model
            best_overall_metrics = metrics
            joblib.dump(best_model, 'improved_rf_model.pkl')
            logger.info("Сохранена базовая модель: improved_rf_model.pkl")
        
        # Сохраняем информацию о важности признаков
        logger.info("\nВажность топ-10 признаков:")
        for name, importance in feature_importance[:10]:
            logger.info(f"{name}: {importance:.4f}")
        
        # Сохраняем отчеты
        save_metrics_report(metrics, feature_importance, "Full_Model")
        save_metrics_report(metrics_advanced, feature_importance, "Advanced_Model")
        
        log_time(start_time, "Модель обучена и оценена")
        print_progress(3, 5)
        
        # Шаг 4/5: Отбор признаков (упрощенный)
        logger.info("======= ШАГ 4/5: ОТБОР ПРИЗНАКОВ =======")
        start_time = time.time()
        
        # Выбор только важных признаков (верхний квартиль)
        threshold = np.percentile(best_model.feature_importances_, 75)
        logger.info(f"Порог важности признаков: {threshold:.4f}")
        
        # Фильтрация признаков
        important_indices = np.where(best_model.feature_importances_ > threshold)[0]
        important_features = [X_train.columns[i] for i in important_indices]
        logger.info(f"Отобрано {len(important_features)} важных признаков из {X_train.shape[1]}")
        
        # Используем только важные признаки
        X_train_important = X_train[important_features]
        X_test_important = X_test[important_features]
        
        # Масштабирование
        scaler_important = StandardScaler()
        X_train_important_scaled = scaler_important.fit_transform(X_train_important)
        X_test_important_scaled = scaler_important.transform(X_test_important)
        
        # Обучаем модель на важных признаках
        logger.info("Обучение модели на отобранных признаках...")
        important_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        important_model.fit(X_train_important_scaled, y_train)
        
        # Оценка модели с отобранными признаками
        y_pred_important = important_model.predict(X_test_important_scaled)
        
        # Метрики для модели с отобранными признаками
        metrics_important = {}
        metrics_important['mse'] = mean_squared_error(y_test, y_pred_important)
        metrics_important['rmse'] = np.sqrt(metrics_important['mse'])
        metrics_important['r2'] = r2_score(y_test, y_pred_important)
        metrics_important['mae'] = mean_absolute_error(y_test, y_pred_important)
        metrics_important['mdae'] = median_absolute_error(y_test, y_pred_important)
        metrics_important['explained_variance'] = explained_variance_score(y_test, y_pred_important)
        
        logger.info("\nМетрики модели с отобранными признаками:")
        logger.info(f"RMSE: {metrics_important['rmse']:.4f}")
        logger.info(f"R²: {metrics_important['r2']:.4f}")
        logger.info(f"MAE: {metrics_important['mae']:.4f}")
        
        # Сохраняем отчеты
        save_metrics_report(metrics_important, feature_importance, "Selected_Features")
        
        log_time(start_time, "Отбор признаков завершен")
        print_progress(4, 5)
        
        # Шаг 5/5: Создание визуализаций
        logger.info("======= ШАГ 5/5: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ =======")
        start_time = time.time()
        
        # 1. Визуализация фактических vs предсказанных значений
        logger.info("Создание визуализации 1/4: сравнение фактических и предсказанных значений")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказанные значения')
        plt.title('Сравнение фактических и предсказанных значений')
        plt.tight_layout()
        plt.savefig('prediction_comparison.png')
        
        # 2. Визуализация важности признаков (топ-15)
        logger.info("Создание визуализации 2/4: важность признаков")
        plt.figure(figsize=(12, 8))
        feature_names = [name for name, _ in feature_importance[:15]]
        feature_values = [importance for _, importance in feature_importance[:15]]
        
        plt.barh(range(len(feature_names)), feature_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Важность')
        plt.ylabel('Признак')
        plt.title('Топ-15 важных признаков')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # 3. Визуализация распределения ошибок
        logger.info("Создание визуализации 3/4: распределение ошибок")
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Ошибка')
        plt.ylabel('Частота')
        plt.title('Распределение ошибок модели')
        plt.tight_layout()
        plt.savefig('error_distribution.png')
        
        # 4. Корреляционная матрица топ признаков
        logger.info("Создание визуализации 4/4: корреляционная матрица")
        top_feature_names = [name for name, _ in feature_importance[:10]]
        top_features_df = X_test[top_feature_names]
        plt.figure(figsize=(12, 8))
        corr_matrix = top_features_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Корреляционная матрица топ-10 признаков')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        
        log_time(start_time, "Визуализации сохранены в файлы")
        print_progress(5, 5)
        
        # Выводим общее время выполнения
        total_elapsed = time.time() - total_start_time
        hours, remainder = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("\n========== ОТЧЕТ О ВЫПОЛНЕНИИ ==========")
        logger.info(f"Общее время выполнения скрипта: {int(hours)}:{int(minutes):02}:{int(seconds):02}")
        logger.info("Базовая модель:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        
        logger.info("Модель с отобранными признаками:")
        logger.info(f"  RMSE: {metrics_important['rmse']:.4f}")
        logger.info(f"  R²: {metrics_important['r2']:.4f}")
        logger.info(f"  MAE: {metrics_important['mae']:.4f}")
        
        logger.info("Модель с расширенными признаками:")
        logger.info(f"  RMSE: {metrics_advanced['rmse']:.4f}")
        logger.info(f"  R²: {metrics_advanced['r2']:.4f}")
        logger.info(f"  MAE: {metrics_advanced['mae']:.4f}")
        
        logger.info("Отчеты сохранены в файлах")
        logger.info("Визуализации сохранены в папке проекта")
        logger.info("==========================================")

    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Выполнение скрипта прервано из-за ошибки")

# Запускаем основной код, только если файл запущен напрямую
if __name__ == "__main__":
    main() 