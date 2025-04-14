import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import logging
import json
import time
import warnings
from typing import List, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error
import gc
import psutil
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score, r2_score


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


# Функция для отображения использования памяти
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # в МБ
    logger.info(f"Использование памяти: {mem:.2f} МБ")


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



# Функция для создания расширенных признаков
def create_advanced_features(X, top_features, n_features=1000):
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
    selected_features = top_features[:10]  # до 10 лучших признаков
    
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


class Model:
    """
    Класс для обучения и предсказания гранулометрии.
    Включает подбор гиперпараметров, создание продвинутых признаков,
    обучение, оценку и сохранение лучшей модели.
    """
    def __init__(self, data: pd.DataFrame, is_load_model: bool = False):
        """
        Инициализирует модель.
        """
        self.data = data # Сохраняем исходные данные
        self.X = None
        self.y = None
        self.target_column = None  # Колонка с целевой переменной (Гранулометрия)
        self.numeric_cols = None  # Список числовых колонок для обучения модели
        self.best_model = None
        self.final_scaler = None # Scaler для лучшей модели
        self.final_features = None # Список признаков для лучшей модели
        self.top_features = None # Список топ-признаков для генерации расширенных (если нужны)
        self.is_advanced_model = False # Флаг, указывающий, является ли лучшая модель расширенной
        
        # Вызываем подготовку данных сразу при инициализации
        if data is not None:
            self._prepare_data(data)
        
        # Загружаем лучшую модель
        if is_load_model:
            self._load_model_state('model.pkl', 'scaler.pkl', 'features.json', 'top_features.json')

    def _prepare_data(self, data: pd.DataFrame):
        self.data = data

        # TODO: посмотреть что это такое
        # # Используем больше данных для лучшего результата
        # self.data = self.data.sample(frac=0.3, random_state=42)
        # logger.info(f"Используем данные, размер: {len(self.data)} записей (30% от общего объема)")

        # Выбор целевой переменной
        self.target_column = [col for col in self.data.columns if 'Гранулометрия' in col][0]
        logger.info(f"Целевые переменные: {self.target_column}")

        # Очистка данных
        self.data = self.data.dropna(subset=[self.target_column])
        logger.info(f"Размер данных после очистки: {self.data.shape}")

        # Выбираем только числовые колонки
        self.numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.numeric_cols = [col for col in self.numeric_cols if col != self.target_column]
        logger.info(f"Всего числовых колонок (без таргета): {len(self.numeric_cols)}")
        
        # Краткий анализ корреляции признаков с целевой переменной
        logger.info("Анализ корреляции признаков с целевой переменной")
        correlations = []
        for col in self.numeric_cols:
            if col != self.target_column:
                correlation, _ = spearmanr(self.data[col].fillna(0), self.data[self.target_column])
                if not np.isnan(correlation):
                    correlations.append((col, correlation))
            
        # Сортируем признаки по модулю корреляции
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        logger.info(f"Топ-5 признаков по корреляции с целевой переменной:")
        for col, corr in correlations[:5]:
            logger.info(f"{col}: {corr:.4f}")

        self.X = self.data[self.numeric_cols]
        self.y = self.data[self.target_column]

        logger.info(f"Размер данных после очистки: {self.X.shape}")
        logger.info(f"Размер целевой переменной: {self.y.shape}")
        logger.info(f"Данные подготовлены")

    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Вычисляет и возвращает метрики качества модели
        
        Параметры:
        ----------
        y_true : array-like
            Истинные значения целевой переменной
        y_pred : array-like
            Предсказанные значения целевой переменной
            
        Возвращает:
        -----------
        dict: словарь с метриками качества модели
        """
        metrics = {}
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mdae'] = median_absolute_error(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        logger.info("\nМетрики качества модели на тестовой выборке:")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"MDAE: {metrics['mdae']:.4f}")
        logger.info(f"Объясненная дисперсия: {metrics['explained_variance']:.4f}")

        return metrics


    def train(self, data: pd.DataFrame):
        self._prepare_data(data)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        logger.info(f"Размер обучающей выборки: {X_train.shape}")
        logger.info(f"Размер тестовой выборки: {X_test.shape}")

        # Масштабирование признаков
        logger.info("Масштабирование признаков")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        del self.data
        free_memory()
        logger.info("Приступаю к модели...")

        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True, False],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse']
        }

        self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Поиск лучших параметров на небольшой выборке
        logger.info("Поиск оптимальных гиперпараметров...")
        sample_size = min(5000, X_train_scaled.shape[0])
        sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
        X_sample = X_train_scaled[sample_indices]
        y_sample = y_train.iloc[sample_indices]


        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            n_iter=10,
            scoring='neg_root_mean_squared_error',
            cv=3,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_sample, y_sample)
        best_params = random_search.best_params_
        logger.info(f"Лучшие параметры: {best_params}")
        logger.info(f"Лучший RMSE: {-random_search.best_score_:.4f}")
        
        # Обучаем модель с лучшими параметрами
        logger.info("Обучение модели с оптимальными параметрами...")
        self.best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, verbose=1)
        self.best_model.fit(X_train_scaled, y_train)
        logger.info("Оценка модели на тестовых данных...")
        y_pred = self.best_model.predict(X_test_scaled)

        # Вычисление метрик
        metrics = self._calculate_metrics(y_test, y_pred)

        # Кросс-валидация
        logger.info("Выполняем кросс-валидацию на 3 фолдах...")
        cv_scores = cross_val_score(
            self.best_model, 
            X_sample,
            y_sample, 
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        logger.info(f"Средний RMSE по кросс-валидации: {-np.mean(cv_scores):.4f} ± {np.std(-cv_scores):.4f}")
        
        # Извлекаем важность признаков
        importances = self.best_model.feature_importances_
        feature_importance = [(self.numeric_cols[i], importance) for i, importance in enumerate(importances)]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = [feature[0] for feature in feature_importance[:50]]
        
        logger.info(f"Топ-10 признаков по важности:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        # Создаем расширенные признаки
        logger.info("Создание расширенных признаков...")
        X_train_advanced = create_advanced_features(X_train, top_features, n_features=200)
        X_test_advanced = create_advanced_features(X_test, top_features, n_features=200)
        advanced_feature_names = list(X_train_advanced.columns) # Получаем имена новых признаков
        
        # Масштабирование расширенных признаков
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
        metrics_advanced = self._calculate_metrics(y_test, y_pred_advanced)
        
        # Сохраняем лучшую модель (базовую или расширенную) и соответствующий scaler/признаки
        if metrics_advanced['rmse'] < metrics['rmse']:
            logger.info("Модель с расширенными признаками показала лучший результат!")
            self.best_model = advanced_model # Обновляем self.best_model
            self.final_scaler = scaler_advanced # Сохраняем scaler для расширенных признаков
            self.final_features = advanced_feature_names # Сохраняем список расширенных признаков
            self.top_features = top_features # Сохраняем список топ-признаков для генерации расширенных
            self.is_advanced_model = True
            best_overall_metrics = metrics_advanced
            logger.info("Сохранена модель с расширенными признаками: advanced_rf_model.pkl, scaler: advanced_scaler.pkl, features: advanced_features.json, top_features: top_features_for_advanced.json")
        else:
            logger.info("Базовая модель показала лучший результат.")
            self.final_scaler = scaler # Сохраняем scaler для базовой модели
            self.final_features = self.numeric_cols # Сохраняем список базовых признаков
            # self.best_model уже содержит базовую модель
            self.is_advanced_model = False
            self.top_features = None
            best_overall_metrics = metrics
            logger.info("Сохранена базовая модель: improved_rf_model.pkl, scaler: improved_scaler.pkl, features: improved_features.json")
        
        # Сохраняем информацию о важности признаков
        logger.info("\nВажность топ-10 признаков:")
        for name, importance in feature_importance[:10]:
            logger.info(f"{name}: {importance:.4f}")

        # Сохраняем отчеты
        save_metrics_report(metrics, feature_importance, "Full_Model")
        save_metrics_report(metrics_advanced, feature_importance, "Advanced_Model")

        # Устанавливаем порог важности признаков на уровне 75-го процентиля
        # Это означает, что мы выберем только те признаки, важность которых 
        # превышает значение, ниже которого находятся 75% всех признаков
        threshold = np.percentile(self.best_model.feature_importances_, 75)
        logger.info(f"Порог важности признаков: {threshold:.4f}")
        
        # Фильтрация признаков
        important_indices = np.where(self.best_model.feature_importances_ > threshold)[0]
        important_features = [self.final_features[i] for i in important_indices]
        logger.info(f"Отобрано {len(important_features)} важных признаков из {X_train.shape[1]}")
        
        # Используем только важные признаки
        X_train_important = X_train_advanced[important_features]
        X_test_important = X_test_advanced[important_features]
        
        # Масштабирование
        scaler_important = StandardScaler() # Локальная переменная
        X_train_important_scaled = scaler_important.fit_transform(X_train_important)
        X_test_important_scaled = scaler_important.transform(X_test_important)
        
        # Обучаем модель на важных признаках
        logger.info("Обучение модели на отобранных признаках...")
        important_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        important_model.fit(X_train_important_scaled, y_train)
        
        # Оценка модели с отобранными признаками
        y_pred_important = important_model.predict(X_test_important_scaled)
        
        # Метрики для модели с отобранными признаками
        metrics_important = self._calculate_metrics(y_test, y_pred_important)

        if metrics_important['rmse'] < best_overall_metrics['rmse']:
            logger.info("Модель с отобранными признаками показала лучший результат!")
            self.best_model = important_model
            self.final_scaler = scaler_important
            self.final_features = important_features
            best_overall_metrics = metrics_important
        else:
            logger.info("Модель с отобранными признаками показала худший результат.")
        
        # Сохраняем отчеты
        save_metrics_report(metrics_important, feature_importance, "Selected_Features")
        
        logger.info("Обучение модели завершено")

        joblib.dump(self.best_model, 'model.pkl')
        joblib.dump(self.final_scaler, 'scaler.pkl')
        with open('features.json', 'w') as f:
            json.dump(self.final_features, f)
        with open('top_features.json', 'w') as f:
            json.dump(self.top_features, f)

        return self.best_model, best_overall_metrics
    

    def _load_model_state(self, model_path: str, scaler_path: str, features_path: str, top_features_path: str):
        """Загружает состояние модели (модель, scaler, признаки) из файлов."""
        logger.info("Загрузка состояния модели из файлов...")

        self.best_model = joblib.load(model_path)
        self.final_scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            self.final_features = json.load(f)
        with open(top_features_path, 'r') as f:
            self.top_features = json.load(f)
        logger.info("Модель с расширенными признаками успешно загружена.")
        self.is_advanced_model = True


    def predict(self, data: pd.DataFrame):
        """
        Делает предсказание на основе лучшей модели.
        Загружает состояние модели из файлов, если необходимо.
        """
        X_pred = data

        if self.is_advanced_model:
            X_pred_final = create_advanced_features(X_pred, self.top_features, n_features=200)
            X_pred_final = X_pred_final[self.final_features]
        else:
            X_pred_final = X_pred[self.final_features]

        logger.info(f"Выбрано {X_pred_final.columns} признаков для предсказания.")

        # 5. Масштабировать данные
        X_scaled = self.final_scaler.transform(X_pred_final)
        logger.info("Данные масштабированы.")

        # 6. Предсказание
        logger.info("Выполнение предсказания...")
        y_pred = self.best_model.predict(X_scaled)
        logger.info("Предсказание завершено.")
        
        return y_pred
    
if __name__ == "__main__":
    data = pd.read_parquet("data/data_after_analys.parquet")
    
    # Определяем целевую переменную (нужно для stratify и удаления перед predict)
    target_column_name = [col for col in data.columns if 'Гранулометрия' in col][0]
    
    # Разделение данных на обучающую и тестовую выборки (DataFrames)
    # Используем stratify, если целевая переменная категориальная или для регрессии можно разбить на бины
    # В данном случае просто разделим
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
    logger.info(f"Размер обучающей выборки (DataFrame): {data_train.shape}")
    logger.info(f"Размер тестовой выборки (DataFrame): {data_test.shape}")

    model = Model(None, is_load_model=False) # Инициализируем без данных, т.к. train сам подготовит
    model.train(data_train) # Передаем обучающий DataFrame
    
    # Или используем существующий экземпляр
    # Передаем тестовый DataFrame без целевой колонки
    start_time = time.time()
    y_pred = model.predict(data_test.drop(target_column_name, axis=1)) 
    end_time = time.time()
    logger.info(f"Время предсказания: {(end_time - start_time):.10f} секунд для {len(y_pred)} записей")

    # Сравнение первых 10 предсказаний
    y_test_values = data_test[target_column_name] # Получаем y_test из data_test
    logger.info("\nСравнение первых 10 реальных и предсказанных значений:")
    for i in range(min(10, len(y_test_values))):
        print(f"Реальное: {y_test_values.iloc[i]:.4f}, Предсказанное: {y_pred[i]:.4f}")
        
    # Вычисляем RMSE (корень из среднеквадратичной ошибки)
    rmse_value = np.sqrt(mean_squared_error(y_test_values, y_pred))
    print(f"\nИтоговый RMSE на тестовых данных: {rmse_value:.4f}")