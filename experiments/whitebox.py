import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
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

# Новый импорт для LightAutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filename='model.log',
    filemode='a',
    encoding='utf-8'
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
def save_metrics_report(metrics, feature_importance, model_name="LightAutoML_Model"):
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
        # Если важность признаков не рассчитывалась, сохраняем пустой список или дефолт
        if feature_importance:
            for i, (name, importance) in enumerate(feature_importance[:20], 1):
                f.write(f"{i}. {name}: {importance:.4f}\n")
        else:
            f.write("Нет данных по важности признаков.\n")
            
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
    Теперь используется LightAutoML с whitebox для автоматизированного обучения модели.
    """
    def __init__(self, data: pd.DataFrame, is_load_model: bool = False):
        self.data = data  # Сохраняем исходные данные
        self.X = None
        self.y = None
        self.target_column = None  # Колонка с целевой переменной (Гранулометрия)
        self.numeric_cols = None  # Список числовых колонок для обучения модели
        self.best_model = None
        self.final_features = None  # Список признаков для лучшей модели
        self.top_features = None  # Список топ-признаков для генерации расширенных (если нужны)

        # Вызываем подготовку данных сразу при инициализации
        if data is not None:
            self._prepare_data(data)

        # Загружаем лучшую модель, если требуется
        if is_load_model:
            self._load_model_state('model.pkl', 'features.json', 'top_features.json')

    def _prepare_data(self, data: pd.DataFrame):
        self.data = data

        # Выбор целевой переменной
        self.target_column = [col for col in self.data.columns if 'Гранулометрия' in col][0]
        logger.info(f"Целевая переменная: {self.target_column}")

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
            correlation, _ = spearmanr(self.data[col].fillna(0), self.data[self.target_column])
            if not np.isnan(correlation):
                correlations.append((col, correlation))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        logger.info(f"Топ-5 признаков по корреляции с целевой переменной:")
        for col, corr in correlations[:5]:
            logger.info(f"{col}: {corr:.4f}")

        self.X = self.data[self.numeric_cols]
        self.y = self.data[self.target_column]
        logger.info(f"Размер данных для обучения: {self.data.shape}")
        logger.info(f"Размер целевой переменной: {self.y.shape}")
        logger.info("Данные подготовлены")

    def _calculate_metrics(self, y_true, y_pred):
        """
        Вычисляет и возвращает метрики качества модели
        """
        metrics = {}
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mdae'] = median_absolute_error(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)

        logger.info("Метрики качества модели на тестовой выборке:")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"MDAE: {metrics['mdae']:.4f}")
        logger.info(f"Объясненная дисперсия: {metrics['explained_variance']:.4f}")

        return metrics

    def train(self, data: pd.DataFrame):
        self._prepare_data(data)

        # Разделение данных на обучающую и тестовую выборки
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        logger.info(f"Размер обучающей выборки: {train_data.shape}")
        logger.info(f"Размер тестовой выборки: {test_data.shape}")

        # Обучение модели LightAutoML с использованием whitebox
        logger.info("Обучение модели LightAutoML с использованием whitebox...")
        automl = TabularAutoML(task='reg',
                               timeout=600,
                               cpu_limit=-1,
                               model_names=['whitebox'])
        oof_preds = automl.fit_predict(train_data)
        
        logger.info("Выполнение предсказания на тестовой выборке LightAutoML с whitebox...")
        y_pred = automl.predict(test_data)
        metrics = self._calculate_metrics(test_data[self.target_column], y_pred.data.ravel())

        # Сохраняем модель и информацию о признаках
        logger.info("Сохранение модели LightAutoML с whitebox...")
        self.best_model = automl
        self.final_features = self.numeric_cols  # Используются все числовые признаки
        self.top_features = self.numeric_cols

        joblib.dump(self.best_model, 'model.pkl')
        with open('features.json', 'w') as f:
            json.dump(self.final_features, f)
        with open('top_features.json', 'w') as f:
            json.dump(self.top_features, f)

        # Сохраняем отчет по метрикам
        # Так как важность признаков не вычислялась, передаем пустой список
        save_metrics_report(metrics, [], "LightAutoML_Model")

        logger.info("Обучение модели завершено")
        return self.best_model, metrics

    def _load_model_state(self, model_path: str, features_path: str, top_features_path: str):
        """Загружает состояние модели (модель и признаки) из файлов."""
        logger.info("Загрузка состояния модели из файлов...")
        self.best_model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            self.final_features = json.load(f)
        with open(top_features_path, 'r') as f:
            self.top_features = json.load(f)
        logger.info("Модель LightAutoML с whitebox успешно загружена.")

    def predict(self, data: pd.DataFrame):
        """
        Делает предсказание на основе лучшей модели LightAutoML с whitebox.
        """
        logger.info("Выполнение предсказания LightAutoML с whitebox...")
        y_pred = self.best_model.predict(data)
        logger.info("Предсказание завершено.")
        
        return y_pred.data.ravel()


if __name__ == "__main__":
    data = pd.read_parquet("data_after_analys.parquet")
    
    # Определяем целевую переменную
    target_column_name = [col for col in data.columns if 'Гранулометрия' in col][0]
    
    # Разделение данных на обучающую и тестовую выборки
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
    logger.info(f"Размер обучающей выборки (DataFrame): {data_train.shape}")
    logger.info(f"Размер тестовой выборки (DataFrame): {data_test.shape}")

    model = Model(None, is_load_model=False)  # Инициализируем без данных, т.к. train сам подготовит
    model.train(data_train)  # Передаем обучающий DataFrame
    
    # Предсказание для тестовой выборки (без целевой колонки)
    start_time = time.time()
    y_pred = model.predict(data_test.drop(target_column_name, axis=1))
    end_time = time.time()
    logger.info(f"Время предсказания: {(end_time - start_time):.10f} секунд для {len(y_pred)} записей")

    # Сравнение первых 10 реальных и предсказанных значений
    y_test_values = data_test[target_column_name]
    logger.info("Сравнение первых 10 реальных и предсказанных значений:")
    for i in range(min(10, len(y_test_values))):
        print(f"Реальное: {y_test_values.iloc[i]:.4f}, Предсказанное: {y_pred[i]:.4f}")

    # Вычисляем RMSE
    rmse_value = np.sqrt(mean_squared_error(y_test_values, y_pred))
    print(f"\nИтоговый RMSE на тестовых данных: {rmse_value:.4f}")
