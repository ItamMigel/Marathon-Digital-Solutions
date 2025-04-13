import pandas as pd
import numpy as np
import time
import os
import random
import datetime
import logging
from typing import Dict, Any, List

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fake_data_generator")

def generate_random_value(base_value: float, variation_percent: float = 5.0) -> float:
    """
    Генерирует случайное значение в пределах указанного процента от базового значения.
    
    Args:
        base_value: Базовое значение
        variation_percent: Процент вариации (по умолчанию 5%)
        
    Returns:
        Случайное значение в пределах вариации
    """
    if base_value == 0:
        return random.uniform(0, 1)
    
    variation = base_value * (variation_percent / 100)
    return base_value + random.uniform(-variation, variation)

def generate_fake_data() -> Dict[str, Any]:
    """
    Генерирует фейковые данные для всех столбцов.
    
    Returns:
        Словарь с фейковыми данными
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Базовые значения для основных параметров
    base_values = {
        "Мощность МПСИ кВт": 2500,
        "Мощность МШЦ кВт": 3000,
        "Ток МПСИ А": 150,
        "Ток МШЦ А": 180,
        "Исходное питание МПСИ т/ч": 400,
        "Возврат руды МПСИ т/ч": 100,
        "Общее питание МПСИ т/ч": 500,
        "Расход воды МПСИ PV м3/ч": 250,
        "Расход воды МПСИ SP м3/ч": 260,
        "Расход воды МПСИ CV %": 95,
        "факт соотношение руда/вода МПСИ": 2.0,
        "Давление на подшипник МПСИ загрузка Бар": 12.5,
        "Давление на подшипник МПСИ разгрузка Бар": 11.8,
        "Обороты насоса %": 85,
        "Давление в ГЦ насоса Бар": 3.2,
        "Расход в ГЦ насоса м3/ч": 320,
        "Расход оборотной воды м3/ч": 450,
        "pH оборотной воды": 7.5,
        "t оборотной воды": 22.5,
        "Поток л/мин": 1200,
        "Температура масла основной маслостанции слив МПСИ": 45,
        "Температура масла маслостанции электродвигатель МШЦ": 55,
        "Температура масла маслостанции электродвигатель МПСИ": 52,
        "Уровень в зумпфе %": 65,
        "Температура масла основной маслостанции подача МШЦ": 42,
        "Температура масла редуктора МПСИ": 58,
        "Температура масла основной маслостанции слив МШЦ": 48,
        "Давление на подшипник МШЦ разгрузка Бар": 10.5,
        "Температура масла основной маслостанции подача МПСИ": 40,
        "Давление на подшипник МШЦ загрузка Бар": 11.2,
        "Плотность слива ГЦ кг/л": 1.25,
        "Расход извести МШЦ л/ч": 85,
        "Температура масла редуктора МШЦ": 60
    }
    
    # Генерация случайных значений для всех параметров
    data = {"Время": current_time}
    for column, base_value in base_values.items():
        data[column] = round(generate_random_value(base_value), 2)
    
    return data

def append_to_csv(file_path: str, data: Dict[str, Any]) -> None:
    """
    Добавляет данные в CSV файл. Если файл не существует, создает его с заголовками.
    
    Args:
        file_path: Путь к CSV файлу
        data: Словарь с данными для добавления
    """
    try:
        # Проверяем существование файла
        file_exists = os.path.isfile(file_path)
        
        # Создаем DataFrame из данных
        df = pd.DataFrame([data])
        
        # Если файл не существует, записываем с заголовками
        # Если существует, добавляем данные без заголовков
        df.to_csv(file_path, mode='a', header=not file_exists, index=False)
        
        logger.info(f"Данные успешно добавлены в {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при добавлении данных в файл {file_path}: {str(e)}")

def generate_data_periodically(file_path: str, interval_seconds: int = 60) -> None:
    """
    Периодически генерирует и добавляет фейковые данные в CSV файл.
    
    Args:
        file_path: Путь к CSV файлу
        interval_seconds: Интервал между генерациями данных в секундах
    """
    logger.info(f"Запуск генератора фейковых данных для файла {file_path} с интервалом {interval_seconds} секунд")
    
    try:
        while True:
            # Генерация и добавление данных
            data = generate_fake_data()
            append_to_csv(file_path, data)
            
            # Ожидание до следующей генерации
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Генератор фейковых данных остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка в работе генератора фейковых данных: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    file = 'data/df_area_1_bez_granulometrii.csv'
    interval = 60
    
    generate_data_periodically(file, interval)
