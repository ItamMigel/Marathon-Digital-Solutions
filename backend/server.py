import datetime
import logging
import os
import asyncio
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
import traceback # Для логгирования стектрейса

# Для работы с данными
import pandas as pd

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect, text, MetaData, Table
from sqlalchemy.ext.declarative import declared_attr, declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import func # для func.now() и func.count()
from sqlalchemy import select # Добавляем select
from sqlalchemy.exc import OperationalError # Для отлова ошибок блокировки

# FastAPI
from fastapi import FastAPI, Depends, HTTPException, status, Query # Добавляем Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import requests
import json

# --- Конфигурация и Логгирование ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='server.log', encoding='utf-8', filemode='a')
logger = logging.getLogger("server")

class Config:
    DATABASE_URL = "sqlite:///data/value.db"
    DEBUG = True
    API_VERSION = "1.0.0"
    API_TITLE = "Система Мониторинга Линий"
    API_DESCRIPTION = "API для мониторинга данных из CSV файлов"

# --- Инициализация FastAPI --- 
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Настройка Базы Данных --- 
Base = declarative_base()
# Добавляем connect_args для увеличения таймаута
engine = create_engine(
    Config.DATABASE_URL, 
    echo=Config.DEBUG,
    connect_args={"timeout": 15} # 15 секунд ожидания блокировки
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ... (Проверка пути к БД остается как была) ...

# Менеджер контекста для сессий БД
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка базы данных: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        # Коммит не нужен здесь, он должен быть в логике эндпоинта или сервиса
        db.close()

# Зависимость для получения сессии БД
def get_db_session():
    with get_db() as session:
        yield session

# --- Модели SQLAlchemy --- 

# Модель для хранения метаданных отслеживаемых линий
class MonitoredLine(Base):
    __tablename__ = 'monitored_lines'
    id = Column(Integer, primary_key=True)
    area_name = Column(String, unique=True, index=True, nullable=False)
    file_path = Column(String, nullable=False)
    last_update = Column(DateTime, server_default=func.now())
    # Добавляем новые поля
    status = Column(String, default="Неизвестно", nullable=False) # Статус линии
    comment = Column(String, nullable=True) # Комментарий к статусу


class LLMRequest(BaseModel):
    prompt: str
    model: str = "deepseek-ai/DeepSeek-V3-0324"
    max_tokens: int = 2024
    temperature: float = 0.7
    role: str = "Ты не пишешь лишние комментарии, только ответ. Тебе будет передавать запрос, из которого нужно будет сделать json схему для построения графика на React."

class LLMResponse(BaseModel):
    response: str
    execution_time: float

# Кеш для динамических моделей таблиц данных
model_cache = {}

# Фабрика для создания моделей таблиц данных (как раньше, с extend_existing)
def create_data_model(area_name):
    class Data(Base):
        __tablename__ = area_name
        __table_args__ = {"extend_existing": True}
        
        id = Column(Integer, primary_key=True)
        Время = Column(DateTime, index=True)
        Мощность_МПСИ_кВт = Column(Float) 
        Мощность_МШЦ_кВт = Column(Float)
        Ток_МПСИ_А = Column(Float) 
        Ток_МШЦ_А = Column(Float)
        Исходное_питание_МПСИ_т_ч = Column(Float)
        Возврат_руды_МПСИ_т_ч = Column(Float)
        Общее_питание_МПСИ_т_ч = Column(Float)
        Расход_воды_МПСИ_PV_м3_ч = Column(Float)
        Расход_воды_МПСИ_SP_м3_ч = Column(Float)
        Расход_воды_МПСИ_CV_процент = Column(Float)
        факт_соотношение_руда_вода_МПСИ = Column(Float)
        Давление_на_подшипник_МПСИ_загрузка_Бар = Column(Float)
        Давление_на_подшипник_МПСИ_разгрузка_Бар = Column(Float)
        Температура_масла_основной_маслостанции_подача_МПСИ = Column(Float)
        Температура_масла_основной_маслостанции_слив_МПСИ = Column(Float)
        Температура_масла_маслостанции_электродвигатель_МПСИ = Column(Float)
        Температура_масла_редуктора_МПСИ = Column(Float)
        Давление_на_подшипник_МШЦ_загрузка_Бар = Column(Float)
        Давление_на_подшипник_МШЦ_разгрузка_Бар = Column(Float)
        Температура_масла_основной_маслостанции_подача_МШЦ = Column(Float)
        Температура_масла_основной_маслостанции_слив_МШЦ = Column(Float)
        Температура_масла_маслостанции_электродвигатель_МШЦ = Column(Float)
        Температура_масла_редуктора_МШЦ = Column(Float)
        Расход_извести_МШЦ_л_ч = Column(Float)
        Уровень_в_зумпфе_процент = Column(Float)
        Обороты_насоса_процент = Column(Float)
        Давление_в_ГЦ_насоса_Бар = Column(Float)
        Плотность_слива_ГЦ_кг_л = Column(Float)
        pH_оборотной_воды = Column(Float)
        t_оборотной_воды = Column(Float)
        Гранулометрия_процент = Column(Float)
        Поток_л_мин = Column(Float)
        Расход_оборотной_воды_м3_ч = Column(Float)
        Расход_в_ГЦ_насоса_м3_ч = Column(Float)

    if area_name not in model_cache:
        model_cache[area_name] = Data
    return Data

# Получение модели для существующей таблицы (как раньше)
def get_table_model(table_name: str):
    if table_name in model_cache:
        return model_cache[table_name]

    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        logger.warning(f"Попытка получить модель для несуществующей таблицы: {table_name}")
        return None # Возвращаем None если таблицы нет

    columns = {column["name"]: column for column in inspector.get_columns(table_name)}

    if not columns:
        logger.warning(f"Не удалось получить колонки для существующей таблицы: {table_name}")
        return None

    # --- ИСПРАВЛЕНИЕ: Используем type() для динамического создания класса ---
    attrs_dict = {
        '__tablename__': table_name,
        '__table_args__': {"extend_existing": True},
        # Добавляем 'id' колонку явно, если она есть и является primary key
        # SQLAlchemy может потребовать явное определение PK
        **{
            col_name: Column(
                col_info["type"],
                primary_key=col_info.get("primary_key", False),
                # Добавляем index=True для колонки 'Время', если она существует
                index=(col_name == 'Время')
            )
            for col_name, col_info in columns.items()
        }
    }

    # Создаем класс динамически
    # Имя класса делаем уникальным, добавляя имя таблицы, чтобы избежать конфликтов
    DynamicTable = type(f"DynamicTableModel_{table_name}", (Base,), attrs_dict)
    # ---------------------------------------------------------------------

    model_cache[table_name] = DynamicTable
    return DynamicTable

# Проверка и создание таблицы данных по шаблону
# Теперь эта функция МОЖЕТ вызвать исключение, если таблица заблокирована
def ensure_data_table_exists(area_name):
    inspector = inspect(engine)
    if not inspector.has_table(area_name):
        logger.info(f"Таблица данных {area_name} не найдена, создаем по шаблону...")
        # Убрали try...except, ошибка будет перехвачена в вызывающей функции (/api/add)
        DataModelTemplate = create_data_model(area_name)
        DataModelTemplate.__table__.create(bind=engine) # Может вызвать OperationalError: database is locked
        logger.info(f"Таблица данных {area_name} успешно создана.")
        # except Exception as e:
        #     logger.error(f"Не удалось создать таблицу данных {area_name}: {e}")
        #     raise # Перебрасываем ошибку

# --- Сервисные функции --- 

# Переименовываем и меняем логику
def read_initial_csv_data(file_path: str, max_lines: int = 1440) -> Optional[pd.DataFrame]:
    """Читает последние N строк из CSV файла (но не более max_lines), обрабатывая разные кодировки."""
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден для чтения: {file_path}")
        return None
    try:
        # Читаем весь файл
        try:
            df = pd.read_csv(file_path, sep=',', encoding='utf-8-sig')
        except UnicodeDecodeError:
            logger.warning(f"Не удалось прочитать CSV {file_path} с UTF-8, пробуем cp1251...")
            df = pd.read_csv(file_path, sep=',', encoding='cp1251')
        
        if df.empty:
             logger.warning(f"CSV файл {file_path} пуст.")
             return None
             
        # Определяем, сколько строк брать (не больше max_lines и не больше, чем есть в файле)
        num_rows_to_take = min(max_lines, len(df))
        logger.info(f"Читаем последние {num_rows_to_take} строк из {len(df)} в файле {file_path}")
        return df.tail(num_rows_to_take)

    except pd.errors.EmptyDataError:
        logger.warning(f"CSV файл {file_path} пуст или содержит только заголовки.")
        return None
    except Exception as e:
        logger.error(f"Ошибка чтения CSV файла {file_path}: {e}")
        return None

def add_dataframe_to_db(db: Session, df: pd.DataFrame, area_name: str):
    """Добавляет данные из DataFrame в указанную таблицу."""
    
    DataModel = get_table_model(area_name)
    if not DataModel:
        logger.error(f"Не найдена модель для таблицы {area_name} при добавлении данных.")
        return 0
        
    valid_columns = [c.key for c in DataModel.__table__.columns]
    added_count = 0
    skipped_count = 0
    renamed_count = 0
    total_rows = len(df)

    logger.info(f"Начинаем обработку {total_rows} строк для добавления в {area_name}.")

    # Внешний try для общих ошибок обработки DataFrame
    try:
        # Переименование колонок
        field_mapping = {
            'Время': 'Время',
            'Мощность МПСИ кВт': 'Мощность_МПСИ_кВт',
            'Мощность МШЦ кВт': 'Мощность_МШЦ_кВт',
            'Ток МПСИ А': 'Ток_МПСИ_А',
            'Ток МШЦ А': 'Ток_МШЦ_А',
            'Исходное питание МПСИ т/ч': 'Исходное_питание_МПСИ_т_ч',
            'Возврат руды МПСИ т/ч': 'Возврат_руды_МПСИ_т_ч',
            'Общее питание МПСИ т/ч': 'Общее_питание_МПСИ_т_ч',
            'Расход воды МПСИ PV м3/ч': 'Расход_воды_МПСИ_PV_м3_ч',
            'Расход воды МПСИ SP м3/ч': 'Расход_воды_МПСИ_SP_м3_ч',
            'Расход воды МПСИ CV %': 'Расход_воды_МПСИ_CV_процент',
            'факт соотношение руда/вода МПСИ': 'факт_соотношение_руда_вода_МПСИ',
            'Давление на подшипник МПСИ загрузка Бар': 'Давление_на_подшипник_МПСИ_загрузка_Бар',
            'Давление на подшипник МПСИ разгрузка Бар': 'Давление_на_подшипник_МПСИ_разгрузка_Бар',
            'Температура масла основной маслостанции подача МПСИ': 'Температура_масла_основной_маслостанции_подача_МПСИ',
            'Температура масла основной маслостанции слив МПСИ': 'Температура_масла_основной_маслостанции_слив_МПСИ',
            'Температура масла маслостанции электродвигатель МПСИ': 'Температура_масла_маслостанции_электродвигатель_МПСИ',
            'Температура масла редуктора МПСИ': 'Температура_масла_редуктора_МПСИ',
            'Давление на подшипник МШЦ загрузка Бар': 'Давление_на_подшипник_МШЦ_загрузка_Бар',
            'Давление на подшипник МШЦ разгрузка Бар': 'Давление_на_подшипник_МШЦ_разгрузка_Бар',
            'Температура масла основной маслостанции подача МШЦ': 'Температура_масла_основной_маслостанции_подача_МШЦ',
            'Температура масла основной маслостанции слив МШЦ': 'Температура_масла_основной_маслостанции_слив_МШЦ',
            'Температура масла маслостанции электродвигатель МШЦ': 'Температура_масла_маслостанции_электродвигатель_МШЦ',
            'Температура масла редуктора МШЦ': 'Температура_масла_редуктора_МШЦ',
            'Расход извести МШЦ л/ч': 'Расход_извести_МШЦ_л_ч',
            'Уровень в зумпфе %': 'Уровень_в_зумпфе_процент',
            'Обороты насоса %': 'Обороты_насоса_процент',
            'Давление в ГЦ насоса Бар': 'Давление_в_ГЦ_насоса_Бар',
            'Плотность слива ГЦ кг/л': 'Плотность_слива_ГЦ_кг_л',
            'pH оборотной воды': 'pH_оборотной_воды',
            't оборотной воды': 't_оборотной_воды',
            'Гранулометрия %': 'Гранулометрия_процент',
            'Поток л/мин': 'Поток_л_мин',
            'Расход оборотной воды м3/ч': 'Расход_оборотной_воды_м3_ч',
            'Расход в ГЦ насоса м3/ч': 'Расход_в_ГЦ_насоса_м3_ч'
        }

        renamed_columns_mapping = {old: new for old, new in field_mapping.items() if old in df.columns and old != new}
        if renamed_columns_mapping:
            df = df.rename(columns=renamed_columns_mapping)
            renamed_count = len(renamed_columns_mapping)
            logger.info(f"Переименовано {renamed_count} колонок для {area_name}.")

        # Преобразование времени
        if 'Время' in df.columns:
            original_time_col = df['Время'].copy()
            df['Время'] = pd.to_datetime(df['Время'], errors='coerce')
            failed_time_parse_indices = df[df['Время'].isna()].index
            if not failed_time_parse_indices.empty:
                logger.warning(f"Не удалось распознать 'Время' в {len(failed_time_parse_indices)} строках для {area_name}. Примеры некорректных значений:")
                for index in failed_time_parse_indices[:5]:
                    logger.warning(f"  Строка {index}: '{original_time_col.loc[index]}'")
                df = df.dropna(subset=['Время']) # Удаляем строки с нераспознанным временем ЗДЕСЬ
                if df.empty: # Проверяем, остались ли строки после удаления
                    logger.error(f"Критическая ошибка: После удаления строк с нераспознанным временем не осталось данных для {area_name}.")
                    return 0
        else: # Добавлен правильный отступ
            logger.error(f"Критическая ошибка: Колонка 'Время' не найдена в данных для {area_name}. Невозможно добавить данные.")
            return 0

        # Итерация по строкам
        for index, row in df.iterrows():
            if pd.isna(row.get('Время')):
                skipped_count += 1
                continue

            data_dict = {k: v for k, v in row.to_dict().items()
                         if k in valid_columns and pd.notna(v)}

            if not data_dict or (len(data_dict) == 1 and 'Время' in data_dict):
                logger.warning(f"Строка {index} для {area_name} не содержит валидных данных (кроме, возможно, времени), пропуск.")
                skipped_count += 1
                continue

            # Внутренний try для ошибок слияния отдельных строк
            try:
                data_instance = DataModel(**data_dict)
                db.merge(data_instance)
                added_count += 1
            except Exception as merge_err:
                logger.error(f"Ошибка при подготовке/слиянии записи (строка DataFrame {index}) в {area_name}: {merge_err}")
                logger.debug(f"Проблемные данные строки {index}: {data_dict}")
                skipped_count += 1

        logger.info(f"Обработка для {area_name} завершена. Подготовлено: {added_count}, Пропущено: {skipped_count} из {total_rows} строк.")
        return added_count

    except Exception as e: # Добавлен except к внешнему try
        logger.error(f"Критическая ошибка при обработке DataFrame для {area_name}: {e}\n{traceback.format_exc()}")
        return 0

# Новая функция для чтения только новых записей
def read_new_csv_records(file_path: str, last_known_timestamp: Optional[datetime.datetime]) -> Optional[pd.DataFrame]:
    """Читает CSV и возвращает DataFrame только с записями новее last_known_timestamp."""
    logger.debug(f"Функция read_new_csv_records вызвана для {file_path}. last_known_timestamp: {last_known_timestamp} (Тип: {type(last_known_timestamp)})")
    
    if not os.path.exists(file_path):
        logger.debug(f"Файл {file_path} не найден при проверке новых записей.")
        return None
    if last_known_timestamp is None:
        logger.debug(f"last_known_timestamp is None для {file_path}, пропускаем проверку новых записей.")
        return None
        
    # Отступ 4 пробела
    try:
        # Отступ 8 пробелов
        try:
            df = pd.read_csv(file_path, sep=',', encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, sep=',', encoding='cp1251')
        
        if df.empty:
            return None

        # Преобразуем время
        if 'Время' not in df.columns:
            logger.warning(f"Нет колонки 'Время' в {file_path} при проверке новых записей.")
            return None
        df['Время'] = pd.to_datetime(df['Время'], errors='coerce')
        df = df.dropna(subset=['Время']) # Удаляем строки с невалидным временем
        
        if df.empty: # Проверяем после удаления строк с некорректным временем
             logger.debug(f"Нет валидных записей со временем в {file_path}.")
             return None

        # ----- ДОБАВЛЯЕМ ДЕТАЛЬНОЕ ЛОГГИРОВАНИЕ ПЕРЕД СРАВНЕНИЕМ -----
        logger.debug(f"[{file_path}] Перед сравнением: last_known_timestamp = {last_known_timestamp} (type: {type(last_known_timestamp)})")
        logger.debug(f"[{file_path}] Перед сравнением: df['Время'].dtype = {df['Время'].dtype}")
        logger.debug(f"[{file_path}] Перед сравнением: Первые 5 значений df['Время']: \n{df['Время'].head().to_string()}")
        logger.debug(f"[{file_path}] Перед сравнением: Последние 5 значений df['Время']: \n{df['Время'].tail().to_string()}")
        # --------------------------------------------------------------
        
        # Фильтруем по времени
        df_new = None # Инициализируем
        try:
            # Пробуем прямое сравнение
            df_new = df[df['Время'] > last_known_timestamp]
            logger.debug(f"[{file_path}] После прямого сравнения найдено {len(df_new) if df_new is not None else 'None'} новых строк.")
        except TypeError as te:
            # ... (обработка TypeError и попытка конвертации - оставляем как есть) ...
            logger.warning(f"Ошибка сравнения времени для {file_path} (возможно, aware vs naive): {te}. Попытка приведения к naive.")
            try:
                # ... (код конвертации) ...
                df_new = df[df_time_naive > last_known_naive]
                logger.debug(f"[{file_path}] После сравнения с конвертацией найдено {len(df_new) if df_new is not None else 'None'} новых строк.")
            except Exception as convert_err:
                 logger.error(f"Не удалось исправить ошибку сравнения времени для {file_path}: {convert_err}")
                 return None
        
        # Проверяем результат фильтрации
        if df_new is None or df_new.empty:
             logger.debug(f"Новых записей в {file_path} не найдено после {last_known_timestamp}.")
             return None
        else: # Исправлен отступ
            max_new_time = df_new['Время'].max()
            logger.info(f"Найдено {len(df_new)} новых записей в {file_path}. Максимальное новое время: {max_new_time}")
            return df_new

    # Отступ 4 пробела (соответствует внешнему try)
    except pd.errors.EmptyDataError:
        # Отступ 8 пробелов
        logger.debug(f"CSV файл {file_path} пуст при проверке новых записей.")
        return None
    except Exception as e:
        # Отступ 8 пробелов
        logger.error(f"Ошибка чтения/фильтрации CSV файла {file_path} для новых записей: {e}")
        return None

# --- Pydantic схемы для API --- 
class AddLineRequest(BaseModel):
    file_path: str
    area_name: str

# Схема для информации о пагинации
class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int

# Модель для данных тренда (Время и Значение)
class TrendDataPoint(BaseModel):
    Время: datetime.datetime
    value: Optional[float] = None

# Обновляем LineDetailResponse
class LineDetailResponse(BaseModel):
    area_name: str
    last_update: Optional[datetime.datetime] = None
    status: str # Теперь будет браться из БД
    comment: Optional[str] = None # Добавляем комментарий
    data: List[Dict[str, Any]]
    pagination: PaginationInfo

# Обновляем LineSummaryResponse
class LineSummaryResponse(BaseModel):
    area_name: str
    last_update: Optional[datetime.datetime] = None
    status: str # Поле уже есть, будет браться из БД
    # Добавляем поле для данных тренда гранулометрии
    granulometry_trend: List[TrendDataPoint] = [] 

# --- Модели для ответа эндпоинта аналитики ---
class ColumnAnalytics(BaseModel):
    count: int
    min: Optional[float] = None
    max: Optional[float] = None
    average: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None

class AnalyticsResponse(BaseModel):
    area_name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    requested_minutes: int
    statistics: Dict[str, ColumnAnalytics] = {}
    # Добавляем данные для графиков по ключевым параметрам
    trends: Dict[str, List[TrendDataPoint]] = {}

# --- API Эндпоинты --- 

# Изменяем /api/get для автоматического обновления данных
@app.get('/api/get', response_model=List[LineSummaryResponse], tags=["Мониторинг"])
def get_lines_summary(db: Session = Depends(get_db_session)):
    """Получить сводный статус для всех линий, ПРЕДВАРИТЕЛЬНО ОБНОВИВ ДАННЫЕ из CSV."""
    logger.info("Запрос /api/get: Начинаем проверку и обновление данных линий...")
    lines_info = db.query(MonitoredLine).all()
    response = []

    for line in lines_info:
        logger.debug(f"Обработка линии: {line.area_name}")
        current_status = line.status
        last_update_time = line.last_update

        # 1. Попытка прочитать и добавить новые данные
        try:
            df_new = read_new_csv_records(line.file_path, line.last_update)
            if df_new is not None:
                added_count = add_dataframe_to_db(db, df_new, line.area_name)
                if added_count > 0:
                    # Находим максимальное время среди добавленных записей
                    new_last_update = df_new['Время'].max()
                    # Обновляем запись в БД
                    line.last_update = new_last_update
                    line.status = "OK" # Обновляем статус на OK
                    db.add(line) # Добавляем измененный объект line в сессию
                    try:
                        db.commit() # Коммитим изменения ДЛЯ ЭТОЙ ЛИНИИ
                        db.refresh(line) # Обновляем объект line из БД
                        last_update_time = line.last_update # Обновляем для ответа
                        current_status = line.status
                        logger.info(f"Линия {line.area_name}: успешно добавлено {added_count} новых записей. Last_update: {new_last_update}")
                    except Exception as commit_err:
                         logger.error(f"Ошибка коммита при обновлении линии {line.area_name}: {commit_err}")
                         db.rollback() # Откатываем изменения для этой линии
                         # Статус оставляем как был до попытки обновления
                         current_status = db.query(MonitoredLine.status).filter(MonitoredLine.id == line.id).scalar() or line.status
            else:
                     logger.info(f"Линия {line.area_name}: новые записи были найдены, но не добавлены (возможно, пропущены при обработке). Обновление не требуется.")
        except Exception as update_err:
            logger.error(f"Ошибка при попытке обновления данных для линии {line.area_name}: {update_err}")
            db.rollback() # Откатываем, если ошибка произошла во время add_dataframe_to_db
            # Статус не меняем, используем тот, что был в начале итерации

        # 2. Финальная проверка статуса (файл, таблица)
        if current_status in ["OK", "Неизвестно"] and not os.path.exists(line.file_path):
            current_status = "Ошибка: Файл не найден"
            logger.warning(f"Файл для линии {line.area_name} не найден, статус для ответа: '{current_status}'.")
        
        # (Опционально) Проверка существования таблицы данных, если статус ОК/Неизвестно
        # inspector = inspect(engine)
        # if current_status in ["OK", "Неизвестно"] and not inspector.has_table(line.area_name):
        #    current_status = "Ошибка: Таблица данных не найдена"
        #    logger.warning(f"Таблица данных {line.area_name} не найдена, статус для ответа: '{current_status}'.")

        # 3. Получаем данные для тренда гранулометрии
        granulometry_data = []
        DataModel = get_table_model(line.area_name) 
        if DataModel and hasattr(DataModel, 'Время') and hasattr(DataModel, 'Гранулометрия_процент'):
            try:
                trend_query = db.query(DataModel.Время, DataModel.Гранулометрия_процент)\
                                .order_by(DataModel.Время.desc())\
                                .limit(20) # Берем последние 20 записей для мини-графика
                trend_results = trend_query.all()
                # Преобразуем в формат TrendDataPoint и разворачиваем порядок, чтобы время шло по возрастанию
                granulometry_data = [
                    TrendDataPoint(Время=row.Время, value=row.Гранулометрия_процент) 
                    for row in reversed(trend_results) # Разворачиваем, чтобы было от старых к новым
                ]
                logger.debug(f"Линия {line.area_name}: Загружено {len(granulometry_data)} точек для тренда гранулометрии.")
            except Exception as trend_err:
                logger.warning(f"Линия {line.area_name}: Ошибка при получении данных тренда гранулометрии: {trend_err}")
        elif not DataModel:
             logger.warning(f"Линия {line.area_name}: Модель данных не найдена, тренд гранулометрии не будет загружен.")
        elif not hasattr(DataModel, 'Время') or not hasattr(DataModel, 'Гранулометрия_процент'):
            logger.warning(f"Линия {line.area_name}: Отсутствует столбец 'Время' или 'Гранулометрия_процент', тренд не будет загружен.")

        # 4. Добавляем результат в ответ (было 3)
        response.append(
            LineSummaryResponse(
                area_name=line.area_name,
                last_update=last_update_time, # Используем обновленное время, если было
                status=current_status,
                granulometry_trend=granulometry_data # Добавляем данные тренда
            )
        )
    
    logger.info(f"Запрос /api/get: Проверка и обновление завершены. Возвращено {len(response)} линий.")
    return response

@app.post('/api/add', status_code=status.HTTP_201_CREATED, tags=["Мониторинг"])
def add_monitored_line(request: AddLineRequest, db: Session = Depends(get_db_session)):
    """Добавить новую производственную линию для мониторинга."""
    logger.info(f"Запрос на добавление линии: {request.area_name}, путь: {request.file_path}")

    # --- Предварительные проверки --- 
    # Проверка 1: Существует ли файл?
    if not os.path.exists(request.file_path):
        logger.error(f"Файл не найден: {request.file_path}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Файл не найден по указанному пути: {request.file_path}")

    # Проверка 2: Уникально ли имя линии (area_name) в monitored_lines?
    existing_line = db.query(MonitoredLine).filter(MonitoredLine.area_name == request.area_name).first()
    if existing_line:
        logger.warning(f"Попытка добавить линию с существующим именем: {request.area_name}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Линия с именем '{request.area_name}' уже существует.")

    # --- Попытка создать таблицу данных (если ее нет) --- 
    try:
        ensure_data_table_exists(request.area_name)
    except Exception as e:
        logger.error(f"Не удалось создать таблицу данных {request.area_name} при добавлении линии: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка создания таблицы данных: {e}")

    # --- Основная транзакция: Добавление записи и начальных данных --- 
    added_count = 0
    new_line = None
    try:
        # Создаем запись в таблице метаданных
        new_line = MonitoredLine(
            area_name=request.area_name,
            file_path=request.file_path
        )
        db.add(new_line)
        db.flush() 
        logger.info(f"Запись для линии {new_line.area_name} подготовлена к добавлению в monitored_lines со статусом '{new_line.status}'.")

        # Читаем начальные данные CSV 
        df_initial = read_initial_csv_data(request.file_path, max_lines=1440) 

        initial_data_added = False
        if df_initial is not None and not df_initial.empty:
            logger.info(f"Прочитано {len(df_initial)} строк из CSV для начальной загрузки в {request.area_name}.")
            
            # Важно: преобразуем 'Время' ПЕРЕД добавлением в БД и взятием max()
            if 'Время' in df_initial.columns:
                df_initial['Время'] = pd.to_datetime(df_initial['Время'], errors='coerce')
                df_initial = df_initial.dropna(subset=['Время']) # Убираем строки с невалидным временем
            else:
                logger.error(f"Нет колонки 'Время' в начальных данных для {request.area_name}. Невозможно добавить.")
                df_initial = None # Сбрасываем df, чтобы не пытаться добавить
                
            if df_initial is not None and not df_initial.empty: # Проверяем еще раз после обработки времени
                added_count = add_dataframe_to_db(db, df_initial, request.area_name)
                if added_count > 0:
                    # Находим максимальное время СРЕДИ ДОБАВЛЕННЫХ начальных записей
                    max_initial_timestamp = df_initial['Время'].max()
                    # Исправляем установку last_update!
                    new_line.last_update = max_initial_timestamp 
                    new_line.status = "OK" 
                    db.add(new_line) 
                    initial_data_added = True
                    logger.info(f"Начальные данные ({added_count} шт.) для {request.area_name} подготовлены. last_update установлен на {max_initial_timestamp}, статус изменен на 'OK'.")
                else: # Исправлен отступ
                    logger.warning(f"Не удалось добавить начальные данные для {request.area_name} (возможно, все пропущены). Статус останется '{new_line.status}'.")
            else: # Исправлен отступ и текст лога
                 logger.warning(f"В начальных данных для {request.area_name} не найдено валидных записей со временем после обработки или сам DataFrame пуст. Статус останется '{new_line.status}'.")
                 
        db.commit() 
        db.refresh(new_line) 
        logger.info(f"Линия {new_line.area_name} успешно сохранена со статусом '{new_line.status}' и last_update '{new_line.last_update}'. Начальных данных добавлено: {added_count} шт.")

        return {"message": "Производственная линия успешно добавлена", "area_name": new_line.area_name, "added_initial_records": added_count, "initial_status": new_line.status}

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при добавлении записи линии/данных {request.area_name}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Внутренняя ошибка сервера при сохранении данных линии: {str(e)}")


@app.delete('/api/lines/{area_name}', status_code=status.HTTP_204_NO_CONTENT, tags=["Мониторинг"])
def delete_monitored_line(area_name: str, db: Session = Depends(get_db_session)):
    """Удалить отслеживаемую линию и связанную с ней таблицу данных."""
    logger.info(f"Запрос на удаление линии: {area_name}")
    
    line_to_delete = db.query(MonitoredLine).filter(MonitoredLine.area_name == area_name).first()
    if not line_to_delete:
        logger.warning(f"Попытка удаления несуществующей линии: {area_name}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Линия '{area_name}' не найдена.")
    
    try:
        # 1. Удаляем запись из monitored_lines
        db.delete(line_to_delete)
        logger.info(f"Запись для линии {area_name} удалена из monitored_lines.")
        
        # 2. Удаляем таблицу данных
        inspector = inspect(engine)
        if inspector.has_table(area_name):
            try:
                metadata = MetaData()
                table_to_drop = Table(area_name, metadata, autoload_with=engine)
                table_to_drop.drop(engine)
                logger.info(f"Таблица данных {area_name} успешно удалена.")
                 # Удаляем из кеша моделей
                if area_name in model_cache:
                    del model_cache[area_name]
                    logger.info(f"Модель для таблицы {area_name} удалена из кеша.")
            except Exception as drop_err:
                # Если не удалось удалить таблицу - это проблема, откатываем удаление записи
                logger.error(f"Не удалось удалить таблицу данных {area_name}: {drop_err}. Откат операции.")
                db.rollback() # Откатываем удаление записи из monitored_lines
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка при удалении таблицы данных: {drop_err}")
            else:
             logger.warning(f"Таблица данных {area_name} для удаляемой линии не найдена в БД.")

        db.commit() # Коммитим удаление записи и таблицы (если она была)
        return # Статус 204

    except HTTPException: # От переброшенных исключений
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при удалении линии {area_name}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Внутренняя ошибка сервера при удалении линии: {str(e)}")


# Изменяем эндпоинт для получения данных ОДНОЙ линии с ПАГИНАЦИЕЙ
@app.get('/api/lines/{area_name}', response_model=LineDetailResponse, tags=["Мониторинг"])
def get_single_line_status(
    area_name: str,
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"), 
    db: Session = Depends(get_db_session)
):
    """Получить статус и ПАГИНИРОВАННЫЕ данные для ОДНОЙ отслеживаемой линии."""
    logger.info(f"Запрос данных для линии: {area_name}, страница: {page}, размер: {page_size}")
    line = db.query(MonitoredLine).filter(MonitoredLine.area_name == area_name).first()
    if not line:
        logger.warning(f"Линия {area_name} не найдена при запросе GET /api/lines/{area_name}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Линия '{area_name}' не найдена.")

    # Инициализация (отступ 4 пробела)
    current_status = line.status
    current_comment = line.comment
    data_records = []
    total_items = 0
    total_pages = 0
    # `page` уже определен в параметрах функции, инициализировать не нужно
    
    # Проверка файла (отступ 4 пробела)
    if current_status in ["OK", "Неизвестно"] and not os.path.exists(line.file_path):
        # Отступ 8 пробелов
        current_status = "Ошибка: Файл не найден"
        logger.warning(f"Файл для линии {line.area_name} не найден, статус временно изменен на '{current_status}' для ответа.")
    else:
        # Отступ 8 пробелов - Начинаем блок try для чтения данных
        try:
            # Отступ 12 пробелов
            DataModel = get_table_model(line.area_name)
            if DataModel:
                # Отступ 16 пробелов
                if hasattr(DataModel, 'id') and hasattr(DataModel, 'Время'): 
                    # Отступ 20 пробелов
                    count_query = db.query(func.count(DataModel.id))
                    total_items = count_query.scalar()
                    logger.info(f"Найдено {total_items} записей в таблице {area_name}.")

                    if total_items > 0:
                        # Отступ 24 пробела
                        total_pages = (total_items + page_size - 1) // page_size
                        # Корректируем page ДО вычисления offset
                        page = max(1, min(page, total_pages))
                        offset = (page - 1) * page_size
                        paginated_query = db.query(DataModel).order_by(DataModel.Время.desc()).offset(offset).limit(page_size)
                        latest_data = paginated_query.all()
                        logger.info(f"Запрошено {len(latest_data)} записей для страницы {page}.")
                        
                        inspector = inspect(DataModel)
                        columns = [c.key for c in inspector.columns]
                        data_records = []
                        for record in latest_data:
                            # Отступ 28 пробелов
                            data_dict = {col: getattr(record, col, None) for col in columns}
                            if isinstance(data_dict.get('Время'), datetime.datetime):
                                # Отступ 32 пробела
                                data_dict['Время'] = data_dict['Время'].isoformat()
                            data_records.append(data_dict)
                    else:
                        # Отступ 24 пробела
                        total_pages = 0
                        page = 1 
                        logger.info(f"Таблица {line.area_name} пуста, данные не возвращаются.")
                else:
                    # Отступ 16 пробелов
                    current_status = "Ошибка: В таблице нет колонки 'Время' или 'id'"
                    logger.error(f"Таблица {line.area_name} не имеет колонки 'Время' или 'id' для пагинации.")
                    total_items, total_pages, page, data_records = 0, 0, 1, []
            else:
                # Отступ 12 пробелов
                current_status = "Ошибка: Таблица данных не найдена"
                logger.warning(f"Не найдена модель/таблица для {line.area_name} при запросе GET /api/lines/...")
                total_items, total_pages, page, data_records = 0, 0, 1, []
        except Exception as e: # Добавлен except
            # Отступ 12 пробелов
            logger.error(f"Ошибка чтения/пагинации данных из таблицы {line.area_name}: {e}\n{traceback.format_exc()}")
            current_status = "Ошибка: Чтения/обработки данных из БД"
            total_items, total_pages, page, data_records = 0, 0, 1, []

    # Возврат ответа (отступ 4 пробела)
    return LineDetailResponse(
        area_name=line.area_name,
        last_update=line.last_update,
        status=current_status,
        comment=current_comment,
        data=data_records, 
        pagination=PaginationInfo(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages
        )
    )

def generate_response_sync(prompt, model="deepseek-ai/DeepSeek-V3-0324", max_tokens=2024, temperature=0.7, role="user"):
    """
    Synchronously generate a response from the LLM API.
    
    Args:
        prompt: The user prompt to send to the model
        model: Model name to use
        max_tokens: Maximum tokens in the response
        temperature: Temperature setting for response randomness
        role: The role of the message sender (e.g., "user", "assistant", "system")
        
    Returns:
        The generated text response as a string
    """
    api_key = 'cpk_b9f646794b554414935934ec5a3513de.f78245306f06593ea49ef7bce2228c8e.kHJVJjyK8dtqB0oD2Ofv4AaME6MSnKDy'
    url = 'https://llm.chutes.ai/v1/chat/completions'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': model,
        'messages': [
            {
                'role': role,
                'content': prompt
            }
        ],
        'stream': True,  # Используем потоковую передачу как в примере
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    
    full_response = ""
    
    # Используем requests вместо aiohttp
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    # Обработка потокового ответа
    try: # Добавлен try для обработки ошибок requests
        for line in response.iter_lines():
            if line:
                try:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        line_text = line_text[6:]
                
                    if line_text.strip() and line_text != '[DONE]':
                        parsed = json.loads(line_text)
                        content = parsed.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if content:
                            full_response += content
                except json.JSONDecodeError:
                    if line_text.strip() == '[DONE]':
                        break
                    logger.error(f"Error parsing JSON line from LLM response: {line_text}") # Логгируем ошибку парсинга JSON
                    continue # Пропускаем эту строку, но продолжаем цикл
                except Exception as e: # Ловим другие ошибки декодирования/обработки строки
                    logger.error(f"Error processing line from LLM response: {str(e)}")
                    continue
    except Exception as e: # Ловим ошибки на уровне requests (например, сетевые)
            logger.error(f"Error during LLM API request or stream processing: {str(e)}")
            # В случае ошибки запроса, возможно, стоит вернуть пустой ответ или поднять исключение
            # Здесь просто вернем то, что успели собрать
            pass # Не используем continue здесь, т.к. мы вне цикла

    return full_response

@app.post('/api/llm', response_model=LLMResponse, tags=["Нейросеть"])
def llm_endpoint(request: LLMRequest):
    """
    Отправляет запрос к LLM API и возвращает ответ.
    """
    try:
        import time
        
        start_time = time.time()
        
        # Получаем ответ синхронно
        full_response = generate_response_sync(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            role=request.role
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"LLM request completed in {execution_time:.2f} seconds")
        
        return LLMResponse(
            response=full_response,
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error in LLM endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing LLM request: {str(e)}")


# --- Инициализация Приложения --- 
def initialize_database():
    """Создает все таблицы, определенные через Base.metadata."""
    # Отступ 4 пробела
    try:
        # Отступ 8 пробелов
        logger.info("Инициализация базы данных... Проверка таблиц.")
        MonitoredLine.__table__.create(bind=engine, checkfirst=True)
        logger.info("Таблица 'monitored_lines' успешно проверена/создана.")
    # Отступ 4 пробела (соответствует try)
    except Exception as e: # Добавлен except
        # Отступ 8 пробелов
        logger.error(f"Ошибка при инициализации таблицы monitored_lines: {e}\n{traceback.format_exc()}")
        raise

@app.on_event("startup")
async def startup_event():
    initialize_database() # Создаем таблицы при старте
    logger.info("Приложение запущено.")
    # Убираем фоновую задачу очистки кеша, т.к. модель создается по запросу
    # asyncio.create_task(refresh_data_cache())

# Добавляем импорт numpy
import numpy as np

# --- Новый эндпоинт для расширенной аналитики --- 
@app.get('/api/lines/{area_name}/analytics', response_model=AnalyticsResponse, tags=["Аналитика"])
def get_line_analytics(
    area_name: str, 
    minutes: int = Query(60, ge=1, description="Количество последних минут для анализа"),
    db: Session = Depends(get_db_session)
):
    """Получить расширенную аналитику по числовым данным линии за последние N минут."""
    logger.info(f"Запрос аналитики для линии: {area_name} за последние {minutes} минут.")
    
    # 1. Проверяем существование линии
    line = db.query(MonitoredLine).filter(MonitoredLine.area_name == area_name).first()
    if not line:
        logger.warning(f"Линия {area_name} не найдена при запросе аналитики.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Линия '{area_name}' не найдена.")

    # 2. Получаем модель таблицы данных
    DataModel = get_table_model(area_name)
    if not DataModel or not hasattr(DataModel, 'Время'):
        logger.error(f"Не удалось получить модель данных или отсутствует колонка 'Время' для {area_name}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка структуры таблицы данных для '{area_name}'.")

    # 3. Определяем временной интервал
    end_time = datetime.datetime.now() # Или можно брать func.now() из БД? Для простоты берем время сервера
    start_time = end_time - datetime.timedelta(minutes=minutes)
    logger.debug(f"Интервал для аналитики {area_name}: {start_time} - {end_time}")
    
    # 4. Запрашиваем данные за интервал с помощью pd.read_sql_query
    try:
        stmt = select(DataModel).where(
            DataModel.Время >= start_time,
            DataModel.Время <= end_time
        ).order_by(DataModel.Время.asc()) # Сортируем по возрастанию для удобства

        # Используем pd.read_sql_query для прямого создания DataFrame
        # Передаем сам объект statement и подключение SQLAlchemy
        df = pd.read_sql_query(sql=stmt, con=db.bind) # db.bind - это подключение к БД

        if df.empty:
             logger.warning(f"Нет данных для линии {area_name} за последние {minutes} минут (используя pd.read_sql_query).")
             # Возвращаем пустой ответ, но с информацией о запросе
             return AnalyticsResponse(
                 area_name=area_name,
                 start_time=start_time,
                 end_time=end_time,
                 requested_minutes=minutes
             )

        logger.debug(f"Получено {len(df)} строк и {len(df.columns)} колонок из БД для {area_name} с помощью pd.read_sql_query.")
        logger.debug(f"Размер DataFrame после создания: {df.shape}") # Теперь ожидаем (N, M), где M > 1

    except OperationalError as db_lock_err:
         logger.error(f"База данных заблокирована при запросе аналитики для {area_name}: {db_lock_err}")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="База данных временно недоступна, попробуйте позже.")
    except Exception as e:
        logger.error(f"Ошибка при запросе данных для аналитики {area_name} с pd.read_sql_query: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка получения данных для аналитики: {str(e)}")

    # 5. Рассчитываем статистику для числовых колонок
    statistics: Dict[str, ColumnAnalytics] = {}
    numeric_columns = df.select_dtypes(include=np.number).columns
    # Исключаем 'id', если он есть
    numeric_columns = [col for col in numeric_columns if col.lower() != 'id'] 

    for col_name in numeric_columns:
        col_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
        count = len(col_data)
        if count > 0:
            stats = ColumnAnalytics(
                count=count,
                min=col_data.min(),
                max=col_data.max(),
                average=col_data.mean(),
                median=col_data.median(),
                std_dev=col_data.std()
            )
        else:
            stats = ColumnAnalytics(count=0)
        statistics[col_name] = stats
        logger.debug(f"Статистика для {area_name} / {col_name}: {stats.dict()}")

    # 6. Подготавливаем данные для трендов для ВСЕХ числовых колонок (кроме id)
    trends: Dict[str, List[TrendDataPoint]] = {}
    if 'Время' in df.columns:
        # Убедимся, что Время - это datetime и оно содержит валидные значения
        df['Время'] = pd.to_datetime(df['Время'], errors='coerce')
        df_valid_time = df.dropna(subset=['Время']) # Работаем только со строками с валидным временем

        if not df_valid_time.empty:
            # Используем список числовых колонок, который мы получили на шаге 5 (numeric_columns)
            # numeric_columns уже не содержит 'id'
            for col_name in numeric_columns: # <--- Итерируем по ВСЕМ числовым колонкам
                if col_name in df_valid_time.columns:
                    # Берем только валидные числовые значения и время для них
                    # Используем df_valid_time, чтобы гарантировать наличие валидного 'Время'
                    trend_df = df_valid_time[['Время', col_name]].copy() # Создаем копию, чтобы избежать SettingWithCopyWarning
                    trend_df[col_name] = pd.to_numeric(trend_df[col_name], errors='coerce')
                    trend_df = trend_df.dropna(subset=[col_name]) # Удаляем строки, где значение колонки не числовое

                    if not trend_df.empty:
                        logger.debug(f"Колонка {col_name}: Найдено {len(trend_df)} валидных точек для тренда.")
                        trends[col_name] = [
                            TrendDataPoint(Время=row['Время'].to_pydatetime(), value=row[col_name])
                            for _, row in trend_df.iterrows()
                        ]
                        logger.debug(f"Подготовлено {len(trends[col_name])} точек тренда для {area_name} / {col_name}.")
                    else:
                         logger.debug(f"Колонка {col_name}: Не найдено валидных числовых точек для тренда после очистки.")
                # else: # Это условие теперь не нужно, т.к. numeric_columns гарантированно есть в df
                #    logger.warning(f"Колонка {col_name} для тренда не найдена в данных {area_name}.")
        else:
             logger.warning(f"Нет строк с валидным временем для подготовки трендов в {area_name}.")
    else:
        logger.error(f"Критическая ошибка: Колонка 'Время' отсутствует в DataFrame для {area_name}, тренды не могут быть созданы.")

    # 7. Формируем и возвращаем ответ
    response_data = AnalyticsResponse(
        area_name=area_name,
        start_time=df['Время'].min() if 'Время' in df and not df['Время'].empty else start_time, # Фактическое начало данных
        end_time=df['Время'].max() if 'Время' in df and not df['Время'].empty else end_time,       # Фактическое окончание данных
        requested_minutes=minutes,
        statistics=statistics,
        trends=trends
    )
    logger.info(f"Аналитика для {area_name} успешно рассчитана.")
    return response_data


if __name__ == '__main__':
    logger.info(f"Текущая рабочая директория: {os.getcwd()}")
    db_path = Config.DATABASE_URL.replace('sqlite:///', '')
    logger.info(f"Попытка подключения к БД: {os.path.abspath(db_path)}")
    
    # Создаем директорию для БД, если ее нет
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
            logger.info(f"Создана директория для базы данных: {db_dir}")
        except OSError as e:
            logger.error(f"Не удалось создать директорию {db_dir}: {e}")
            # Решаем, критично ли это. Скорее всего, да.
            raise
            
    # Инициализация БД перед запуском uvicorn
    initialize_database()
    
    # Запуск uvicorn с возможностью перезагрузки при отладке
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=Config.DEBUG)
