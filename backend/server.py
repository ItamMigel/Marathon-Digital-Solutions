import datetime
import logging
import os
import asyncio
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

# Для работы с данными
import pandas as pd

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, Date, Float, DateTime, inspect, text
from sqlalchemy.ext.declarative import declared_attr, declarative_base
from sqlalchemy.orm import declarative_base, Session, sessionmaker

# FastAPI
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='server.log', encoding='utf-8', filemode='a')
logger = logging.getLogger("server")

# Конфигурация приложения
class Config:
    DATABASE_URL = "sqlite:///../value.db"  # Match the path from main.py
    DEBUG = True
    API_VERSION = "1.0.0"
    API_TITLE = "Система телеметрии"
    API_DESCRIPTION = "API для работы с данными телеметрии"

# Инициализация FastAPI
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация базы данных
Base = declarative_base()

# Создание движка базы данных
engine = create_engine(Config.DATABASE_URL, echo=Config.DEBUG)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Логирование пути к базе данных
import os
db_path = Config.DATABASE_URL.replace('sqlite:///', '')
logger.info(f"Using database at path: {db_path}")
absolute_path = os.path.abspath(db_path)
logger.info(f"Absolute path to database: {absolute_path}")
if os.path.exists(absolute_path):
    logger.info(f"Database file exists: {absolute_path}")
else:
    logger.warning(f"Database file NOT FOUND: {absolute_path}")
    # Попытка найти базу данных в других возможных местах
    possible_paths = [
        'value.db',
        '../value.db',
        '../../value.db',
        '../../../value.db',
        os.path.join(os.getcwd(), 'value.db')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found database at: {path}")
            # Обновляем URL базы данных
            Config.DATABASE_URL = f"sqlite:///{path}"
            engine = create_engine(Config.DATABASE_URL, echo=Config.DEBUG)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            break

# Менеджер контекста для сессий БД
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка базы данных: {str(e)}")
        raise
    finally:
        db.close()

# Зависимость для получения сессии БД в эндпоинтах
def get_db_session():
    with get_db() as session:
        yield session

# Базовая модель для всех таблиц
class BaseDBModel:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    id = Column(Integer, primary_key=True)
    
    @classmethod
    def get_by_id(cls, db: Session, id: int):
        return db.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def get_all(cls, db: Session, skip: int = 0, limit: int = 100):
        return db.query(cls).offset(skip).limit(limit).all()
    
    def save(self, db: Session):
        db.add(self)
        db.commit()
        db.refresh(self)
        return self
    
    def update(self, db: Session, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self.save(db)
    
    def delete(self, db: Session):
        db.delete(self)
        db.commit()
        return True

# Schemas for API
class TableInfo(BaseModel):
    name: str
    columns: List[str]

class TableData(BaseModel):
    table_name: str
    columns: List[str]
    data: List[Dict[str, Any]]
    total_records: int

class AnalysisResult(BaseModel):
    table_name: str
    production_line: Optional[int] = None
    summary: Dict[str, Any]
    recommendations: List[str]

# Кеш для хранения созданных моделей
_model_cache = {}
model_cache = {}

# Get dynamic model for a table
def get_table_model(table_name: str):
    if table_name in model_cache:
        return model_cache[table_name]
    
    # Inspect the table to get column information
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns(table_name)}
    
    if not columns:
        return None
    
    # Create a dynamic model
    class DynamicTable(Base):
        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}
        
        # Add columns dynamically based on database inspection
        for col_name, col_info in columns.items():
            locals()[col_name] = Column(
                col_info["type"], 
                primary_key=col_info.get("primary_key", False)
            )
    
    model_cache[table_name] = DynamicTable
    return DynamicTable

# Фабрика моделей
def create_data_model(area_name):
    # Если модель уже создана - возвращаем её из кеша
    if area_name in _model_cache:
        return _model_cache[area_name]
    
    class Data(Base, BaseDBModel):
        __tablename__ = area_name
        
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
        Номер_производственной_линии = Column(Integer, index=True)

        def __repr__(self):
            return f'''(
                id: {self.id},
                Время: {self.Время},
                Мощность_МПСИ_кВт: {self.Мощность_МПСИ_кВт},
                Мощность_МШЦ_кВт: {self.Мощность_МШЦ_кВт},
                Ток_МПСИ_А: {self.Ток_МПСИ_А},
                Ток_МШЦ_А: {self.Ток_МШЦ_А},
                Исходное_питание_МПСИ_т_ч: {self.Исходное_питание_МПСИ_т_ч},
                Возврат_руды_МПСИ_т_ч: {self.Возврат_руды_МПСИ_т_ч},
                Общее_питание_МПСИ_т_ч: {self.Общее_питание_МПСИ_т_ч},
                Расход_воды_МПСИ_PV_м3_ч: {self.Расход_воды_МПСИ_PV_м3_ч},
                Расход_воды_МПСИ_SP_м3_ч: {self.Расход_воды_МПСИ_SP_м3_ч},
                Расход_воды_МПСИ_CV_процент: {self.Расход_воды_МПСИ_CV_процент},
                факт_соотношение_руда_вода_МПСИ: {self.факт_соотношение_руда_вода_МПСИ},
                Давление_на_подшипник_МПСИ_загрузка_Бар: {self.Давление_на_подшипник_МПСИ_загрузка_Бар},
                Давление_на_подшипник_МПСИ_разгрузка_Бар: {self.Давление_на_подшипник_МПСИ_разгрузка_Бар},
                Температура_масла_основной_маслостанции_подача_МПСИ: {self.Температура_масла_основной_маслостанции_подача_МПСИ},
                Температура_масла_основной_маслостанции_слив_МПСИ: {self.Температура_масла_основной_маслостанции_слив_МПСИ},
                Температура_масла_маслостанции_электродвигатель_МПСИ: {self.Температура_масла_маслостанции_электродвигатель_МПСИ},
                Температура_масла_редуктора_МПСИ: {self.Температура_масла_редуктора_МПСИ},
                Давление_на_подшипник_МШЦ_загрузка_Бар: {self.Давление_на_подшипник_МШЦ_загрузка_Бар},
                Давление_на_подшипник_МШЦ_разгрузка_Бар: {self.Давление_на_подшипник_МШЦ_разгрузка_Бар},
                Температура_масла_основной_маслостанции_подача_МШЦ: {self.Температура_масла_основной_маслостанции_подача_МШЦ},
                Температура_масла_основной_маслостанции_слив_МШЦ: {self.Температура_масла_основной_маслостанции_слив_МШЦ},
                Температура_масла_маслостанции_электродвигатель_МШЦ: {self.Температура_масла_маслостанции_электродвигатель_МШЦ},
                Температура_масла_редуктора_МШЦ: {self.Температура_масла_редуктора_МШЦ},
                Расход_извести_МШЦ_л_ч: {self.Расход_извести_МШЦ_л_ч},
                Уровень_в_зумпфе_процент: {self.Уровень_в_зумпфе_процент},
                Обороты_насоса_процент: {self.Обороты_насоса_процент},
                Давление_в_ГЦ_насоса_Бар: {self.Давление_в_ГЦ_насоса_Бар},
                Плотность_слива_ГЦ_кг_л: {self.Плотность_слива_ГЦ_кг_л},
                pH_оборотной_воды: {self.pH_оборотной_воды},
                t_оборотной_воды: {self.t_оборотной_воды},
                Гранулометрия_процент: {self.Гранулометрия_процент},
                Поток_л_мин: {self.Поток_л_мин},
                Расход_оборотной_воды_м3_ч: {self.Расход_оборотной_воды_м3_ч},
                Расход_в_ГЦ_насоса_м3_ч: {self.Расход_в_ГЦ_насоса_м3_ч},
                Номер_производственной_линии: {self.Номер_производственной_линии}
            )'''
        
        def __str__(self):
            return f'''(
                    {self.id},
                    {self.Время},
                    {self.Гранулометрия_процент},
                    {self.Поток_л_мин},
                    {self.Номер_производственной_линии}
                )'''
        
        @classmethod
        def get_by_date(cls, db: Session, date: datetime.datetime, limit: int = 10):
            try:
                result = db.query(cls).filter(cls.Время == date).limit(limit).all()
                if not result:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Данные на дату {date} не найдены"
                    )
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Ошибка базы данных: {status.HTTP_404_NOT_FOUND}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Данные на дату {date} не найдены"
                )
    
    # Сохраняем модель в кеш
    _model_cache[area_name] = Data
    return Data

# Дополнительная функция для проверки и создания таблицы
def ensure_table_exists(area_name):
    # Получаем или создаем модель таблицы
    DataModel = create_data_model(area_name)
    
    # Проверяем существование таблицы
    inspector = inspect(engine)
    if area_name not in inspector.get_table_names():
        # Создаем таблицу, если она не существует
        DataModel.__table__.create(engine)
        logger.info(f"Динамически создана таблица: {area_name}")
    
    return DataModel

# Сервис для работы с данными
class DataService:
    @staticmethod
    def add_from_json(db: Session, json_data: Dict[str, Any], area_name='value_1234'):
        try:
            # Проверяем и создаем таблицу при необходимости
            DataModel = ensure_table_exists(area_name)
            
            data_dict = json_data
            processed_dict = {key.replace(' ', '_').replace('/', '_'): value for key, value in data_dict.items()}
            
            data_instance = DataModel(**processed_dict)
            db.add(data_instance)
            db.commit()
            db.refresh(data_instance)
            return data_instance
        except Exception as e:
            db.rollback()
            logger.error(f"Ошибка при добавлении данных из JSON: {str(e)}")
            raise

    @staticmethod
    def add_from_pandas(db: Session, df: pd.DataFrame, area_name='default'):
        added_data = []
        try:
            # Проверяем и создаем таблицу при необходимости
            DataModel = ensure_table_exists(area_name)
            
            # Получаем список допустимых колонок модели
            valid_columns = [c.name for c in DataModel.__table__.columns]
            
            # Преобразуем колонку 'Время' в объекты datetime перед циклом
            # Используем errors='coerce' для замены невалидных дат на NaT (Not a Time)
            if 'Время' in df.columns and 'Время' in valid_columns:
                try:
                    df['Время'] = pd.to_datetime(df['Время'], errors='coerce')
                    # Логируем, если были ошибки преобразования
                    failed_conversion = df['Время'].isna().sum()
                    if failed_conversion > 0:
                        logger.warning(f"В таблице '{area_name}' найдено {failed_conversion} строк с неверным форматом 'Время'. Эти строки будут пропущены.")
                except Exception as e:
                    logger.error(f"Ошибка при преобразовании колонки 'Время' в datetime для таблицы '{area_name}': {e}. Попытка продолжить без преобразования.")
            elif 'Время' in valid_columns:
                 logger.warning(f"Колонка 'Время' ожидается в таблице '{area_name}', но не найдена в DataFrame.")
                 # Обработка ситуации, когда колонка 'Время' отсутствует, но необходима

            for _, row in df.iterrows():
                data_dict = row.to_dict()

                # Пропускаем строки, где 'Время' не удалось преобразовать (стало NaT)
                # или если 'Время' обязательно и отсутствует
                if 'Время' in valid_columns:
                    время_value = data_dict.get('Время')
                    if pd.isna(время_value):
                         logger.warning(f"Пропуск строки из-за некорректного или отсутствующего значения 'Время': {row.to_dict()}")
                         continue

                # Отфильтровываем только те ключи, которые есть в модели
                filtered_dict = {k: v for k, v in data_dict.items() if k in valid_columns}

                if not filtered_dict:
                    # Преобразуем колонки, заменяя пробелы на подчеркивания и сохраняя единицы измерения
                    processed_dict = {}
                    for key, value in data_dict.items():
                        # Заменяем пробелы на подчеркивания и слеш на подчеркивание
                        processed_key = key.replace(' ', '_').replace('/', '_')

                        # Проверяем, есть ли такой ключ в допустимых колонках
                        if processed_key in valid_columns:
                            processed_dict[processed_key] = value
                        else:
                            # Пробуем другие варианты форматирования ключа
                            # Убираем возможные скобки, точки и другие спецсимволы
                            clean_key = ''.join(c if c.isalnum() or c == '_' else '_' for c in processed_key)
                            if clean_key in valid_columns:
                                processed_dict[clean_key] = value
                            else:
                                # Поиск ближайшего соответствия по ключевым словам
                                for column in valid_columns:
                                    # Проверяем, содержит ли допустимая колонка основные части ключа
                                    if all(part in column for part in processed_key.split('_') if len(part) > 2):
                                        processed_dict[column] = value
                                        break
                else:
                    processed_dict = filtered_dict

                # Дополнительная проверка на наличие 'Время' перед созданием объекта
                if 'Время' in valid_columns and ('Время' not in processed_dict or pd.isna(processed_dict.get('Время'))):
                     logger.warning(f"Пропуск строки из-за отсутствия или некорректного значения 'Время' после обработки: {data_dict}")
                     continue

                try:
                    # Убираем NaT значения перед передачей в модель (если они остались)
                    final_dict = {k: v for k, v in processed_dict.items() if not pd.isna(v)}
                    data_instance = DataModel(**final_dict)
                    db.add(data_instance)
                    added_data.append(data_instance)
                except Exception as e:
                    # Пропускаем записи с ошибками, но логируем их
                    logger.error(f"Ошибка при добавлении записи: {str(e)}")
                    logger.error(f"Проблемные данные: {processed_dict}")
                    continue

            db.commit()
            return added_data
        except Exception as e:
            db.rollback()
            logger.error(f"Ошибка при добавлении данных из DataFrame: {str(e)}")
            raise

    @staticmethod
    def to_response_dict(data) -> Dict[str, Any]:
        return {
            "id": data.id,
            "date": str(data.Время),
            "power_MPSI_kW": data.Мощность_МПСИ_кВт,
            "power_MHC_kW": data.Мощность_МШЦ_кВт,
            "tok_MPSI_A": data.Ток_МПСИ_А,
            "tok_MHC_A": data.Ток_МШЦ_А,
            "source_feed_MPSI_t_h": data.Исходное_питание_МПСИ_т_ч,
            "return_ore_MPSI_t_h": data.Возврат_руды_МПСИ_т_ч,
            "total_feed_MPSI_t_h": data.Общее_питание_МПСИ_т_ч,
            "water_flow_MPSI_PV_m3_h": data.Расход_воды_МПСИ_PV_м3_ч,
            "water_flow_MPSI_SP_m3_h": data.Расход_воды_МПСИ_SP_м3_ч,
            "water_flow_MPSI_CV_percent": data.Расход_воды_МПСИ_CV_процент,
            "ratio_water_ore_MPSI": data.факт_соотношение_руда_вода_МПСИ,
            "pressure_bearing_MPSI_loading_Bar": data.Давление_на_подшипник_МПСИ_загрузка_Бар,
            "pressure_bearing_MPSI_unloading_Bar": data.Давление_на_подшипник_МПСИ_разгрузка_Бар,
            "temp_oil_main_oil_station_supply_MPSI": data.Температура_масла_основной_маслостанции_подача_МПСИ,
            "temp_oil_main_oil_station_drain_MPSI": data.Температура_масла_основной_маслостанции_слив_МПСИ,
            "temp_oil_oil_station_electric_motor_MPSI": data.Температура_масла_маслостанции_электродвигатель_МПСИ,
            "temp_oil_reducer_MPSI": data.Температура_масла_редуктора_МПСИ,
            "pressure_bearing_MHC_loading_Bar": data.Давление_на_подшипник_МШЦ_загрузка_Бар,
            "pressure_bearing_MHC_unloading_Bar": data.Давление_на_подшипник_МШЦ_разгрузка_Бар,
            "temp_oil_main_oil_station_supply_MHC": data.Температура_масла_основной_маслостанции_подача_МШЦ,
            "temp_oil_main_oil_station_drain_MHC": data.Температура_масла_основной_маслостанции_слив_МШЦ,
            "temp_oil_oil_station_electric_motor_MHC": data.Температура_масла_маслостанции_электродвигатель_МШЦ,
            "temp_oil_reducer_MHC": data.Температура_масла_редуктора_МШЦ,
            "lime_flow_MHC_l_h": data.Расход_извести_МШЦ_л_ч,
            "sump_level_percent": data.Уровень_в_зумпфе_процент,
            "pump_speed_percent": data.Обороты_насоса_процент,
            "pressure_GC_pump_Bar": data.Давление_в_ГЦ_насоса_Бар,
            "density_GC_drain_kg_l": data.Плотность_слива_ГЦ_кг_л,
            "pH_circulating_water": data.pH_оборотной_воды,
            "t_circulating_water": data.t_оборотной_воды,
            "granulometry_percent": data.Гранулометрия_процент,
            "flow_l_min": data.Поток_л_мин,
            "circulation_water_flow_m3_h": data.Расход_оборотной_воды_м3_ч,
            "GC_pump_flow_m3_h": data.Расход_в_ГЦ_насоса_м3_ч,
            "production_line_number": data.Номер_производственной_линии
        }

# Инициализация базы данных
def create_db(area_names=[]):
    try:
        # Если не указаны имена таблиц, проверим уже существующие
        if not area_names:
            inspector = inspect(engine)
            area_names = inspector.get_table_names()
            logger.info(f"Found existing tables: {area_names}")
        
        # Создаем динамические таблицы
        for area_name in area_names:
            data_model = create_data_model(area_name)
            logger.info(f"Created model for table: {area_name}")
            
            # Добавляем в оба кеша для совместимости
            _model_cache[area_name] = data_model
            model_cache[area_name] = data_model
            
        # Создаем все таблицы в базе данных
        Base.metadata.create_all(bind=engine)
        logger.info("База данных успешно создана или обновлена")
    except Exception as e:
        logger.error(f"Ошибка при создании базы данных: {str(e)}")
        raise

# Pydantic модели для API
class ValuesRequest(BaseModel):
    date: datetime.datetime
    limit: int = Field(default=10, ge=1, le=100)
    area_name: List[str] = Field(default=[], description="[] - поиск по всем участкам")
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2023-01-01T00:00:00",
                "limit": 10,
                "area_name": [],
            }
        }

class DataResponse(BaseModel):
    id: int
    date: str
    power_MPSI_kW: Optional[float] = None
    power_MHC_kW: Optional[float] = None
    tok_MPSI_A: Optional[float] = None
    tok_MHC_A: Optional[float] = None
    source_feed_MPSI_t_h: Optional[float] = None
    return_ore_MPSI_t_h: Optional[float] = None
    total_feed_MPSI_t_h: Optional[float] = None
    water_flow_MPSI_PV_m3_h: Optional[float] = None
    water_flow_MPSI_SP_m3_h: Optional[float] = None
    water_flow_MPSI_CV_percent: Optional[float] = None
    ratio_water_ore_MPSI: Optional[float] = None
    pressure_bearing_MPSI_loading_Bar: Optional[float] = None
    pressure_bearing_MPSI_unloading_Bar: Optional[float] = None
    temp_oil_main_oil_station_supply_MPSI: Optional[float] = None
    temp_oil_main_oil_station_drain_MPSI: Optional[float] = None
    temp_oil_oil_station_electric_motor_MPSI: Optional[float] = None
    temp_oil_reducer_MPSI: Optional[float] = None
    pressure_bearing_MHC_loading_Bar: Optional[float] = None
    pressure_bearing_MHC_unloading_Bar: Optional[float] = None
    temp_oil_main_oil_station_supply_MHC: Optional[float] = None
    temp_oil_main_oil_station_drain_MHC: Optional[float] = None
    temp_oil_oil_station_electric_motor_MHC: Optional[float] = None
    temp_oil_reducer_MHC: Optional[float] = None
    lime_flow_MHC_l_h: Optional[float] = None
    sump_level_percent: Optional[float] = None
    pump_speed_percent: Optional[float] = None
    pressure_GC_pump_Bar: Optional[float] = None
    density_GC_drain_kg_l: Optional[float] = None
    pH_circulating_water: Optional[float] = None
    t_circulating_water: Optional[float] = None
    granulometry_percent: Optional[float] = None
    flow_l_min: Optional[float] = None
    circulation_water_flow_m3_h: Optional[float] = None
    GC_pump_flow_m3_h: Optional[float] = None
    production_line_number: Optional[int] = None

# Функции для работы с данными
def add_from_json(json_data, area_name='default'):
    with get_db() as db:
        return DataService.add_from_json(db, json_data, area_name=area_name)

def add_from_pandas(df, area_name='default'):
    with get_db() as db:
        return DataService.add_from_pandas(db, df, area_name=area_name)

def load_data(file_content=None, filename=None, table_name=None, file_type='parquet', file_path=None):
    """
    Load data from uploaded file content or file path into the specified database table.
    
    Args:
        file_content (bytes, optional): Content of the uploaded file
        filename (str, optional): Name of the uploaded file
        table_name (str): Name of the table where data should be loaded (will be created if doesn't exist)
        file_type (str): Type of the file ('parquet' or 'excel')
        file_path (str, optional): Path to the file to load data from (alternative to file_content)
    
    Returns:
        List: List of added data objects
    """
    try:
        import tempfile
        import os
        
        # Определение режима работы (загрузка содержимого или чтение из файла)
        using_file_content = file_content is not None
        using_file_path = file_path is not None and os.path.exists(file_path)
        
        if not using_file_content and not using_file_path:
            raise ValueError("Необходимо указать либо содержимое файла (file_content), либо путь к файлу (file_path)")
        
        if not table_name:
            raise ValueError("Необходимо указать имя таблицы (table_name)")
        
        # Создаем временный файл, если используем file_content
        temp_file_path = None
        if using_file_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
        
        try:
            # Обрабатываем файл в зависимости от типа
            if file_type.lower() == 'parquet':
                df = pd.read_parquet(temp_file_path if using_file_content else file_path)
            elif file_type.lower() == 'excel':
                df = pd.read_excel(temp_file_path if using_file_content else file_path)
            else:
                raise ValueError(f"Неподдерживаемый тип файла: {file_type}. Используйте 'parquet' или 'excel'.")
            
            # Словарь соответствия имен полей из файла в имена полей базы данных
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
            
            # Переименовываем колонки в DataFrame
            renamed_columns = {}
            for old_name in df.columns:
                if old_name in field_mapping:
                    renamed_columns[old_name] = field_mapping[old_name]
            
            if renamed_columns:
                df = df.rename(columns=renamed_columns)
            
            # Загружаем данные в базу данных
            result = add_from_pandas(df, area_name=table_name)
            logger.info(f"Успешно загружено {len(result)} записей из {file_type} файла в таблицу '{table_name}'")
            return result
        finally:
            # Очищаем временный файл, если он был создан
            if using_file_content and temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Ошибка загрузки данных из {file_type} файла: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка загрузки данных: {str(e)}"
        )

# API эндпоинты
@app.get('/', tags=["Статус"])
def read_root():
    return {
        'message': 'Кокоджамбо всё',
        'version': Config.API_VERSION,
        'status': 'online'
    }

@app.post('/value', response_model=List[DataResponse], tags=["Данные"])
def get_value(value: ValuesRequest, db: Session = Depends(get_db_session)):
    try:
        date_post = value.date
        limit = value.limit
        
        results = []
        if len(value.area_name):
            for area_name in value.area_name:
                model = create_data_model(area_name)
                query = db.query(model).filter(model.Время == date_post)
                area_results = query.limit(limit).all()
                results.extend(area_results)
        else:
            for area_name, model in _model_cache.items():
                query = db.query(model).filter(model.Время == date_post)
                
                area_results = query.limit(limit).all()
                results.extend(area_results)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Данные на дату {date_post} не найдены"
            )
            
        return [DataService.to_response_dict(item) for item in results]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении данных: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

@app.post('/load-data', tags=["Загрузка данных"])
async def load_data_endpoint(
    file: UploadFile = File(..., description="Excel или Parquet файл с данными"),
    table_name: str = Form(..., description="Имя таблицы для загрузки данных"),
    file_type: str = Form(default="parquet", description="Тип файла (parquet или excel)")
):
    """
    Загрузка данных из файла (Excel или Parquet) в указанную таблицу базы данных.
    Таблица будет создана, если она не существует.
    """
    try:
        # Проверка типа файла
        if file_type.lower() not in ["parquet", "excel"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемый тип файла: {file_type}. Используйте 'parquet' или 'excel'."
            )
        
        # Проверка расширения файла
        file_extension = file.filename.split('.')[-1].lower()
        expected_extension = "xlsx" if file_type.lower() == "excel" else "parquet"
        
        if (file_type.lower() == "excel" and file_extension not in ["xlsx", "xls"]) or \
           (file_type.lower() == "parquet" and file_extension != "parquet"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Расширение файла не соответствует указанному типу. Ожидалось: {expected_extension}"
            )
        
        # Чтение содержимого файла
        file_content = await file.read()
        
        # Дополнительная диагностика файла
        import tempfile
        import os
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Чтение файла для предварительного анализа
            if file_type.lower() == 'parquet':
                df_preview = pd.read_parquet(temp_file_path)
            else:
                df_preview = pd.read_excel(temp_file_path)
            
            logger.info(f"Колонки в загружаемом файле: {df_preview.columns.tolist()}")
        finally:
            os.unlink(temp_file_path)
        
        # Загрузка данных
        result = load_data(
            file_content=file_content,
            filename=file.filename,
            table_name=table_name,
            file_type=file_type
        )
        
        return {
            "status": "success",
            "message": f"Успешно загружено {len(result)} записей в таблицу '{table_name}'",
            "records_count": len(result),
            "columns_in_file": df_preview.columns.tolist() if 'df_preview' in locals() else []
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка в процессе загрузки данных: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

@app.post('/load-data-from-path', tags=["Загрузка данных"])
async def load_data_from_path_endpoint(
    file_path: str = Form(..., description="Путь к Excel или Parquet файлу с данными"),
    table_name: str = Form(..., description="Имя таблицы для загрузки данных"),
    file_type: str = Form(default="parquet", description="Тип файла (parquet или excel)")
):
    """
    Загрузка данных из файла по указанному пути (Excel или Parquet) в указанную таблицу базы данных.
    Таблица будет создана, если она не существует.
    """
    try:
        # Проверка типа файла
        if file_type.lower() not in ["parquet", "excel"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемый тип файла: {file_type}. Используйте 'parquet' или 'excel'."
            )
        
        # Проверка существования файла
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Файл не найден по указанному пути: {file_path}"
            )
        
        # Проверка расширения файла
        file_extension = file_path.split('.')[-1].lower()
        expected_extension = "xlsx" if file_type.lower() == "excel" else "parquet"
        
        if (file_type.lower() == "excel" and file_extension not in ["xlsx", "xls"]) or \
           (file_type.lower() == "parquet" and file_extension != "parquet"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Расширение файла не соответствует указанному типу. Ожидалось: {expected_extension}"
            )
        
        # Чтение данных из файла
        try:
            # Предварительный анализ файла
            if file_type.lower() == 'parquet':
                df_preview = pd.read_parquet(file_path)
            else:
                df_preview = pd.read_excel(file_path)
            
            logger.info(f"Колонки в загружаемом файле: {df_preview.columns.tolist()}")
            
            # Чтение содержимого файла
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Получаем имя файла из пути
            filename = os.path.basename(file_path)
            
            # Загрузка данных
            result = load_data(
                file_content=file_content,
                filename=filename,
                table_name=table_name,
                file_type=file_type
            )
            
            return {
                "status": "success",
                "message": f"Успешно загружено {len(result)} записей из файла '{filename}' в таблицу '{table_name}'",
                "records_count": len(result),
                "columns_in_file": df_preview.columns.tolist()
            }
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка при чтении файла: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка в процессе загрузки данных из пути к файлу: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

# API Endpoints for new monitoring API
@app.get('/api/tables', response_model=List[TableInfo])
def get_tables(db: Session = Depends(get_db_session)):
    """Get all available tables in the database"""
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        tables = []
        for table_name in table_names:
            try:
                columns = [column["name"] for column in inspector.get_columns(table_name)]
                tables.append(TableInfo(name=table_name, columns=columns))
            except Exception as e:
                logger.error(f"Error getting columns for table {table_name}: {str(e)}")
        
        logger.info(f"Found {len(tables)} tables: {[t.name for t in tables]}")
        return tables
    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        # Не бросаем исключение, чтобы предотвратить отказ API
        return []

@app.get('/api/tables/{table_name}', response_model=TableData)
def get_table_data(
    table_name: str,
    db: Session = Depends(get_db_session),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    columns: Optional[str] = None,
    filters: Optional[str] = None,
):
    """Get data from a specific table with pagination and filtering"""
    try:
        # Get table model
        table_model = get_table_model(table_name)
        if not table_model:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
        
        # Get column information
        inspector = inspect(engine)
        all_columns = [column["name"] for column in inspector.get_columns(table_name)]
        
        # Filter selected columns or use all
        selected_columns = all_columns
        if columns:
            col_list = columns.split(",")
            selected_columns = [col for col in col_list if col in all_columns]
        
        # Base query
        query = db.query(table_model)
        
        # Apply filters if provided
        if filters:
            try:
                filter_parts = filters.split(",")
                for part in filter_parts:
                    if ":" in part and "=" in part:
                        col_name, operator, value = part.split(":")
                        if col_name in all_columns:
                            column = getattr(table_model, col_name)
                            if operator == "eq":
                                query = query.filter(column == value)
                            elif operator == "gt":
                                query = query.filter(column > float(value))
                            elif operator == "lt":
                                query = query.filter(column < float(value))
                            elif operator == "contains":
                                query = query.filter(column.like(f"%{value}%"))
            except Exception as e:
                logger.error(f"Error applying filters: {str(e)}")
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        # Execute query
        results = query.all()
        
        # Convert results to dict
        data = []
        for row in results:
            row_dict = {}
            for col in selected_columns:
                row_dict[col] = getattr(row, col)
            data.append(row_dict)
        
        return TableData(
            table_name=table_name,
            columns=selected_columns,
            data=data,
            total_records=total_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get('/api/analysis/{table_name}', response_model=AnalysisResult)
def get_analysis(
    table_name: str, 
    production_line: Optional[int] = None,
    db: Session = Depends(get_db_session)
):
    """Get analysis results for a specific table/production line"""
    try:
        # Placeholder for actual analysis logic
        # In a real application, this would perform actual data analysis
        
        summary = {
            "average_values": {
                "power": 125.7,
                "temperature": 85.2,
                "efficiency": 78.9
            },
            "trends": {
                "power_trend": "increasing",
                "temperature_trend": "stable",
                "efficiency_trend": "decreasing"
            },
            "anomalies_detected": 3
        }
        
        recommendations = [
            "Consider maintenance for production line due to efficiency decrease",
            "Monitor power consumption trends over next 24 hours",
            "Temperature readings are within normal parameters"
        ]
        
        return AnalysisResult(
            table_name=table_name,
            production_line=production_line,
            summary=summary,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error getting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Background task to periodically refresh data cache
async def refresh_data_cache():
    """Background task to refresh data cache every 10 minutes"""
    while True:
        try:
            logger.info("Refreshing data cache...")
            # In a real application, this would update any cached data
            
            # Reset model cache to ensure fresh schema
            model_cache.clear()
            
            logger.info("Data cache refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing data cache: {str(e)}")
        
        # Wait for 10 minutes before refreshing again
        await asyncio.sleep(600)  # 600 seconds = 10 minutes

@app.on_event("startup")
async def startup_event():
    """Start background tasks when application starts"""
    # Initialize table models from the database
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        logger.info(f"Found {len(table_names)} tables in database: {table_names}")
        
        # Initialize models for each table
        for table_name in table_names:
            try:
                table_model = get_table_model(table_name)
                logger.info(f"Successfully initialized model for table: {table_name}")
            except Exception as e:
                logger.error(f"Error initializing model for table {table_name}: {str(e)}")
        
        # Create sample table if none exists
        if not table_names:
            logger.warning("No tables found in database. Creating sample tables.")
            create_db([])
    except Exception as e:
        logger.error(f"Error during startup initialization: {str(e)}")
    
    # Start the data refresh task
    asyncio.create_task(refresh_data_cache())
    logger.info("Application started, background tasks initialized")

if __name__ == '__main__':
    # Print the current working directory to help with debugging
    import os
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Attempting to connect to database at: {os.path.abspath(Config.DATABASE_URL.replace('sqlite:///', ''))}")
    
    create_db()
    
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=Config.DEBUG)