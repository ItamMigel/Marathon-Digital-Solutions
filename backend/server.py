from sqlalchemy import create_engine, Column, Integer, String, Date, Float, DateTime
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from sqlalchemy.ext.declarative import declared_attr
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime
from contextlib import contextmanager
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='server.log', encoding='utf-8', filemode='a')
logger = logging.getLogger(__name__)

# Конфигурация приложения
class Config:
    DATABASE_URL = "sqlite:///value.db"
    DEBUG = True
    API_VERSION = "1.0.0"
    API_TITLE = "Система телеметрии"
    API_DESCRIPTION = "API для работы с данными телеметрии"

# Инициализация FastAPI
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Инициализация базы данных
Base = declarative_base()

# Создание движка базы данных
engine = create_engine(Config.DATABASE_URL, echo=Config.DEBUG)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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

# Модель данных телеметрии
class Data(Base, BaseDBModel):
    __tablename__ = "value"
    
    id = Column(Integer, primary_key=True)
    Время = Column(DateTime, index=True)
    Мощность_МПСИ = Column(Float)
    Мощность_МШЦ = Column(Float)
    Ток_МПСИ = Column(Float)
    Ток_МШЦ = Column(Float)
    Питание_МПСИ = Column(Float)
    Возврат_руды_МПСИ = Column(Float)
    Расход_воды_МПСИ_PV = Column(Float)
    Расход_воды_МПСИ_SP = Column(Float)
    Расход_воды_МПСИ_CV = Column(Float)
    факт_соотношение_руда_вода_МПСИ = Column(Float)
    Давление_на_подшипник_МПСИ_загрузка = Column(Float)
    Давление_на_подшипник_МПСИ_разгрузка = Column(Float)
    Расход_оборотной_воды = Column(Float)
    pH_оборотной_воды = Column(Float)
    t_оборотной_воды = Column(Float)
    Гранулометрия = Column(Float)
    Поток = Column(Float)

    def __repr__(self):
        return f'''(
            id: {self.id},
            Время: {self.Время},
            Мощность_МПСИ: {self.Мощность_МПСИ},
            Мощность_МШЦ: {self.Мощность_МШЦ},
            Ток_МПСИ: {self.Ток_МПСИ},
            Ток_МШЦ: {self.Ток_МШЦ},
            Питание_МПСИ: {self.Питание_МПСИ},
            Возврат_руды_МПСИ: {self.Возврат_руды_МПСИ},
            Расход_воды_МПСИ_PV: {self.Расход_воды_МПСИ_PV},
            Расход_воды_МПСИ_SP: {self.Расход_воды_МПСИ_SP},
            Расход_воды_МПСИ_CV: {self.Расход_воды_МПСИ_CV},
            факт_соотношение_руда_вода_МПСИ: {self.факт_соотношение_руда_вода_МПСИ},
            Давление_на_подшипник_МПСИ_загрузка: {self.Давление_на_подшипник_МПСИ_загрузка},
            Давление_на_подшипник_МПСИ_разгрузка: {self.Давление_на_подшипник_МПСИ_разгрузка},
            Расход_оборотной_воды: {self.Расход_оборотной_воды},
            pH_оборотной_воды: {self.pH_оборотной_воды},
            t_оборотной_воды: {self.t_оборотной_воды},
            Гранулометрия: {self.Гранулометрия},
            Поток: {self.Поток}
        )'''
    
    def __str__(self):
       return f'''(
            {self.id},
            {self.Время},
            {self.Гранулометрия},
            {self.Поток}
        )'''
    
    @classmethod
    def get_by_date(cls, db: Session, date: datetime.date, limit: int = 10):
        return db.query(cls).filter(cls.Время.cast(Date) == date).limit(limit).all()

# Сервис для работы с данными
class DataService:
    @staticmethod
    def add_from_json(db: Session, json_data: Dict[str, Any]) -> Data:
        try:
            data_dict = json_data
            processed_dict = {key.replace(' ', '_').replace('/', '_'): value for key, value in data_dict.items()}
            data = Data(**processed_dict)
            db.add(data)
            db.commit()
            db.refresh(data)
            return data
        except Exception as e:
            db.rollback()
            logger.error(f"Ошибка при добавлении данных из JSON: {str(e)}")
            raise

    @staticmethod
    def add_from_pandas(db: Session, df: pd.DataFrame) -> List[Data]:
        added_data = []
        try:
            for _, row in df.iterrows():
                data_dict = row.to_dict()
                processed_dict = {key.replace(' ', '_').replace('/', '_'): value for key, value in data_dict.items()}
                data = Data(**processed_dict)
                db.add(data)
                added_data.append(data)
            db.commit()
            return added_data
        except Exception as e:
            db.rollback()
            logger.error(f"Ошибка при добавлении данных из DataFrame: {str(e)}")
            raise

    @staticmethod
    def to_response_dict(data: Data) -> Dict[str, Any]:
        return {
            "id": data.id,
            "Время": str(data.Время),
            "Мощность_МПСИ": data.Мощность_МПСИ,
            "Мощность_МШЦ": data.Мощность_МШЦ,
            "Ток_МПСИ": data.Ток_МПСИ,
            "Ток_МШЦ": data.Ток_МШЦ,
            "Питание_МПСИ": data.Питание_МПСИ,
            "Возврат_руды_МПСИ": data.Возврат_руды_МПСИ,
            "Расход_воды_МПСИ_PV": data.Расход_воды_МПСИ_PV,
            "Расход_воды_МПСИ_SP": data.Расход_воды_МПСИ_SP,
            "Расход_воды_МПСИ_CV": data.Расход_воды_МПСИ_CV,
            "факт_соотношение_руда_вода_МПСИ": data.факт_соотношение_руда_вода_МПСИ,
            "Давление_на_подшипник_МПСИ_загрузка": data.Давление_на_подшипник_МПСИ_загрузка,
            "Давление_на_подшипник_МПСИ_разгрузка": data.Давление_на_подшипник_МПСИ_разгрузка,
            "Расход_оборотной_воды": data.Расход_оборотной_воды,
            "pH_оборотной_воды": data.pH_оборотной_воды,
            "t_оборотной_воды": data.t_оборотной_воды,
            "Гранулометрия": data.Гранулометрия,
            "Поток": data.Поток
        }

# Инициализация базы данных
def create_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("База данных успешно создана")
    except Exception as e:
        logger.error(f"Ошибка при создании базы данных: {str(e)}")
        raise

# Pydantic модели для API
class ValuesRequest(BaseModel):
    date: datetime.datetime
    limit: int = Field(default=10, ge=1, le=100)
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2023-01-01T00:00:00",
                "limit": 10
            }
        }

class DataResponse(BaseModel):
    id: int
    date: str
    power_MPSI: Optional[float] = None
    power_MHC: Optional[float] = None
    tok_MPSI: Optional[float] = None
    tok_MHC: Optional[float] = None
    pitanie_MPSI: Optional[float] = None
    return_ore_MPSI: Optional[float] = None
    rashod_water_MPSI_PV: Optional[float] = None
    rashod_water_MPSI_SP: Optional[float] = None
    rashod_water_MPSI_CV: Optional[float] = None
    ratio_water_ore_MPSI: Optional[float] = None
    pressure_podshipnik_MPSI_zagruzka: Optional[float] = None
    pressure_podshipnik_MPSI_razgruzka: Optional[float] = None
    rashod_oborot_water: Optional[float] = None
    ph_oborot_water: Optional[float] = None
    t_oborot_water: Optional[float] = None
    granul: Optional[float] = None
    potok: Optional[float] = None

# Функции для работы с данными
def add_from_json(json_data):
    with get_db() as db:
        return DataService.add_from_json(db, json_data)

def add_from_pandas(df):
    with get_db() as db:
        return DataService.add_from_pandas(db, df)

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
        date_post = value.date.date()
        limit = value.limit
        
        results = Data.get_by_date(db, date_post, limit)
        
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

# Запуск приложения
if __name__ == '__main__':
    # Создаем базу данных перед запуском приложения
    create_db()
    
    # Загружаем данные только если таблица успешно создана
    try:
        df = pd.read_parquet('data/data_after_analys.parquet').head(100)
        add_from_pandas(df)
        df = pd.read_parquet('data/data_after_analys.parquet').tail(100)
        add_from_pandas(df)
        logger.info(f"Успешно загружено {len(df)} записей из parquet файла")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из parquet файла: {str(e)}")
    
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=Config.DEBUG)