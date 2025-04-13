import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
import datetime
from typing import Dict, List, Any, Optional
import os

# Настройка страницы
st.set_page_config(
    page_title="Система мониторинга гранулометрии",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Глобальные переменные
API_URL = os.environ.get("API_URL", "http://localhost:8000")
UPDATE_INTERVAL = 60  # Интервал обновления в секундах
HISTORY_LENGTH = 100  # Количество точек в истории
COLOR_THRESHOLD_LOW = 70  # Нижний порог для цветового кодирования
COLOR_THRESHOLD_HIGH = 85  # Верхний порог для цветового кодирования

# Кэш данных
site_data_cache = {}
prediction_history = {}
last_update_time = {}

# Функции для работы с API
def get_sites():
    """Получает список всех участков"""
    try:
        response = requests.get(f"{API_URL}/sites")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка при получении списка участков: {response.text}")
            return []
    except Exception as e:
        st.error(f"Ошибка при подключении к API: {str(e)}")
        return []

def get_site_details(site_id):
    """Получает детальную информацию об участке"""
    try:
        response = requests.get(f"{API_URL}/sites/{site_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка при получении данных участка: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка при подключении к API: {str(e)}")
        return None

def get_feature_importance(site_id):
    """Получает важность признаков для участка"""
    try:
        response = requests.get(f"{API_URL}/sites/{site_id}/importance")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка при получении важности признаков: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка при подключении к API: {str(e)}")
        return None

def send_prediction_request(site_id, features):
    """Отправляет запрос на предсказание"""
    try:
        payload = {
            "site_id": site_id,
            "features": features,
            "timestamp": datetime.datetime.now().isoformat()
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка при получении предсказания: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка при подключении к API: {str(e)}")
        return None

# Функции визуализации
def color_scale(value, min_val=COLOR_THRESHOLD_LOW, max_val=COLOR_THRESHOLD_HIGH):
    """Определяет цвет на основе значения гранулометрии"""
    if value < min_val:
        return "#ff0000"  # Красный для низких значений
    elif value > max_val:
        return "#00ff00"  # Зеленый для высоких значений
    else:
        # Градиент от красного к зеленому
        ratio = (value - min_val) / (max_val - min_val)
        r = int(255 * (1 - ratio))
        g = int(255 * ratio)
        return f"#{r:02x}{g:02x}00"

def create_gauge_chart(value, min_val=0, max_val=100, title="Гранулометрия"):
    """Создает график в виде датчика"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color_scale(value)},
            'steps': [
                {'range': [min_val, COLOR_THRESHOLD_LOW], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [COLOR_THRESHOLD_LOW, COLOR_THRESHOLD_HIGH], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [COLOR_THRESHOLD_HIGH, max_val], 'color': 'rgba(0, 255, 0, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_history_chart(history, site_name):
    """Создает график истории гранулометрии"""
    if not history:
        return None
    
    # Преобразуем историю в DataFrame
    df = pd.DataFrame(history)
    
    # Создаем подграфики
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"История гранулометрии - {site_name}", "Отклонение от среднего"),
        vertical_spacing=0.12
    )
    
    # График истории
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['prediction'], 
            mode='lines+markers',
            name='Гранулометрия',
            line=dict(color='blue', width=2),
            marker=dict(
                color=[color_scale(v) for v in df['prediction']],
                size=8
            )
        ),
        row=1, col=1
    )
    
    # Добавляем пороговые линии
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=COLOR_THRESHOLD_LOW,
        x1=df['timestamp'].max(),
        y1=COLOR_THRESHOLD_LOW,
        line=dict(color="Red", width=1, dash="dash"),
        row=1, col=1
    )
    
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(),
        y0=COLOR_THRESHOLD_HIGH,
        x1=df['timestamp'].max(),
        y1=COLOR_THRESHOLD_HIGH,
        line=dict(color="Green", width=1, dash="dash"),
        row=1, col=1
    )
    
    # График отклонений
    if len(df) > 1:
        mean_val = df['prediction'].mean()
        deviations = df['prediction'] - mean_val
        
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=deviations,
                marker_color=[color_scale(v) for v in df['prediction']],
                name='Отклонение'
            ),
            row=2, col=1
        )
        
        # Добавляем нулевую линию
        fig.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=0,
            x1=df['timestamp'].max(),
            y1=0,
            line=dict(color="Gray", width=1),
            row=2, col=1
        )
    
    # Форматирование графика
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    fig.update_xaxes(title_text="Время", row=2, col=1)
    fig.update_yaxes(title_text="Гранулометрия (%)", row=1, col=1)
    fig.update_yaxes(title_text="Отклонение (%)", row=2, col=1)
    
    return fig

def create_importance_chart(importance_data):
    """Создает график важности признаков"""
    if not importance_data or 'feature_importance' not in importance_data:
        return None
    
    # Извлекаем данные важности
    importance = importance_data['feature_importance']
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=True).tail(10)  # Берем топ-10
    
    # Создаем горизонтальную гистограмму
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Топ-10 важных признаков',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Bluered
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Интерфейс приложения
def main():
    """Основная функция приложения"""
    st.title("Система мониторинга гранулометрии в реальном времени")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("Настройки")
        
        # Получаем список участков
        sites = get_sites()
        
        if not sites:
            st.warning("Не удалось получить список участков. Проверьте подключение к API.")
            site_options = []
        else:
            site_options = [site['site_id'] for site in sites]
        
        # Выбор участка
        selected_site = st.selectbox(
            "Выберите участок:",
            options=site_options,
            format_func=lambda x: next((site['name'] for site in sites if site['site_id'] == x), x)
        )
        
        # Настройки обновления
        auto_refresh = st.checkbox("Автоматическое обновление", value=True)
        refresh_interval = st.slider(
            "Интервал обновления (сек):", 
            min_value=10, 
            max_value=300, 
            value=UPDATE_INTERVAL,
            step=10,
            disabled=not auto_refresh
        )
        
        # Кнопка ручного обновления
        if st.button("Обновить данные"):
            st.experimental_rerun()
        
        # Информация о системе
        st.divider()
        st.subheader("Информация о системе")
        
        try:
            response = requests.get(f"{API_URL}/statistics")
            if response.status_code == 200:
                stats = response.json()
                st.metric("Обработано запросов", stats.get('requests_processed', 0))
                st.metric("Время работы", f"{stats.get('uptime_seconds', 0) // 3600} ч {(stats.get('uptime_seconds', 0) % 3600) // 60} мин")
            else:
                st.warning("Не удалось получить статистику API")
        except Exception as e:
            st.warning(f"Ошибка при подключении к API: {str(e)}")
    
    # Автоматическое обновление
    if auto_refresh:
        # Используем st.empty() для обновления без перезагрузки страницы
        placeholder = st.empty()
        
        while True:
            with placeholder.container():
                display_dashboard(selected_site, sites)
            
            time.sleep(refresh_interval)
    else:
        display_dashboard(selected_site, sites)

def display_dashboard(selected_site, sites):
    """Отображает панель мониторинга для выбранного участка"""
    if not selected_site:
        st.info("Выберите участок в боковой панели для просмотра данных.")
        return
    
    # Получаем детальную информацию об участке
    site_details = get_site_details(selected_site)
    
    if not site_details:
        st.error(f"Не удалось получить информацию об участке {selected_site}")
        return
    
    # Отображаем заголовок с информацией об участке
    site_name = site_details.get('name', selected_site)
    site_status = site_details.get('status', 'Неизвестно')
    last_update = site_details.get('last_update', 'Нет данных')
    
    # Преобразуем временную метку в читаемый формат
    if last_update != 'Нет данных':
        try:
            last_update_dt = datetime.datetime.fromisoformat(last_update)
            last_update = last_update_dt.strftime("%d.%m.%Y %H:%M:%S")
        except:
            pass
    
    st.header(f"Мониторинг участка: {site_name}")
    
    # Отображаем информацию об участке в колонках
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Статус", site_status)
    
    with col2:
        st.metric("Последнее обновление", last_update)
    
    with col3:
        target_range = site_details.get('target_range', [COLOR_THRESHOLD_LOW, COLOR_THRESHOLD_HIGH])
        st.metric("Целевой диапазон", f"{target_range[0]}-{target_range[1]}%")
    
    # Получаем текущее предсказание
    current_prediction = None
    
    # Используем кэшированные данные, если они есть
    if selected_site in site_data_cache:
        current_prediction = site_data_cache[selected_site]
    else:
        # Иначе получаем предсказание от API
        # Тут должны быть реальные признаки, но для демонстрации используем пустой словарь
        # В реальной системе здесь будут данные с датчиков или из SCADA
        dummy_features = {"feature1": 0.5, "feature2": 0.8, "feature3": 0.2}
        prediction_result = send_prediction_request(selected_site, dummy_features)
        
        if prediction_result and 'prediction' in prediction_result:
            current_prediction = prediction_result
            site_data_cache[selected_site] = current_prediction
    
    # Если есть предсказание, добавляем его в историю
    if current_prediction:
        # Создаем запись для истории
        history_entry = {
            'timestamp': datetime.datetime.now(),
            'prediction': current_prediction['prediction'],
            'lower_bound': current_prediction.get('lower_bound', current_prediction['prediction'] * 0.95),
            'upper_bound': current_prediction.get('upper_bound', current_prediction['prediction'] * 1.05)
        }
        
        # Инициализируем историю для участка, если её нет
        if selected_site not in prediction_history:
            prediction_history[selected_site] = []
        
        # Добавляем запись в историю и ограничиваем длину
        prediction_history[selected_site].append(history_entry)
        if len(prediction_history[selected_site]) > HISTORY_LENGTH:
            prediction_history[selected_site] = prediction_history[selected_site][-HISTORY_LENGTH:]
    
    # Отображаем информацию о текущем состоянии
    if current_prediction:
        # Создаем таблицу с предсказанием и дополнительной информацией
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Отображаем датчик гранулометрии
            st.subheader("Текущая гранулометрия")
            gauge_chart = create_gauge_chart(
                current_prediction['prediction'],
                min_val=0,
                max_val=100,
                title="Гранулометрия (%)"
            )
            st.plotly_chart(gauge_chart, use_container_width=True)
            
            # Отображаем дополнительную информацию
            st.subheader("Детали предсказания")
            
            details_df = pd.DataFrame({
                "Метрика": [
                    "Предсказание",
                    "Нижняя граница (95%)",
                    "Верхняя граница (95%)",
                    "Уверенность модели"
                ],
                "Значение": [
                    f"{current_prediction['prediction']:.2f}%",
                    f"{current_prediction.get('lower_bound', current_prediction['prediction'] * 0.95):.2f}%",
                    f"{current_prediction.get('upper_bound', current_prediction['prediction'] * 1.05):.2f}%",
                    f"{current_prediction.get('confidence', 0.9) * 100:.1f}%"
                ]
            })
            
            st.table(details_df)
        
        with col2:
            # График истории гранулометрии
            st.subheader("История гранулометрии")
            history_chart = create_history_chart(prediction_history.get(selected_site, []), site_name)
            if history_chart:
                st.plotly_chart(history_chart, use_container_width=True)
            else:
                st.info("Недостаточно данных для построения графика истории.")
    else:
        st.warning("Нет данных о текущей гранулометрии для выбранного участка.")
    
    # Отображаем важность признаков
    st.subheader("Важность признаков")
    importance_data = get_feature_importance(selected_site)
    
    if importance_data:
        importance_chart = create_importance_chart(importance_data)
        if importance_chart:
            st.plotly_chart(importance_chart, use_container_width=True)
        else:
            st.info("Не удалось получить данные о важности признаков.")
    else:
        st.info("Нет данных о важности признаков для выбранного участка.")
    
    # Отображаем рекомендации и аналитику
    st.subheader("Рекомендации и аналитика")
    
    if current_prediction:
        prediction_value = current_prediction['prediction']
        target_min, target_max = site_details.get('target_range', [COLOR_THRESHOLD_LOW, COLOR_THRESHOLD_HIGH])
        
        if prediction_value < target_min:
            st.error(f"⚠️ Гранулометрия ниже целевого диапазона ({target_min}%). Рекомендуется проверить настройки оборудования.")
            
            # Дополнительные рекомендации в зависимости от степени отклонения
            if prediction_value < target_min * 0.9:
                st.error("❗ Критически низкое значение! Необходимо срочное вмешательство.")
                st.markdown("""
                **Возможные причины:**
                - Неправильная настройка оборудования
                - Изменение состава входного сырья
                - Износ оборудования
                
                **Рекомендуемые действия:**
                1. Проверить настройки измельчителей
                2. Проверить состав входного сырья
                3. Провести техническое обслуживание оборудования
                """)
        elif prediction_value > target_max:
            st.warning(f"⚠️ Гранулометрия выше целевого диапазона ({target_max}%). Возможно, требуется корректировка параметров.")
            
            # Дополнительные рекомендации в зависимости от степени отклонения
            if prediction_value > target_max * 1.1:
                st.warning("❗ Значительное превышение! Рекомендуется корректировка.")
                st.markdown("""
                **Возможные причины:**
                - Слишком высокая скорость измельчения
                - Неравномерное распределение сырья
                - Необходимость калибровки измерительного оборудования
                
                **Рекомендуемые действия:**
                1. Уменьшить скорость измельчения
                2. Проверить систему подачи сырья
                3. Калибровка измерительных приборов
                """)
        else:
            st.success(f"✅ Гранулометрия в пределах целевого диапазона ({target_min}-{target_max}%).")
            
            # Дополнительная информация при нормальных значениях
            if prediction_history.get(selected_site) and len(prediction_history[selected_site]) > 5:
                recent_values = [entry['prediction'] for entry in prediction_history[selected_site][-5:]]
                mean_recent = sum(recent_values) / len(recent_values)
                
                if abs(prediction_value - mean_recent) > 5:
                    st.info(f"ℹ️ Наблюдается изменение тренда (среднее за последние 5 измерений: {mean_recent:.2f}%).")
    
    # Отображаем время последнего обновления информации на дашборде
    st.divider()
    st.caption(f"Последнее обновление дашборда: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

if __name__ == "__main__":
    main() 