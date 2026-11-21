"""Интерактивный дашборд по данным о пациентах с циррозом печени

Дашборд построен на данных Multi-Class Prediction of Cirrhosis Outcomes
(публикация Kaggle, идентификатор набора `playground-series-s3e26`;
первоисточник — UCI Machine Learning Repository).

Запуск:
    1. Убедитесь, что зависимости установлены:
       pip install streamlit lifelines pandas numpy matplotlib seaborn

    2. Запустите дашборд:
       streamlit run cirrhosis_dashboard.py

    3. Откройте в браузере:
       http://localhost:8501

Особенности запуска:
- Дашборд использует файл `data/train.csv` из текущей директории.
  Если вы запускаете его не из папки проекта, укажите путь к датасету
  в переменной `DATA_PATH` в начале скрипта (по умолчанию `data/train.csv`).
- Для работы Kaplan-Meier графиков требуется установленная библиотека `lifelines`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from lifelines import KaplanMeierFitter

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================

# Путь к файлу с данными (относительно текущей директории запуска)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'train.csv')

st.set_page_config(
    page_title='Анализ данных: цирроз печени',
    layout='wide',
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Загрузка и базовая подготовка данных."""
    if not os.path.exists(path):
        st.error(f"Файл не найден: {path}")
        st.stop()

    df = pd.read_csv(path, index_col=0)
    df['Age_years'] = (df['Age'] / 365.25).round(1)
    df['Edema_num'] = df['Edema'].map({'N': 0.0, 'S': 0.5, 'Y': 1.0})
    return df


# ==============================================================================
# UI
# ==============================================================================

st.title('Анализ данных о пациентах с циррозом печени')
st.markdown(
    '**Исследовательский анализ данных Multi-Class Prediction of Cirrhosis Outcomes**\n\n'
    'Фильтруйте данные по стадиям, исходу и возрасту, чтобы исследовать взаимосвязи.\n\n'
    '> **Совет:** Используйте вкладки для переключения между типами графиков.'
)

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

# --- Боковая панель (фильтры) ---
st.sidebar.header('Фильтры')

selected_stages = st.sidebar.multiselect(
    'Стадия (Stage)',
    options=sorted(df['Stage'].unique()),
    default=sorted(df['Stage'].unique()),
)

selected_status = st.sidebar.multiselect(
    'Исход (Status)',
    options=sorted(df['Status'].unique()),
    default=sorted(df['Status'].unique()),
)

age_range = st.sidebar.slider(
    'Возраст (лет)',
    min_value=int(df['Age_years'].min()),
    max_value=int(df['Age_years'].max()),
    value=(int(df['Age_years'].min()), int(df['Age_years'].max())),
)

# --- Основной контент ---
df_filtered = df[
    df['Stage'].isin(selected_stages)
    & df['Status'].isin(selected_status)
    & df['Age_years'].between(*age_range)
]

# Метрики
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Всего пациентов', len(df_filtered))
with col2:
    st.metric('Медиана N_Days', int(df_filtered['N_Days'].median()))
with col3:
    st.metric('Медиана Bilirubin', round(df_filtered['Bilirubin'].median(), 2))

st.divider()

# --- Вкладки ---
tab1, tab2, tab3, tab4 = st.tabs(
    ['Распределения', 'Выживаемость', 'Корреляции', 'Данные']
)

with tab1:
    st.subheader('Распределения признаков')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_filtered['N_Days'], bins=40, kde=True, ax=axes[0])
    axes[0].set_title('Распределение N_Days')
    sns.histplot(df_filtered['Age_years'], bins=30, kde=True, ax=axes[1])
    axes[1].set_title('Распределение возраста (лет)')
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader('Выживаемость по Stage (Kaplan-Meier)')
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for stage, group in df_filtered.groupby('Stage'):
        event = (group['Status'] == 'D').astype(int)
        kmf.fit(durations=group['N_Days'], event_observed=event,
            label=f'Stage={stage}')
        kmf.plot_survival_function(ax=ax)
    ax.set_title('Kaplan-Meier: выживаемость по Stage',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('N_Days', fontsize=12)
    ax.set_ylabel('S(t)', fontsize=12)
    st.pyplot(fig)

with tab3:
    st.subheader('Корреляции Пирсона')
    numeric_cols = ['N_Days', 'Age_years', 'Bilirubin', 'Albumin',
                    'Copper', 'Platelets', 'Prothrombin']
    numeric_cols = [c for c in numeric_cols if c in df_filtered.columns]
    corr = df_filtered[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Матрица корреляций Пирсона')
    st.pyplot(fig)

with tab4:
    st.subheader('Данные')
    st.dataframe(df_filtered.head(100))
