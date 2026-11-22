# Инструменты проекта

Справочник по библиотекам и методам, которые использованы в ноутбуках: что вызвано,
с какими параметрами и что это дало. Версии — из `requirements.txt`.

---

## 1. Библиотеки

| Библиотека | Версия | Где применена | Что делает в проекте |
|---|---|---|---|
| `pandas` | 2.3.3 | все стадии | Чтение CSV и parquet, таблицы, группировки, сводки |
| `numpy` | 2.3.5 | все стадии | Массивы, логарифмы, процентили, сэмплирование без внешних библиотек |
| `scikit-learn` | 1.9.1 | стадии 1, 3–8 | Модели, пайплайны, метрики, схемы проверки, поиск гиперпараметров |
| `scipy` | 1.16.3 | стадии 1, 6 | Статистические тесты и распределения для случайного поиска |
| `matplotlib` | 3.10.0 | стадии 1, 7 | Базовые графики и настройка осей |
| `seaborn` | 0.13.2 | стадии 1, 7 | Гистограммы с KDE, box и violin, тепловые карты |
| `phik` | 0.12.5 | стадия 1 | Матрица связей φk, работающая с числами и категориями |
| `lifelines` | 0.30.3 | стадия 1, дашборд | Кривые выживаемости Kaplan — Meier |
| `ydata-profiling` | 4.18.4 | стадия 1 | Автоматический отчёт о качестве данных |
| `optuna` | 5.1.0 | стадия 5 | Байесовская оптимизация гиперпараметров |
| `pyarrow` | 26.0.1 | стадии 1–8 | Движок parquet: обмен данными между стадиями |
| `kagglehub` | 1.0.3 | стадия 1 | Первичная загрузка набора данных |
| `streamlit` | 1.67.0 | `cirrhosis_dashboard.py` | Интерактивный дашборд с результатами модели |

`imbalanced-learn` в окружении нет: сокращение и воспроизводство выборки в стадии 3
сделаны на `numpy`. Это осознанно — лишняя зависимость ради двух операций не нужна.

---

## 2. Методы по стадиям

### Стадия 1. Анализ данных

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `pd.read_csv` | `pd.read_csv(f'{local_dir}/train.csv', index_col=0)` | 7 905 строк × 19 столбцов |
| `DataFrame.phik_matrix` | `phik_matrix(interval_cols=num_cols, njobs=1)` | Связь каждого признака с `Status`; пара «отёки ↔ асцит» 0.635 |
| `mutual_info_classif` | `mutual_info_classif(X_enc, y_num, discrete_features=disc, random_state=RANDOM_STATE)` | Подтвердила бесполезность `Drug`: 0.0004 |
| `chi2_contingency` | `chi2_contingency(tab)` | Исходы в группах препарата и плацебо неразличимы: p = 0.053 |
| `kruskal` | `kruskal(*[X.loc[y == g, c].values for g in ['C', 'CL', 'D']])` | Сравнение распределений признака между тремя классами |
| `stats.shapiro`, `stats.normaltest` | `stats.shapiro(s_sample).pvalue`, `stats.normaltest(s).pvalue` | Почти все числовые признаки отличаются от нормального распределения |
| `stats.zscore` | `np.abs(stats.zscore(df[num_cols], nan_policy='omit')) > 3` | Второе правило поиска выбросов |
| `KaplanMeierFitter.fit` | `kmf.fit(durations=group['N_Days'], event_observed=(group['Status'] == 'D'), label=f'Stage={stage}')` | Кривые выживаемости по стадиям расходятся |
| `cross_val_score` с `LinearRegression` | `cross_val_score(LinearRegression(), X_nd, y_nd, cv=5, scoring='r2').mean()` | Период наблюдения объясняется клиникой на 0.225, со статусом — на 0.234 |
| `StratifiedKFold` + `cross_val_score` | `cv=folds, scoring='accuracy'` | Спорная пара: с асцитом 0.8000, без асцита 0.7997, без отёков 0.7958 |
| `ProfileReport` | `ProfileReport(df_clean, title=..., explorative=True, correlations={'phi_k': {'calculate': True}})` | HTML-отчёт по качеству данных |

### Стадия 2. Проверка применимости линейной регрессии

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `Series.corr` (Пирсон) | `X.corrwith(y).sort_values(key=abs, ascending=False)` | Максимум \|r\| = 0.534 у `log_Bilirubin` |
| `Series.corr(method='spearman')` | `X[f].corr(y, method='spearman')` | Разрыв с Пирсоном не больше 0.024 — связи почти линейны |
| `DataFrame.corr` | `X.corr()` и отбор верхнего треугольника | Сильнейшая пара `log_Bilirubin ↔ log_Copper` 0.538, пар выше 0.6 нет |

Модели на этой стадии не обучаются: проверяются только предпосылки метода.

### Стадия 3. Базовая модель и редкий класс

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `train_test_split` | `test_size=0.25, stratify=y, random_state=RANDOM_STATE` | Контрольная часть 1 977 строк, в ней 69 классов `CL` |
| `LogisticRegression` | `LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)` | Базовая модель: 0.7951 / 0.5322 / 0.5438, полнота `CL` 0.0145 |
| `class_weight='balanced'` | `LogisticRegression(..., class_weight='balanced')` | Полнота `CL` 0.3913 ценой лог-потери 0.7800 |
| Сокращение выборки на numpy | `rng.choice(np.where(y_arr == cls)[0], n_min, replace=False)` | 618 строк; полнота `CL` 0.4493, лог-потеря 0.8814 |
| Воспроизводство выборки | `rng.choice(np.where(y_arr == cls)[0], n_max, replace=True)` | 11 169 строк; полнота `CL` 0.3913 |
| `accuracy_score`, `balanced_accuracy_score`, `f1_score`, `log_loss`, `recall_score` | Функция `evaluate()` | Единый набор метрик для всех стадий |
| Порог по контрольной части | `predict_proba` и перебор порога 0.02–0.95 | Порог 0.07: полнота `CL` 0.3478, доля верных 0.7304 |

### Стадия 4. Полиномы и регуляризация

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `PolynomialFeatures` | `PolynomialFeatures(degree=2, include_bias=False)` | 15 признаков → 135 столбцов |
| `GridSearchCV` | `GridSearchCV(estimator, grid, scoring='neg_log_loss', cv=CV, n_jobs=-1)` | Подбор `C` по кросс-валидации, без участия контрольной части |
| `LogisticRegression(penalty='l2')` | `LogisticRegression(penalty='l2', max_iter=5000, ...)` | Лог-потеря 0.5363, обнулённых коэффициентов 0 |
| `LogisticRegression(penalty='l1', solver='saga')` | `LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=5000, tol=1e-3, ...)` | Лог-потеря 0.5296, обнулено 233 из 396 |
| `LogisticRegression(penalty='elasticnet', solver='saga')` | `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=..., ...)` | Лог-потеря 0.5248, обнулено 266 из 396 |
| `cross_val_predict` | `cross_val_predict(l1_best, X_tr_p, y_train, cv=CV, method='predict_proba')` | Порог `CL` 0.13 по out-of-fold вероятностям: F-мера 0.5954, полнота 0.2319 |

### Стадия 5. Сетка против байесовской оптимизации

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `GridSearchCV` | 80 комбинаций (`degree` × `l1_ratio` × `C`), 5 блоков | 400 обучений, 59.1 с, лог-потеря 0.4991 |
| `optuna.create_study` | `create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))` | Воспроизводимый байесовский поиск |
| `trial.suggest_categorical`, `trial.suggest_float` | `suggest_float('C', 1e-3, 10.0, log=True)` | Непрерывная шкала силы штрафа вместо восьми точек сетки |
| `study.optimize` | `study.optimize(objective, n_trials=25)` | 125 обучений, 29.9 с, лог-потеря 0.4989 |

### Стадия 6. Пайплайны и схемы проверки

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `Pipeline` | `Pipeline([('scaler', ...), ('poly', ...), ('model', ...)])` | Предобработка обучается внутри блока, утечки нет |
| `KFold`, `StratifiedKFold` | `n_splits=5, shuffle=True, random_state=RANDOM_STATE` | Разброс доли `CL` по блокам 0.00474 против 0.00033 |
| `RepeatedStratifiedKFold` | `n_splits=5, n_repeats=5, random_state=RANDOM_STATE` | Ошибка среднего 0.0025 против 0.0077 — уточнение в 3.12 раза |
| `LeaveOneOut` | `LeaveOneOut()` на 200 строках | 1.61 с против 0.04 с, качество то же; на полном объёме около 46 мин |
| `GridSearchCV` | `scoring='neg_log_loss', cv=CV, n_jobs=-1` | 400 обучений, 46.1 с, лог-потеря 0.4975 |
| `RandomizedSearchCV` | `n_iter=25, param_distributions={'model__C': loguniform(1e-3, 10.0), 'model__l1_ratio': uniform(0, 1)}` | 125 обучений, 14.3 с, лог-потеря 0.5022 |
| `HalvingGridSearchCV` | `factor=2, min_resources=600, random_state=RANDOM_STATE` | 750 обучений, 23.0 с, лог-потеря 0.4959 — лучшая по CV |
| `HalvingRandomSearchCV` | `n_candidates='exhaust', factor=2, min_resources=600` | 95 обучений, 5.9 с, лог-потеря 0.5047 |

### Стадия 7. Клинические признаки

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `ColumnTransformer` | `ColumnTransformer([('bin', 'passthrough', binaries), ('num', scaler, numerics)])` | Бинарные столбцы не масштабируются |
| `PolynomialFeatures` | `PolynomialFeatures(degree=2, include_bias=False)` | 22 признака → 275 столбцов |
| `RobustScaler`, `StandardScaler`, `MinMaxScaler`, `passthrough` | Четыре варианта в одном цикле | Лог-потеря 0.4968 / 0.4969 / 0.5180 / 0.5868 |
| `LogisticRegression(solver='saga', l1_ratio=1.0, C=0.1)` | `max_iter=5000, tol=1e-3, random_state=RANDOM_STATE` | Итоговая модель стадии |
| `SplineTransformer` | `SplineTransformer(n_knots=4, degree=2, include_bias=False)` | 34 столбца, лог-потеря 0.5069 — хуже полинома на 0.0101 |
| `cross_val_score` | `scoring='neg_log_loss', cv=StratifiedKFold(5)` | Среднее и разброс по блокам для всех сравнений |
| `cross_val_predict` | `method='predict_proba', n_jobs=-1` | Порог `CL` 0.11 по out-of-fold вероятностям |

### Стадия 8. Сводка

| Метод | Вызов в проекте | Что дал |
|---|---|---|
| `Pipeline` + `ColumnTransformer` | Повторяется построение итоговой модели стадии 7 | `assert` подтвердил совпадение: 0.8037 / 0.5381 / 0.5316 / 0.5180 |

---
