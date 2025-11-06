# TestMethods - Директория для Тестирования Алгоритмов Кластеризации

## Содержание

- [Назначение](#назначение)
- [Структура Директории](#структура-директории)
- [Быстрый Старт](#быстрый-старт)
- [Создание Нового Теста](#создание-нового-теста)
- [Примеры Использования](#примеры-использования)
- [Рекомендации](#рекомендации)
- [FAQ](#faq)

---

## Назначение

Директория `TestMethods/` предназначена для **тестирования и валидации** алгоритмов кластеризации перед их интеграцией в основной GUI фреймворк.

### Основные Цели

 - **Проверка корректности** работы алгоритмов  
 - **Визуализация результатов** кластеризации  
 - **Сравнение параметров** и их влияния на результаты  
 - **Генерация документации** с примерами работы  
 - **Отладка** новых методов кластеризации  

---

## Структура Директории

```
DMM_ClusteringSystem/
├── TestMethods/                      # Корневая директория для тестов
│   ├── README.md                     # Этот файл - документация
│   ├── test_wave.py                  # Пример: тест WaveClustering
│   ├── test_dbscan.py                # Пример: тест DBSCAN (опционально)
│   ├── test_template.py              # Шаблон для новых тестов
│   │
│   └── Images/                       # Директория для результатов
│       ├── WaveClustering/           # Результаты WaveClustering
│       │   ├── test_wave_2d.png
│       │   ├── test_wave_3d.png
│       │   └── test_wave_strategy.png
│       │
│       ├── DBSCAN/                   # Результаты DBSCAN
│       │   ├── test_dbscan_eps05.png
│       │   └── test_dbscan_eps10.png
│       │
│       └── SpectralClustering/       # Результаты Spectral Clustering
│           └── ...
```

### Описание Компонентов

| Файл/Папка | Назначение |
|------------|------------|
| `test_*.py` | Скрипты для тестирования конкретных алгоритмов |
| `test_template.py` | Шаблон для создания новых тестов |
| `Images/` | Хранение визуализаций результатов |
| `Images/{AlgorithmName}/` | Отдельная папка для каждого алгоритма |

---

## Быстрый Старт

### Шаг 1: Запуск Существующего Теста

```bash
# Перейдите в корневую директорию проекта
cd D:\pycharm\clustering\faze4\DMM_ClusteringSystem

# Запустите тест WaveClustering
python TestMethods/test_wave.py
```

### Шаг 2: Просмотр Результатов

После выполнения теста:

1. **Консоль** покажет статистику:
   ```
   ================================================================================
   ТЕСТ 1: 2D Кластеризация
   ================================================================================
   Количество точек: 300
   Найдено кластеров: 3
   ✅ 2D визуализация сохранена: TestMethods/Images/WaveClustering/test_wave_2d.png
   ```

2. **Графики** автоматически откроются в matplotlib

3. **Файлы PNG** сохранятся в `TestMethods/Images/WaveClustering/`

### Шаг 3: Анализ Результатов

Откройте созданные изображения:
- `test_wave_2d.png` - 2D кластеризация
- `test_wave_3d.png` - 3D кластеризация с проекциями
- `test_wave_strategy.png` - тест интеграции с фреймворком

---

## Создание Нового Теста

### Вариант 1: Использование Шаблона

#### Шаг 1: Создайте Файл Теста

Создайте новый файл в `TestMethods/`:
```
TestMethods/test_your_algorithm.py
```

#### Шаг 2: Скопируйте Шаблон

```python
"""
Тест для алгоритма YourAlgorithm
Автор: Ваше Имя
Дата: гггг-мм-дд
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Импорт вашего алгоритма
from ClusteringMethods.YourAlgorithmFile import (
    YourAlgorithmClass,
    ConcreteStrategyYourAlgorithm
)


# ============================================================================
# Настройка Путей для Сохранения
# ============================================================================

# Определяем пути
TEST_DIR = Path(__file__).parent
IMAGES_DIR = TEST_DIR / "Images" / "YourAlgorithm"

# Создаем директорию, если не существует
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Результаты будут сохранены в: {IMAGES_DIR}")


# ============================================================================
# Тестовые Функции
# ============================================================================

def generate_test_data(n_samples=300, n_clusters=3, noise=0.1):
    """
    Генерация тестовых данных для кластеризации
    
    Parameters:
    -----------
    n_samples : int
        Общее количество точек
    n_clusters : int
        Количество кластеров
    noise : float
        Уровень шума
    """
    from sklearn.datasets import make_blobs
    
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_clusters,
        cluster_std=noise,
        random_state=42
    )
    
    return X, y_true


def test_basic_clustering():
    """
    Базовый тест кластеризации
    """
    print("="*80)
    print("ТЕСТ 1: Базовая Кластеризация")
    print("="*80)
    
    # Генерация данных
    X, y_true = generate_test_data(n_samples=300, n_clusters=3)
    
    # Создание и применение модели
    model = YourAlgorithmClass(
        parameter1=value1,
        parameter2=value2
    )
    
    y_pred = model.fit_predict(X)
    
    # Статистика
    print(f"Количество точек: {len(X)}")
    print(f"Найдено кластеров: {len(np.unique(y_pred))}")
    print(f"Истинное количество: {len(np.unique(y_true))}")
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Истинные кластеры
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
                   alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[0].set_title('Истинные Кластеры', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X₁')
    axes[0].set_ylabel('X₂')
    axes[0].grid(True, alpha=0.3)
    
    # Предсказанные кластеры
    axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                   alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[1].set_title('YourAlgorithm Результаты', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X₁')
    axes[1].set_ylabel('X₂')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение
    output_file = IMAGES_DIR / "test_basic.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Визуализация сохранена: {output_file}")
    
    return X, y_pred


def test_parameter_sensitivity():
    """
    Тест чувствительности к параметрам
    """
    print("\n" + "="*80)
    print("ТЕСТ 2: Чувствительность к Параметрам")
    print("="*80)
    
    # Генерация данных
    X, y_true = generate_test_data()
    
    # Тестируем разные значения параметра
    param_values = [value1, value2, value3, value4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, param_val in enumerate(param_values):
        model = YourAlgorithmClass(parameter=param_val)
        y_pred = model.fit_predict(X)
        
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                       alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'Parameter = {param_val}\\n({len(np.unique(y_pred))} кластеров)',
                         fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        print(f"  Parameter={param_val}: {len(np.unique(y_pred))} кластеров")
    
    plt.tight_layout()
    
    # Сохранение
    output_file = IMAGES_DIR / "test_parameters.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Визуализация сохранена: {output_file}")


# ============================================================================
# Главная Функция
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ: YourAlgorithm")
    print("="*80 + "\n")
    
    try:
        # Запуск тестов
        test_basic_clustering()
        test_parameter_sensitivity()
        
        print("\n" + "="*80)
        print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО! ✅")
        print("="*80)
        print(f"\n📊 Результаты сохранены в: {IMAGES_DIR}")
        print("="*80 + "\n")
        
        # Показать графики
        plt.show()
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
```

#### Шаг 3: Настройте Тест

Замените:
- `YourAlgorithm` → имя вашего алгоритма
- `parameter1`, `parameter2` → реальные параметры
- Добавьте специфичные тесты для вашего алгоритма

#### Шаг 4: Запустите Тест

```bash
python TestMethods/test_your_algorithm.py
```

---

### Вариант 2: Расширенный Тест (как test_wave.py)

Для более сложного тестирования с множественными визуализациями:

```python
"""
Расширенный тест для YourAlgorithm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ... импорты ...

# Настройка путей
TEST_DIR = Path(__file__).parent
IMAGES_DIR = TEST_DIR / "Images" / "YourAlgorithm"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def test_2d_clustering():
    """Тест 2D кластеризации"""
    # ... код теста ...
    
    # Сохранение
    output = IMAGES_DIR / "test_2d.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✅ 2D визуализация: {output}")


def test_3d_clustering():
    """Тест 3D кластеризации"""
    # ... код теста ...
    
    output = IMAGES_DIR / "test_3d.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✅ 3D визуализация: {output}")


def test_different_datasets():
    """Тест на разных наборах данных"""
    from sklearn.datasets import make_moons, make_circles, make_blobs
    
    datasets = [
        ("Blobs", make_blobs(n_samples=300, centers=3, random_state=42)),
        ("Moons", make_moons(n_samples=300, noise=0.05, random_state=42)),
        ("Circles", make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42))
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, (X, _)) in enumerate(datasets):
        model = YourAlgorithmClass(params)
        y_pred = model.fit_predict(X)
        
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                       alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'{name}\\n({len(np.unique(y_pred))} кластеров)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output = IMAGES_DIR / "test_datasets.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✅ Сравнение датасетов: {output}")


def test_strategy_integration():
    """Тест интеграции с фреймворком"""
    from ClusteringMethods.ClasteringAlgorithms import Context, StrategiesManager
    
    # ... код теста через Strategy Pattern ...
    
    output = IMAGES_DIR / "test_strategy.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✅ Strategy Pattern: {output}")


if __name__ == "__main__":
    print("="*80)
    print("РАСШИРЕННОЕ ТЕСТИРОВАНИЕ: YourAlgorithm")
    print("="*80 + "\n")
    
    test_2d_clustering()
    test_3d_clustering()
    test_different_datasets()
    test_strategy_integration()
    
    print("\n" + "="*80)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ! ✅")
    print(f"📊 Результаты: {IMAGES_DIR}")
    print("="*80)
    
    plt.show()
```

---

## Примеры Использования

### Пример 1: Быстрый Тест Нового Алгоритма

```bash
# 1. Создайте файл теста
cd TestMethods
cp test_template.py test_myalgorithm.py

# 2. Отредактируйте test_myalgorithm.py
# - Замените имена классов
# - Настройте параметры

# 3. Запустите
python test_myalgorithm.py

# 4. Проверьте результаты
ls Images/MyAlgorithm/
```

### Пример 2: Сравнение Параметров

```python
# В вашем test_*.py

def compare_parameters():
    """Сравнение разных значений параметра"""
    
    X, y_true = generate_data()
    
    param_values = [0.1, 0.5, 1.0, 2.0]
    results = []
    
    for param in param_values:
        model = YourAlgorithm(param=param)
        y_pred = model.fit_predict(X)
        results.append((param, y_pred, len(np.unique(y_pred))))
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, (param, labels, n_clusters) in enumerate(results):
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
        axes[i].set_title(f'Param={param}, Clusters={n_clusters}')
    
    plt.savefig(IMAGES_DIR / "comparison.png")
```

### Пример 3: Тест на Реальных Данных

```python
def test_real_data():
    """Тест на реальном датасете"""
    
    # Загрузка данных
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data[:, :2]  # Только первые 2 признака для визуализации
    
    # Кластеризация
    model = YourAlgorithm()
    labels = model.fit_predict(X)
    
    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('Iris Dataset Clustering')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    
    plt.savefig(IMAGES_DIR / "iris_test.png")
```

---

## Рекомендации

### Хорошие Практики

1. **Именование файлов:**
   - `test_{algorithm_name}.py` - для скриптов
   - `test_{algorithm}_{test_type}.png` - для изображений
   - Пример: `test_wave_2d.png`, `test_wave_parameters.png`

2. **Структура теста:**
   ```python
   # 1. Импорты
   # 2. Настройка путей
   # 3. Функции генерации данных
   # 4. Тестовые функции
   # 5. Главная функция main
   ```

3. **Визуализация:**
   - Используйте высокое разрешение (dpi=150 или 300)
   - Добавляйте заголовки и подписи осей
   - Используйте сетку для удобства чтения
   - Сохраняйте с `bbox_inches='tight'`

4. **Документация:**
   - Добавляйте docstring к каждой функции
   - Комментируйте сложные участки кода
   - Указывайте автора и дату в заголовке

5. **Обработка ошибок:**
   ```python
   try:
       test_function()
   except Exception as e:
       print(f"❌ Ошибка: {e}")
       import traceback
       traceback.print_exc()
   ```

### Чего Избегать

1.  Жестко зашитые пути (используйте `Path(__file__).parent`)
2.  Сохранение в корень проекта (используйте `Images/` директорию)
3.  Отсутствие обработки ошибок
4.  Визуализации без подписей и заголовков
5.  Тесты без вывода статистики

---

## Шаблон Визуализации

### Базовый График

```python
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10',
           alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
plt.title('Algorithm Name Results', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
```

### Сравнение (Multiple Subplots)

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for i, (title, data, labels) in enumerate(test_cases):
    axes[i].scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
```

### 3D Визуализация

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, 
                    cmap='tab10', alpha=0.6, s=30)
ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
ax.set_zlabel('X₃')
ax.set_title('3D Clustering Results', fontsize=14, fontweight='bold')

plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
```

---

## FAQ

### Где должен находиться файл теста?

**Ответ:** В директории `TestMethods/`:
```
TestMethods/test_your_algorithm.py
```

### Куда сохраняются результаты?

**Ответ:** В поддиректорию `Images/`:
```
TestMethods/Images/{AlgorithmName}/
```

### Как автоматически создать директорию для результатов?

**Ответ:** Используйте `Path` и `mkdir`:
```python
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "Images" / "YourAlgorithm"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
```

### Как добавить метаданные к изображениям?

**Ответ:** Используйте параметр `metadata`:
```python
metadata = {
    'Title': 'Test Results',
    'Author': 'Your Name',
    'Algorithm': 'YourAlgorithm',
    'Date': 'гггг-мм-дд'
}

plt.savefig(output, dpi=150, bbox_inches='tight', metadata=metadata)
```

### Можно ли запускать несколько тестов одновременно?

**Ответ:** Да, каждый тест независим и сохраняет результаты в свою директорию.

### Как сравнить разные алгоритмы?

**Ответ:** Создайте отдельный тест `test_comparison.py`:
```python
from ClusteringMethods.WaveClusteringAlgorithm import WaveClustering
from ClusteringMethods.ClasteringAlgorithms import ConcreteStrategyDBSCANfromSKLEARN

# Сравнение
algorithms = [
    ("WaveClustering", WaveClustering()),
    ("DBSCAN", DBSCAN(eps=0.5))
]

for name, model in algorithms:
    labels = model.fit_predict(X)
    # Визуализация и сохранение
```

---

## Поддержка

Если у вас возникли вопросы:
1. Проверьте примеры в `test_wave.py`
2. Используйте шаблон `test_template.py`
3. Обратитесь к документации алгоритма

---

**Авторы:** Левинский Григорий Дмитриевич [levinskiy@mirea.ru]  
**Последнее обновление:** 2025-10-21

---
