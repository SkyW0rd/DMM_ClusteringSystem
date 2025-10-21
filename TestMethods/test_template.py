"""
Шаблон для создания тестов алгоритмов кластеризации
Автор: Григорий Дмитриевич [glevinskiy@gmail.com]
Последнее обновление: 2025-10-21

ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:
1. Скопируйте этот файл: test_template.py -> test_your_algorithm.py
2. Замените все "YourAlgorithm" на название вашего алгоритма
3. Настройте параметры в разделе "НАСТРОЙКИ"
4. Реализуйте тестовые функции
5. Запустите: python TestMethods/test_your_algorithm.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.datasets import make_blobs, make_moons, make_circles

# Добавляем корневую директорию проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# НАСТРОЙКИ - ИЗМЕНИТЕ ЭТО
# ============================================================================

# Имя вашего алгоритма (для папки с изображениями)
ALGORITHM_NAME = "YourAlgorithm"

# Импорт вашего алгоритма - ЗАМЕНИТЕ НА ВАШИ КЛАССЫ
try:
    from ClusteringMethods.YourAlgorithmFile import (
        YourAlgorithmClass,
        ConcreteStrategyYourAlgorithm
    )
    ALGORITHM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Предупреждение: {e}")
    print("   Убедитесь, что ваш алгоритм правильно импортирован")
    ALGORITHM_AVAILABLE = False

# Также импортируем базовые классы для теста интеграции
try:
    from ClusteringMethods.ClasteringAlgorithms import (
        Context,
        StrategiesManager
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False


# ============================================================================
# НАСТРОЙКА ПУТЕЙ
# ============================================================================

TEST_DIR = Path(__file__).parent
IMAGES_DIR = TEST_DIR / "Images" / ALGORITHM_NAME

# Создаем директорию для изображений
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"📁 Директория для результатов: {IMAGES_DIR}")
print("="*80 + "\n")


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def generate_test_data_2d(dataset_type='blobs', n_samples=300, **kwargs):
    """
    Генерация 2D тестовых данных
    
    Parameters:
    -----------
    dataset_type : str
        Тип датасета: 'blobs', 'moons', 'circles'
    n_samples : int
        Количество точек
    **kwargs : dict
        Дополнительные параметры для генератора
    
    Returns:
    --------
    X : ndarray, shape (n_samples, 2)
        Данные
    y_true : ndarray, shape (n_samples,)
        Истинные метки
    """
    np.random.seed(kwargs.get('random_state', 42))
    
    if dataset_type == 'blobs':
        X, y_true = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=kwargs.get('centers', 3),
            cluster_std=kwargs.get('cluster_std', 0.5),
            random_state=kwargs.get('random_state', 42)
        )
    elif dataset_type == 'moons':
        X, y_true = make_moons(
            n_samples=n_samples,
            noise=kwargs.get('noise', 0.05),
            random_state=kwargs.get('random_state', 42)
        )
    elif dataset_type == 'circles':
        X, y_true = make_circles(
            n_samples=n_samples,
            noise=kwargs.get('noise', 0.05),
            factor=kwargs.get('factor', 0.5),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X, y_true


def generate_test_data_3d(n_samples=300, n_clusters=3, cluster_std=0.5):
    """
    Генерация 3D тестовых данных
    
    Returns:
    --------
    X : ndarray, shape (n_samples, 3)
        3D данные
    y_true : ndarray, shape (n_samples,)
        Истинные метки
    """
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=3,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=42
    )
    return X, y_true


def save_figure(fig, filename, dpi=150):
    """
    Сохранение фигуры с метаданными
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Фигура для сохранения
    filename : str
        Имя файла (без пути)
    dpi : int
        Разрешение
    """
    output_path = IMAGES_DIR / filename
    
    metadata = {
        'Title': f'{ALGORITHM_NAME} Test Results',
        'Author': 'DMM Clustering System',
        'Algorithm': ALGORITHM_NAME,
    }
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', metadata=metadata)
    print(f"✅ Сохранено: {output_path.name}")


# ============================================================================
# ТЕСТОВЫЕ ФУНКЦИИ - РЕАЛИЗУЙТЕ СВОИ ТЕСТЫ
# ============================================================================

def test_basic_2d_clustering():
    """
    ТЕСТ 1: Базовая 2D кластеризация
    
    Описание:
    - Генерирует простой 2D датасет
    - Применяет алгоритм кластеризации
    - Визуализирует результаты
    - Сравнивает с истинными метками
    """
    print("="*80)
    print("ТЕСТ 1: Базовая 2D Кластеризация")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    # Генерация данных
    X, y_true = generate_test_data_2d('blobs', n_samples=300, centers=3, cluster_std=0.5)
    
    # ЗАМЕНИТЕ НА ВАШ АЛГОРИТМ И ПАРАМЕТРЫ
    model = YourAlgorithmClass(
        # parameter1=value1,
        # parameter2=value2,
    )
    
    # Кластеризация
    y_pred = model.fit_predict(X)
    
    # Статистика
    n_clusters_pred = len(np.unique(y_pred))
    n_clusters_true = len(np.unique(y_true))
    
    print(f"Количество точек: {len(X)}")
    print(f"Истинное количество кластеров: {n_clusters_true}")
    print(f"Найдено кластеров: {n_clusters_pred}")
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Истинные кластеры
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                              alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[0].set_title('Истинные Кластеры', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X₁', fontsize=12)
    axes[0].set_ylabel('X₂', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Истинная метка')
    
    # Subplot 2: Предсказанные кластеры
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                              alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[1].set_title(f'{ALGORITHM_NAME} Результаты\n({n_clusters_pred} кластеров)',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X₁', fontsize=12)
    axes[1].set_ylabel('X₂', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Предсказанная метка')
    
    plt.tight_layout()
    save_figure(fig, 'test_basic_2d.png')
    
    print()
    return X, y_pred


def test_parameter_sensitivity():
    """
    ТЕСТ 2: Чувствительность к параметрам
    
    Описание:
    - Тестирует алгоритм с разными значениями ключевого параметра
    - Визуализирует влияние параметра на результаты
    """
    print("="*80)
    print("ТЕСТ 2: Чувствительность к Параметрам")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    # Генерация данных
    X, y_true = generate_test_data_2d('blobs', n_samples=300, centers=3)
    
    # НАСТРОЙТЕ ПАРАМЕТРЫ ДЛЯ ТЕСТИРОВАНИЯ
    # Пример: тестируем разные значения eps для DBSCAN
    param_name = "parameter"  # ЗАМЕНИТЕ на название параметра
    param_values = [0.1, 0.5, 1.0, 2.0]  # ЗАМЕНИТЕ на значения
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    
    for i, param_value in enumerate(param_values):
        # ЗАМЕНИТЕ НА ВАШ АЛГОРИТМ
        model = YourAlgorithmClass(
            # **{param_name: param_value}
        )
        
        y_pred = model.fit_predict(X)
        n_clusters = len(np.unique(y_pred))
        
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                       alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'{param_name}={param_value}\n({n_clusters} кластеров)',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].grid(True, alpha=0.3)
        
        print(f"  {param_name}={param_value}: {n_clusters} кластеров")
    
    plt.tight_layout()
    save_figure(fig, 'test_parameters.png')
    
    print()


def test_different_datasets():
    """
    ТЕСТ 3: Разные типы датасетов
    
    Описание:
    - Тестирует алгоритм на датасетах разных форм
    - Проверяет способность находить нелинейные структуры
    """
    print("="*80)
    print("ТЕСТ 3: Разные Типы Датасетов")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    # Различные типы датасетов
    datasets = [
        ("Blobs (Сферические)", 'blobs', {'centers': 3, 'cluster_std': 0.5}),
        ("Moons (Полумесяцы)", 'moons', {'noise': 0.05}),
        ("Circles (Круги)", 'circles', {'noise': 0.05, 'factor': 0.5}),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, dataset_type, params) in enumerate(datasets):
        X, _ = generate_test_data_2d(dataset_type, n_samples=300, **params)
        
        # ЗАМЕНИТЕ НА ВАШ АЛГОРИТМ
        model = YourAlgorithmClass()
        y_pred = model.fit_predict(X)
        n_clusters = len(np.unique(y_pred))
        
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                       alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'{name}\n({n_clusters} кластеров)',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].grid(True, alpha=0.3)
        
        print(f"  {name}: {n_clusters} кластеров")
    
    plt.tight_layout()
    save_figure(fig, 'test_datasets.png')
    
    print()


def test_3d_clustering():
    """
    ТЕСТ 4: 3D кластеризация (опционально)
    
    Описание:
    - Генерирует 3D данные
    - Применяет кластеризацию
    - Визуализирует в 3D и 2D проекциях
    """
    print("="*80)
    print("ТЕСТ 4: 3D Кластеризация")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    # Генерация 3D данных
    X, y_true = generate_test_data_3d(n_samples=300, n_clusters=3)
    
    # Для 2D алгоритмов - используем проекции
    # Для 3D алгоритмов - используем напрямую
    
    # ЗАМЕНИТЕ В ЗАВИСИМОСТИ ОТ ВАШЕГО АЛГОРИТМА
    # Для 2D алгоритма:
    X_2d = X[:, :2]  # Берем только первые 2 измерения
    model = YourAlgorithmClass()
    y_pred = model.fit_predict(X_2d)
    
    # Для 3D алгоритма (если поддерживается):
    # model = YourAlgorithmClass()
    # y_pred = model.fit_predict(X)
    
    n_clusters = len(np.unique(y_pred))
    print(f"Найдено кластеров: {n_clusters}")
    
    # Визуализация
    fig = plt.figure(figsize=(16, 6))
    
    # 3D визуализация истинных кластеров
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true, cmap='viridis',
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax1.set_title('3D: Истинные Кластеры', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('X₃')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # 3D визуализация результатов
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap='tab10',
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax2.set_title(f'3D: {ALGORITHM_NAME}\n({n_clusters} кластеров)',
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_zlabel('X₃')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # 2D проекция
    ax3 = fig.add_subplot(1, 3, 3)
    scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax3.set_title('2D Проекция (XY)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X₁')
    ax3.set_ylabel('X₂')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    save_figure(fig, 'test_3d.png')
    
    print()


def test_strategy_integration():
    """
    ТЕСТ 5: Интеграция с фреймворком (Strategy Pattern)
    
    Описание:
    - Проверяет работу через паттерн Strategy
    - Тестирует интеграцию с основной системой
    """
    print("="*80)
    print("ТЕСТ 5: Интеграция с Фреймворком (Strategy Pattern)")
    print("="*80)
    
    if not STRATEGY_AVAILABLE:
        print("❌ Strategy Pattern не доступен. Пропускаем тест.")
        return
    
    # Генерация данных
    X, y_true = generate_test_data_2d('blobs', n_samples=200, centers=2)
    
    try:
        # Получение конфигурации
        # ЗАМЕНИТЕ "youralgorithm" на ID вашего алгоритма
        config = StrategiesManager.getStrategyRunConfigById("youralgorithm")
        
        # Настройка параметров
        # ЗАМЕНИТЕ на ваши параметры
        # config["parameter1"] = value1
        # config["parameter2"] = value2
        
        # Создание контекста
        strategy = ConcreteStrategyYourAlgorithm()
        context = Context(strategy)
        
        # Кластеризация (ВАЖНО: transpose для совместимости с фреймворком)
        y_pred = context.do_some_clustering_points(X.T, config)
        
        n_clusters = len(np.unique(y_pred))
        print(f"Найдено кластеров: {n_clusters}")
        print("✅ Интеграция с фреймворком работает!")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                           alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        ax.set_title(f'{ALGORITHM_NAME} через Strategy Pattern\n({n_clusters} кластеров)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X₁', fontsize=12)
        ax.set_ylabel('X₂', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Кластер')
        
        plt.tight_layout()
        save_figure(fig, 'test_strategy.png')
        
    except Exception as e:
        print(f"❌ Ошибка интеграции: {e}")
        import traceback
        traceback.print_exc()
    
    print()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция - запускает все тесты
    """
    print("\n" + "="*80)
    print(f"ТЕСТИРОВАНИЕ АЛГОРИТМА: {ALGORITHM_NAME}")
    print("="*80 + "\n")
    
    if not ALGORITHM_AVAILABLE:
        print("❌ ОШИБКА: Алгоритм не найден!")
        print("   Проверьте импорты в начале файла")
        print("   Убедитесь, что ваш алгоритм правильно установлен")
        return
    
    try:
        # Запуск всех тестов
        test_basic_2d_clustering()
        test_parameter_sensitivity()
        test_different_datasets()
        test_3d_clustering()
        test_strategy_integration()
        
        print("="*80)
        print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО! ✅")
        print("="*80)
        print(f"\n📊 Результаты сохранены в: {IMAGES_DIR}")
        print(f"📁 Всего файлов: {len(list(IMAGES_DIR.glob('*.png')))}")
        print("\nОткройте изображения для просмотра результатов!")
        print("="*80 + "\n")
        
        # Показать все графики
        plt.show()
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
