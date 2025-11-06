"""
Тестовый скрипт для проверки DBSCAN с визуализацией
Автор: Данила Антонович sidorov.d.a1@edu.mirea.ru и Виктор Бредихин vibread@mail.ru

Описание:
Этот скрипт проверяет работу алгоритма DBSCAN и создает
визуализации для 2D и 3D данных, а также тестирует интеграцию
с паттерном Strategy из основного фреймворка.
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
# НАСТРОЙКИ
# ============================================================================

ALGORITHM_NAME = "DBSCAN"

try:
    from ClusteringMethods.ClasteringAlgorithms import (
        ConcreteStrategyDBSCAN_from_SKLEARN,
        Context,
        StrategiesManager
    )
    ALGORITHM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Предупреждение: {e}")
    ALGORITHM_AVAILABLE = False

STRATEGY_AVAILABLE = ALGORITHM_AVAILABLE

# ============================================================================
# НАСТРОЙКА ПУТЕЙ
# ============================================================================

TEST_DIR = Path(__file__).parent
IMAGES_DIR = TEST_DIR / "Images" / ALGORITHM_NAME
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"📁 Директория для результатов: {IMAGES_DIR}")
print("="*80 + "\n")

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def generate_test_data_2d(dataset_type='blobs', n_samples=300, **kwargs):
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
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=3,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=42
    )
    return X, y_true

def save_figure(fig, filename, dpi=150):
    output_path = IMAGES_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Сохранено: {output_path.name}")

# ============================================================================
# ТЕСТОВЫЕ ФУНКЦИИ
# ============================================================================

def test_basic_2d_clustering():
    print("="*80)
    print("ТЕСТ 1: Базовая 2D Кластеризация")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    X, y_true = generate_test_data_2d('blobs', n_samples=300, centers=3, cluster_std=0.5)
    
    config = StrategiesManager.getStrategyRunConfigById("dbscan_sk")
    config["eps"] = 0.5
    config["min_samples"] = 5
    
    strategy = ConcreteStrategyDBSCAN_from_SKLEARN()
    y_pred = strategy.clastering_points(X, config)
    
    n_clusters_pred = len(np.unique(y_pred[y_pred != -1]))
    n_clusters_true = len(np.unique(y_true))
    n_noise = np.sum(y_pred == -1)
    
    print(f"Количество точек: {len(X)}")
    print(f"Истинное количество кластеров: {n_clusters_true}")
    print(f"Найдено кластеров: {n_clusters_pred}")
    print(f"Шумовых точек: {n_noise}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                              alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[0].set_title('Истинные Кластеры', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X₁', fontsize=12)
    axes[0].set_ylabel('X₂', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Истинная метка')
    
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                              alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[1].set_title(f'{ALGORITHM_NAME} Результаты\n({n_clusters_pred} кластеров, {n_noise} шум)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X₁', fontsize=12)
    axes[1].set_ylabel('X₂', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Предсказанная метка')
    
    plt.tight_layout()
    save_figure(fig, 'test_basic_2d.png')
    print()

def test_parameter_sensitivity():
    print("="*80)
    print("ТЕСТ 2: Чувствительность к Параметрам (eps)")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    X, y_true = generate_test_data_2d('blobs', n_samples=300, centers=3)
    
    eps_values = [0.3, 0.5, 0.7, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    
    for i, eps_val in enumerate(eps_values):
        config = StrategiesManager.getStrategyRunConfigById("dbscan_sk")
        config["eps"] = eps_val
        config["min_samples"] = 5
        
        strategy = ConcreteStrategyDBSCAN_from_SKLEARN()
        y_pred = strategy.clastering_points(X, config)
        n_clusters = len(np.unique(y_pred[y_pred != -1]))
        n_noise = np.sum(y_pred == -1)
        
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                       alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'eps={eps_val}\n({n_clusters} кластеров, {n_noise} шум)',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].grid(True, alpha=0.3)
        
        print(f"  eps={eps_val}: {n_clusters} кластеров, {n_noise} шум")
    
    plt.tight_layout()
    save_figure(fig, 'test_parameters.png')
    print()

def test_different_datasets():
    print("="*80)
    print("ТЕСТ 3: Разные Типы Датасетов")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    datasets = [
        ("Blobs (Сферические)", 'blobs', {'centers': 3, 'cluster_std': 0.5}),
        ("Moons (Полумесяцы)", 'moons', {'noise': 0.05}),
        ("Circles (Круги)", 'circles', {'noise': 0.05, 'factor': 0.5}),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, dataset_type, params) in enumerate(datasets):
        X, _ = generate_test_data_2d(dataset_type, n_samples=300, **params)
        
        config = StrategiesManager.getStrategyRunConfigById("dbscan_sk")
        config["eps"] = 0.3 if dataset_type != 'blobs' else 0.5
        config["min_samples"] = 5
        
        strategy = ConcreteStrategyDBSCAN_from_SKLEARN()
        y_pred = strategy.clastering_points(X, config)
        n_clusters = len(np.unique(y_pred[y_pred != -1]))
        n_noise = np.sum(y_pred == -1)
        
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10',
                       alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[i].set_title(f'{name}\n({n_clusters} кластеров, {n_noise} шум)',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X₁')
        axes[i].set_ylabel('X₂')
        axes[i].grid(True, alpha=0.3)
        
        print(f"  {name}: {n_clusters} кластеров, {n_noise} шум")
    
    plt.tight_layout()
    save_figure(fig, 'test_datasets.png')
    print()

def test_3d_clustering():
    print("="*80)
    print("ТЕСТ 4: 3D Кластеризация")
    print("="*80)
    
    if not ALGORITHM_AVAILABLE:
        print("❌ Алгоритм не доступен. Пропускаем тест.")
        return
    
    X, y_true = generate_test_data_3d(n_samples=300, n_clusters=3)
    
    config = StrategiesManager.getStrategyRunConfigById("dbscan_sk")
    config["eps"] = 0.7
    config["min_samples"] = 5
    
    strategy = ConcreteStrategyDBSCAN_from_SKLEARN()
    y_pred = strategy.clastering_points(X, config)
    
    n_clusters = len(np.unique(y_pred[y_pred != -1]))
    n_noise = np.sum(y_pred == -1)
    print(f"Найдено кластеров: {n_clusters}, шум: {n_noise}")
    
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true, cmap='viridis',
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax1.set_title('3D: Истинные Кластеры', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('X₃')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap='tab10',
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax2.set_title(f'3D: {ALGORITHM_NAME}\n({n_clusters} кластеров)', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_zlabel('X₃')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
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
    print("="*80)
    print("ТЕСТ 5: Интеграция с Фреймворком (Strategy Pattern)")
    print("="*80)
    
    if not STRATEGY_AVAILABLE:
        print("❌ Strategy Pattern не доступен. Пропускаем тест.")
        return
    
    X, y_true = generate_test_data_2d('blobs', n_samples=200, centers=2)
    
    try:
        config = StrategiesManager.getStrategyRunConfigById("dbscan_sk")
        config["eps"] = 0.5
        config["min_samples"] = 5
        
        strategy = ConcreteStrategyDBSCAN_from_SKLEARN()
        context = Context(strategy)
        
        y_pred = context.do_some_clustering_points(X.T, config)
        
        n_clusters = len(np.unique(y_pred[y_pred != -1]))
        n_noise = np.sum(y_pred == -1)
        print(f"Найдено кластеров: {n_clusters}, шум: {n_noise}")
        print("✅ Интеграция с фреймворком работает!")
        
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
    print("\n" + "="*80)
    print(f"ТЕСТИРОВАНИЕ АЛГОРИТМА: {ALGORITHM_NAME}")
    print("="*80 + "\n")
    
    if not ALGORITHM_AVAILABLE:
        print("❌ ОШИБКА: Алгоритм не найден!")
        return
    
    try:
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
        
        plt.show()
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


