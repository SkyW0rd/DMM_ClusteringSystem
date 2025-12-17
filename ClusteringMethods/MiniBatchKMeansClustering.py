"""
MiniBatchKMeans Clustering Algorithm Implementation
Автор: Левинский Григорий [glevinskiy@gmail.com]
Последнее обновление: 2025-12-16
"""

import numpy as np
from typing import Dict, List

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ВАЖНО: Импорты из проекта DMM Clustering System
from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


@StrategiesManager.registerStrategy(
    "minibatchkmeans_sk",
    "MiniBatchKMeans (SKLearn)",
    "Кластеризация MiniBatchKMeans из sklearn"
)
class ConcreteStrategyMiniBatchKMeans_from_SKLEARN(Strategy):
    """
    Использует MiniBatchKMeans из sklearn
    
    MiniBatchKMeans - это быстрая версия KMeans, которая использует
    мини-батчи для обработки данных. Подходит для больших наборов данных.
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров"""

        cls._addParam(
            "n_clusters",
            "Количество кластеров",
            StrategyParamType.UNumber,
            """
            Желаемое количество кластеров для MiniBatchKMeans.
            Примеры: 2, 3, 4, 5...
            """,
            3
        )

        cls._addParam(
            "init",
            "Метод инициализации",
            StrategyParamType.Switch,
            """
            Метод инициализации центроидов:
            
            - k-means++: умная инициализация (рекомендуется)
            - random: случайная инициализация
            """,
            "k-means++",
            switches=["k-means++", "random"]
        )

        cls._addParam(
            "max_iter",
            "Максимум итераций",
            StrategyParamType.UNumber,
            """
            Максимальное количество итераций алгоритма MiniBatchKMeans.
            Рекомендуется: 100-300.
            """,
            100
        )

        cls._addParam(
            "batch_size",
            "Размер батча",
            StrategyParamType.UNumber,
            """
            Размер мини-батча для обработки данных.
            Большие значения дают более точные результаты, но медленнее.
            Рекомендуется: 100-1000.
            """,
            100
        )

        cls._addParam(
            "verbose",
            "Подробный вывод",
            StrategyParamType.Bool,
            """
            Включить/выключить подробный вывод процесса кластеризации.
            """,
            False
        )

        cls._addParam(
            "compute_labels",
            "Вычислять метки",
            StrategyParamType.Bool,
            """
            Вычислять метки для всех точек после обучения.
            Если False, метки будут вычисляться только при необходимости.
            """,
            True
        )

        cls._addParam(
            "random_state",
            "Состояние случайности",
            StrategyParamType.UNumber,
            """
            Инициализация генератора случайных чисел для воспроизводимости результатов.
            """,
            42
        )

        cls._addParam(
            "reassignment_ratio",
            "Коэффициент перераспределения",
            StrategyParamType.UFloating,
            """
            Коэффициент перераспределения центроидов.
            Контролирует частоту перераспределения центров кластеров.
            Рекомендуется: 0.01-0.1.
            """,
            0.01
        )

        cls._addParam(
            "use_pca",
            "Использовать PCA",
            StrategyParamType.Bool,
            """
            Включить/выключить PCA декомпозицию перед кластеризацией.
            Если False, используется только стандартизация данных.
            """,
            True
        )

        cls._addParam(
            "n_components",
            "Количество компонент PCA",
            StrategyParamType.UNumber,
            """
            Количество главных компонент для PCA декомпозиции.
            Если 0, используется автоматический выбор (min(n_samples, n_features)).
            Рекомендуется: 2-3 для визуализации, больше для сохранения информации.
            """,
            0  # 0 означает автоматический выбор
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация изображения методом MiniBatchKMeans.
        """
        pixels = np.asarray(pixels, dtype=np.float64)

        # Коррекция формата данных
        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        # Стандартизация данных
        scaler = StandardScaler()
        pixel_values_scaled = scaler.fit_transform(pixels)

        # PCA декомпозиция (если включена)
        pca = None
        if params["use_pca"]:
            n_components = params["n_components"]
            if n_components == 0 or n_components is None:
                # Автоматический выбор: минимум между количеством образцов и признаков
                n_components = min(pixels.shape[0], pixels.shape[1])
                # Ограничиваем максимальное количество компонент для производительности
                n_components = min(n_components, 50)

            pca = PCA(n_components=n_components)
            pixel_values_scaled = pca.fit_transform(pixel_values_scaled)

        # MiniBatchKMeans кластеризация
        model = MiniBatchKMeans(
            n_clusters=int(params["n_clusters"]),
            init=params["init"],
            max_iter=int(params["max_iter"]),
            batch_size=int(params["batch_size"]),
            verbose=bool(params["verbose"]),
            compute_labels=bool(params["compute_labels"]),
            random_state=int(params["random_state"]) if params["random_state"] is not None else None,
            reassignment_ratio=float(params["reassignment_ratio"])
        )
        labels = model.fit_predict(pixel_values_scaled)

        return labels

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом MiniBatchKMeans.
        """
        points = np.asarray(points, dtype=np.float64)

        # Коррекция формата данных
        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            points = points.T

        # Стандартизация данных
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)

        # PCA декомпозиция (если включена)
        if params["use_pca"]:
            n_components = params["n_components"]
            if n_components == 0 or n_components is None:
                # Автоматический выбор: минимум между количеством образцов и признаков
                n_components = min(points.shape[0], points.shape[1])
                # Ограничиваем максимальное количество компонент для производительности
                n_components = min(n_components, 50)

            pca = PCA(n_components=n_components)
            points_scaled = pca.fit_transform(points_scaled)

        # MiniBatchKMeans кластеризация
        model = MiniBatchKMeans(
            n_clusters=int(params["n_clusters"]),
            init=params["init"],
            max_iter=int(params["max_iter"]),
            batch_size=int(params["batch_size"]),
            verbose=bool(params["verbose"]),
            compute_labels=bool(params["compute_labels"]),
            random_state=int(params["random_state"]) if params["random_state"] is not None else None,
            reassignment_ratio=float(params["reassignment_ratio"])
        )
        labels = model.fit_predict(points_scaled)

        return labels

