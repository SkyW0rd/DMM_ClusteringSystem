"""
WaveClustering Algorithm Implementation
Автор: Мысин Юрий [yuriymysin@yandex.ru]
Последнее обновление: 2025-12-11
"""

import numpy as np
from typing import Dict, List

from sklearn.cluster import KMeans
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
    "kmeans_sk",
    "KMeans (SKLearn)",
    "Кластеризация KMeans из sklearn"
)
class ConcreteStrategyKMeans_from_SKLEARN(Strategy):
    """
    Использует KMeans из sklearn
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров"""

        cls._addParam(
            "n_clusters",
            "Количество кластеров",
            StrategyParamType.UNumber,
            """
            Желаемое количество кластеров для KMeans.
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
            "n_init",
            "Количество инициализаций",
            StrategyParamType.UNumber,
            """
            Количество различных инициализаций для выбора лучшего результата.
            Больше значений = лучше результат, но медленнее.
            Рекомендуется: 10-20.
            """,
            10
        )

        cls._addParam(
            "max_iter",
            "Максимум итераций",
            StrategyParamType.UNumber,
            """
            Максимальное количество итераций алгоритма KMeans.
            Рекомендуется: 300-500.
            """,
            300
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
        Кластеризация изображения методом KMeans.
        """
        pixels = np.asarray(pixels, dtype=np.float64)

        # Коррекция формата данных
        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        # Ограничение размера данных для предотвращения зависания
        MAX_SAMPLES = 10000  # Максимальное количество пикселей для обработки

        if len(pixels) > MAX_SAMPLES:
            # Равномерная выборка по сетке для лучшего покрытия изображения
            step = int(np.sqrt(len(pixels) / MAX_SAMPLES))
            indices = np.arange(0, len(pixels), step)[:MAX_SAMPLES]
            sample_pixels = pixels[indices]
            use_sampling = True
        else:
            sample_pixels = pixels
            use_sampling = False

        # Стандартизация данных
        scaler = StandardScaler()
        pixel_values_scaled = scaler.fit_transform(sample_pixels)

        # PCA декомпозиция (если включена)
        pca = None
        if params["use_pca"]:
            n_components = params["n_components"]
            if n_components == 0 or n_components is None:
                # Автоматический выбор: минимум между количеством образцов и признаков
                n_components = min(sample_pixels.shape[0], sample_pixels.shape[1])
                # Ограничиваем максимальное количество компонент для производительности
                n_components = min(n_components, 50)

            pca = PCA(n_components=n_components)
            pixel_values_scaled = pca.fit_transform(pixel_values_scaled)

        # KMeans кластеризация
        model = KMeans(
            n_clusters=int(params["n_clusters"]),
            init=params["init"],
            n_init=int(params["n_init"]),
            max_iter=int(params["max_iter"]),
            random_state=int(params["random_state"]) if params["random_state"] is not None else None
        )
        sample_labels = model.fit_predict(pixel_values_scaled)

        if use_sampling:
            # Применяем те же преобразования к полному набору данных
            pixels_scaled = scaler.transform(pixels)
            if params["use_pca"] and pca is not None:
                pixels_scaled = pca.transform(pixels_scaled)
            labels = model.predict(pixels_scaled)
        else:
            labels = sample_labels

        return labels

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом KMeans.
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

        # KMeans кластеризация
        model = KMeans(
            n_clusters=int(params["n_clusters"]),
            init=params["init"],
            n_init=int(params["n_init"]),
            max_iter=int(params["max_iter"]),
            random_state=int(params["random_state"]) if params["random_state"] is not None else None
        )
        labels = model.fit_predict(points_scaled)

        return labels
