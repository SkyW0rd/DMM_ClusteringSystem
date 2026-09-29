"""
CLIQUE Clustering Algorithm Implementation
Стратегия кластеризации на основе алгоритма CLIQUE из библиотеки pyclustering.

CLIQUE (CLustering In QUEst) — density-based grid-алгоритм кластеризации.
Данные разбиваются на grid-ячейки по каждому измерению, плотные ячейки
объединяются в кластеры, точки в разреженных ячейках помечаются как шум (-1).
"""

import numpy as np
from pyclustering.cluster.clique import clique

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


@StrategiesManager.registerStrategy(
    "clique",
    "CLIQUE",
    "Grid-based density clustering (pyclustering)"
)
class ConcreteStrategyCLIQUE(Strategy):
    """Метод кластеризации с использованием CLIQUE из pyclustering.

    CLIQUE разбивает пространство на grid-ячейки и объединяет плотные
    смежные ячейки в кластеры. Точки в разреженных ячейках помечаются как шум.
    """

    @classmethod
    def _setupParams(cls):
        cls._addParam(
            "amount_intervals",
            "Количество интервалов",
            StrategyParamType.UNumber,
            """(Amount intervals) Количество интервалов в каждом измерении,
            определяет количество блоков сетки CLIQUE.
            Больше интервалов — мельче сетка, больше кластеров, меньше шума.""",
            5
        )
        cls._addParam(
            "density_threshold",
            "Порог плотности",
            StrategyParamType.UNumber,
            """(Density threshold) Минимальное количество точек в блоке сетки,
            чтобы точки не считались выбросами (шумом).
            Больше порог — больше точек помечаются как шум.""",
            10
        )
        cls._addParam(
            "ccore",
            "Использовать C++",
            StrategyParamType.Bool,
            """Если истинно, тогда используется C++ часть библиотеки для обработки.
            Автоматически отключается, если C++ ядро недоступно.""",
            True
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64)

        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        instance = clique(
            data=pixels.tolist(),
            amount_intervals=int(params["amount_intervals"]),
            density_threshold=int(params["density_threshold"]),
            ccore=bool(params["ccore"])
        )
        instance.process()
        return self._clusters_and_noise_to_labels(
            instance.get_clusters(),
            instance.get_noise(),
            len(pixels)
        )

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)

        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            points = points.T

        instance = clique(
            data=points.tolist(),
            amount_intervals=int(params["amount_intervals"]),
            density_threshold=int(params["density_threshold"]),
            ccore=bool(params["ccore"])
        )
        instance.process()
        return self._clusters_and_noise_to_labels(
            instance.get_clusters(),
            instance.get_noise(),
            len(points)
        )

    def _clusters_and_noise_to_labels(self, clusters, noise, size) -> np.ndarray:
        """Преобразует кластеры и шум из CLIQUE в массив меток.

        Точки шума получают метку -1 (как в DBSCAN).
        """
        labels = np.full(size, -1, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for i in cluster:
                labels[i] = cluster_idx
        return labels
