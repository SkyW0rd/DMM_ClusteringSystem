"""
CLARANS Clustering Algorithm Implementation
Стратегия кластеризации на основе алгоритма CLARANS из библиотеки pyclustering.

CLARANS (Clustering Large Applications based on RANdomized Search) —
алгоритм кластеризации на основе K-Medoids с randomized search.
В отличие от K-Means (центроиды = среднее), использует реальные точки
данных как медоиды (центры кластеров). Оптимизирует выбор медоидов
через случайный поиск соседей.
"""

import numpy as np
from pyclustering.cluster.clarans import clarans

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


@StrategiesManager.registerStrategy(
    "clarans",
    "CLARANS",
    "K-Medoids with randomized search (pyclustering)"
)
class ConcreteStrategyCLARANS(Strategy):
    """Метод кластеризации с использованием CLARANS из pyclustering.

    CLARANS ищет оптимальные медоиды (реальные точки данных как центры
    кластеров) через случайный поиск. Чем больше numlocal и maxneighbor —
    тем точнее результат, но медленнее работа.
    """

    @classmethod
    def _setupParams(cls):
        cls._addParam(
            "number_clusters",
            "Количество кластеров",
            StrategyParamType.UNumber,
            """(Number clusters) Количество кластеров для выделения.
            Алгоритм ищет указанное количество медоидов.""",
            3
        )
        cls._addParam(
            "numlocal",
            "Количество локальных минимумов",
            StrategyParamType.UNumber,
            """(Numlocal) Количество локальных минимумов для поиска
            (amount of iterations for solving the problem).
            Больше значений — лучше результат, но медленнее.""",
            2
        )
        cls._addParam(
            "maxneighbor",
            "Макс. количество соседей",
            StrategyParamType.UNumber,
            """(Maxneighbor) Максимальное количество соседей для проверки
            за одну итерацию. Чем больше, тем ближе CLARANS к K-Medoids
            и тем дольше каждый поиск локального минимума.""",
            5
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64)

        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        instance = clarans(
            data=pixels.tolist(),
            number_clusters=int(params["number_clusters"]),
            numlocal=int(params["numlocal"]),
            maxneighbor=int(params["maxneighbor"])
        )
        instance.process()
        return np.array(self.clusters_to_labels(instance.get_clusters()))

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)

        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            points = points.T

        instance = clarans(
            data=points.tolist(),
            number_clusters=int(params["number_clusters"]),
            numlocal=int(params["numlocal"]),
            maxneighbor=int(params["maxneighbor"])
        )
        instance.process()
        return np.array(self.clusters_to_labels(instance.get_clusters()))
