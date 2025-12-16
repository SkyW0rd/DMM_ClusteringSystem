"""
BallTree Clustering Algorithm Implementation
Авторы: Кирьянов Даниил [danyavolskiy@gmail.com] Ахмерова Анастасия [anastasia.akhmerova.03@mail.ru]
Последнее обновление: 2025-9-11
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# ВАЖНО: Импорты из проекта DMM Clustering System
from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


class BallTreeClustering:
    """
    Parameters:
    -----------
    n_neighbors : int, default=5
        Количество ближайших соседей.

    leaf_size : int, default=40
        Размер листа BallTree.

    metric : str, default='euclidean'
        Метрика расстояния.

    linkage_method : str, default='ward'
        Метод linkage для иерархии.

    n_clusters : int, default=4
        Количество кластеров.

    normalize : bool, default=True
        Нормализировать данные перед кластеризацией (ВАЖНО!).
    """

    def __init__(self, n_neighbors=5, leaf_size=40, metric='euclidean',
                 linkage_method='ward', n_clusters=4, normalize=True):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.metric = metric
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters
        self.normalize = normalize

        self.labels_ = None
        self.n_clusters_found_ = 0
        self.X_ = None
        self.X_normalized_ = None
        self.scaler_ = None
        self.linkage_matrix_ = None

    def fit(self, X: np.ndarray) -> 'BallTreeClustering':
        """
        Выполнить кластеризацию на X.
        """
        X = np.asarray(X, dtype=np.float64)

        if len(X) < self.n_clusters:
            raise ValueError(
                f"n_samples ({len(X)}) < n_clusters ({self.n_clusters})"
            )

        self.X_ = X.copy()

        if self.normalize:
            self.scaler_ = StandardScaler()
            self.X_normalized_ = self.scaler_.fit_transform(X)
        else:
            self.X_normalized_ = X.copy()

        pairwise_dist = pdist(self.X_normalized_, metric=self.metric)

        self.linkage_matrix_ = linkage(pairwise_dist, method=self.linkage_method)

        self.labels_ = fcluster(
            self.linkage_matrix_,
            self.n_clusters,
            criterion='maxclust'
        ) - 1  # Привести к 0-индексации

        self.n_clusters_found_ = len(np.unique(self.labels_))

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Вычислить кластеры и вернуть метки.
        """
        return self.fit(X).labels_


# ============================================================================
# ИНТЕГРАЦИЯ В ПРОЕКТ DMM CLUSTERING SYSTEM
# ============================================================================

@StrategiesManager.registerStrategy(
    "balltree",
    "BallTree Hierarchical Clustering",
    "Иерархическая кластеризация с использованием BallTree и linkage"
)
class ConcreteStrategyBallTree(Strategy):
    """
    BallTree strategy для DMM Clustering System (v2.1 - CORRECTED).
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров"""

        cls._addParam(
            "n_clusters",
            "Количество кластеров",
            StrategyParamType.UNumber,
            """
            Желаемое количество кластеров.
            
            В отличие от DBSCAN, здесь ОБЯЗАТЕЛЬНО указывать количество.
            Примеры: 2, 3, 4, 5...
            """,
            4
        )

        cls._addParam(
            "linkage_method",
            "Метод linkage",
            StrategyParamType.Switch,
            """
            Метод объединения кластеров в иерархии:
            
            - ward: минимизирует дисперсию (РЕКОМЕНДУЕТСЯ)
            - complete: максимальное расстояние
            - average: среднее расстояние
            - single: минимальное расстояние
            """,
            "ward",
            switches=["ward", "complete", "average", "single"]
        )

        cls._addParam(
            "metric",
            "Метрика расстояния",
            StrategyParamType.Switch,
            """
            Метрика для расчета расстояния:
            
            - euclidean: евклидово (рекомендуется)
            - manhattan: манхэттенское
            - chebyshev: максимальная норма
            """,
            "euclidean",
            switches=["euclidean", "manhattan", "chebyshev"]
        )

        cls._addParam(
            "n_neighbors",
            "K для k-NN (не используется в v2.1)",
            StrategyParamType.UNumber,
            """
            Зарезервировано для совместимости.
            В v2.1 не используется (используется полная матрица расстояний).
            """,
            5
        )

        cls._addParam(
            "leaf_size",
            "Размер листа BallTree (не используется в v2.1)",
            StrategyParamType.UNumber,
            """
            Зарезервировано для совместимости.
            В v2.1 не используется.
            """,
            40
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """Кластеризация изображения"""
        pixels = np.asarray(pixels, dtype=np.float64)

        # Коррекция формата
        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        model = BallTreeClustering(
            n_clusters=int(params["n_clusters"]),
            linkage_method=params["linkage_method"],
            metric=params["metric"],
            normalize=True  # КРИТИЧЕСКИ ВАЖНО!
        )

        return model.fit_predict(pixels)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """Кластеризация точек"""
        points = np.asarray(points, dtype=np.float64)

        # Коррекция формата
        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            points = points.T

        model = BallTreeClustering(
            n_clusters=int(params["n_clusters"]),
            linkage_method=params["linkage_method"],
            metric=params["metric"],
            normalize=True  # КРИТИЧЕСКИ ВАЖНО!
        )

        return model.fit_predict(points)
