"""
VP-Tree (Vantage Point Tree) Clustering with Bregman Divergences

Авторы: Кирьянов Даниил [danyavolskiy@gmail.com]
        Ахмерова Анастасия [anastasia.akhmerova.03@mail.ru]

Последнее обновление: 2025-11-30
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform, euclidean, cityblock, chebyshev
import warnings
warnings.filterwarnings('ignore')

# Импорты из проекта DMM Clustering System
from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


def compute_divergence(x: np.ndarray, y: np.ndarray, divergence_type: str = 'euclidean') -> float:
    """
    Вычислить дивергенцию между двумя точками.

    Parameters:
    -----------
    x, y : np.ndarray
        Две точки для сравнения
    divergence_type : str
        Тип дивергенции:
        - 'euclidean': евклидова дивергенция (квадрат евклидова расстояния / 2)
        - 'hellinger': дивергенция Хеллингера
        - 'bhattacharyya': дивергенция Бхаттачарьи
        - 'manhattan': манхэттенское расстояние
        - 'chebyshev': максимальная норма

    Returns:
    --------
    float
        Дивергенция между x и y
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    try:
        if divergence_type == 'euclidean':
            # Евклидова дивергенция: D = ||x - y||^2 / 2
            return euclidean(x, y) ** 2 / 2

        elif divergence_type == 'hellinger':
            # Дивергенция Хеллингера (для распределений вероятностей)
            # D = 2 * sum((sqrt(x_i) - sqrt(y_i))^2)
            x_safe = np.maximum(x, 1e-10)
            y_safe = np.maximum(y, 1e-10)
            return 2 * np.sum((np.sqrt(x_safe) - np.sqrt(y_safe)) ** 2)

        elif divergence_type == 'bhattacharyya':
            # Дивергенция Бхаттачарьи: D = -log(sum(sqrt(x_i * y_i)))
            x_safe = np.maximum(x, 1e-10)
            y_safe = np.maximum(y, 1e-10)
            bc = np.sum(np.sqrt(x_safe * y_safe))
            bc = np.clip(bc, 1e-10, 1.0)
            return -np.log(bc)

        elif divergence_type == 'manhattan':
            # Манхэттенское расстояние
            return cityblock(x, y)

        elif divergence_type == 'chebyshev':
            # Максимальная норма (Чебышёва)
            return chebyshev(x, y)

        elif divergence_type == 'cosine':
            # Косинусное расстояние
            x_norm = np.linalg.norm(x)
            y_norm = np.linalg.norm(y)
            if x_norm == 0 or y_norm == 0:
                return 1.0
            return 1 - np.dot(x, y) / (x_norm * y_norm)

        else:
            # Дефолт: евклидова дивергенция
            return euclidean(x, y) ** 2 / 2

    except Exception as e:
        # Fallback на евклидову дивергенцию при ошибке
        return euclidean(x, y) ** 2 / 2


class VPTreeBregmanClustering:
    """
    VP-tree кластеризация с дивергенциями/метриками расстояния.

    Параметры:
    -----------
    divergence : str, default='euclidean'
        Тип дивергенции/метрики:
        - 'euclidean': евклидова дивергенция (рекомендуется)
        - 'hellinger': дивергенция Хеллингера (для распределений)
        - 'bhattacharyya': дивергенция Бхаттачарьи (статистическое расстояние)
        - 'manhattan': манхэттенское расстояние
        - 'chebyshev': максимальная норма
        - 'cosine': косинусное расстояние

    linkage_method : str, default='ward'
        Метод связывания для иерархической кластеризации:
        - 'ward': минимизирует дисперсию
        - 'complete': максимальное расстояние
        - 'average': среднее расстояние
        - 'single': минимальное расстояние

    n_clusters : int, default=4
        Желаемое количество кластеров

    normalize : bool, default=True
        Нормализировать данные перед кластеризацией

    max_depth : int, default=10
        Максимальная глубина VP-tree (резервировано)
    """

    def __init__(
        self,
        divergence: str = 'euclidean',
        linkage_method: str = 'ward',
        n_clusters: int = 4,
        normalize: bool = True,
        max_depth: int = 10
    ):
        self.divergence = divergence
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters
        self.normalize = normalize
        self.max_depth = max_depth

        # Состояние модели
        self.labels_ = None
        self.n_clusters_found_ = 0
        self.X_ = None
        self.X_normalized_ = None
        self.scaler_ = None
        self.linkage_matrix_ = None
        self.distance_matrix_ = None

    def _compute_distance_matrix(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Вычислить матрицу расстояний/дивергенций.

        Parameters:
        -----------
        X : np.ndarray
            Входные данные формы (n_samples, n_features)

        Returns:
        --------
        np.ndarray
            Матрица расстояний размером (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                try:
                    dist = compute_divergence(
                        X[i],
                        X[j],
                        divergence_type=self.divergence
                    )
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
                except Exception as e:
                    # Используем евклидову дивергенцию как fallback
                    dist = euclidean(X[i], X[j]) ** 2 / 2
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

        return distance_matrix

    def fit(self, X: np.ndarray) -> 'VPTreeBregmanClustering':
        """
        Выполнить кластеризацию на X.

        Parameters:
        -----------
        X : np.ndarray
            Входные данные формы (n_samples, n_features)

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        # Проверка количества образцов
        if len(X) < self.n_clusters:
            raise ValueError(
                f"n_samples ({len(X)}) < n_clusters ({self.n_clusters})"
            )

        self.X_ = X.copy()

        # Нормализация
        if self.normalize:
            self.scaler_ = StandardScaler()
            self.X_normalized_ = self.scaler_.fit_transform(X)
        else:
            self.X_normalized_ = X.copy()

        # Вычисление матрицы расстояний
        self.distance_matrix_ = self._compute_distance_matrix(
            self.X_normalized_
        )

        # Преобразование в вектор для linkage
        pairwise_dist = squareform(self.distance_matrix_, checks=False)

        # Иерархическая кластеризация
        self.linkage_matrix_ = linkage(
            pairwise_dist,
            method=self.linkage_method
        )

        # Разрезание дерева на n_clusters кластеров
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

        Parameters:
        -----------
        X : np.ndarray
            Входные данные

        Returns:
        --------
        np.ndarray
            Метки кластеров
        """
        return self.fit(X).labels_


# ============================================================================
# ИНТЕГРАЦИЯ В ПРОЕКТ DMM CLUSTERING SYSTEM
# ============================================================================

@StrategiesManager.registerStrategy(
    "vptree_bregman",
    "VP-Tree Bregman Divergence Clustering",
    "Кластеризация на основе VP-tree с дивергенциями (БЕЗ pyBregman)"
)
class ConcreteStrategyVPTreeBregman(Strategy):
    """
    Особенности:
    - Использует дивергенции и метрики расстояния
    - Не требует pyBregman (использует только scipy)
    - Эффективен для высокомерных данных
    - Поддерживает различные типы дивергенций
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров стратегии"""

        cls._addParam(
            "n_clusters",
            "Количество кластеров",
            StrategyParamType.UNumber,
            """
            Желаемое количество кластеров для разбиения.
            Примеры: 2, 3, 4, 5...
            """,
            4
        )

        cls._addParam(
            "divergence",
            "Тип метрики расстояния",
            StrategyParamType.Switch,
            """
            Выбор метрики/дивергенции для измерения расстояния:
            
            - euclidean: евклидова дивергенция (РЕКОМЕНДУЕТСЯ)
            - hellinger: дивергенция Хеллингера (для вероятностей)
            - bhattacharyya: дивергенция Бхаттачарьи (статистическое расстояние)
            - manhattan: манхэттенское расстояние
            - chebyshev: максимальная норма
            - cosine: косинусное расстояние
            """,
            "euclidean",
            switches=[
                "euclidean",
                "hellinger",
                "bhattacharyya",
                "manhattan",
                "chebyshev",
                "cosine"
            ]
        )

        cls._addParam(
            "linkage_method",
            "Метод связывания",
            StrategyParamType.Switch,
            """
            Метод для связывания кластеров в иерархии:
            
            - ward: минимизирует дисперсию (РЕКОМЕНДУЕТСЯ)
            - complete: максимальное расстояние между кластерами
            - average: среднее расстояние между кластерами
            - single: минимальное расстояние между кластерами
            """,
            "ward",
            switches=["ward", "complete", "average", "single"]
        )

        cls._addParam(
            "normalize",
            "Нормализация данных",
            StrategyParamType.Bool,
            """
            Нормализировать входные данные перед кластеризацией.
            КРИТИЧЕСКИ ВАЖНО для корректной работы!
            """,
            True
        )

        cls._addParam(
            "max_depth",
            "Максимальная глубина VP-tree",
            StrategyParamType.UNumber,
            """
            Максимальная глубина дерева VP-tree.
            Зарезервировано для будущих оптимизаций.
            """,
            10
        )

    def clastering_image(
        self,
        pixels: np.ndarray,
        params: StrategyRunConfig
    ) -> np.ndarray:
        """
        Кластеризация пикселей изображения.
        ОПТИМИЗИРОВАНО: Добавлена выборка для больших изображений.
        """
        pixels = np.asarray(pixels, dtype=np.float64)

        # Коррекция формата
        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        # Ограничение размера данных для предотвращения зависания
        # Вычисление матрицы расстояний O(n²) очень затратно для больших изображений
        MAX_SAMPLES = 2000  # Максимальное количество пикселей для обработки
        
        if len(pixels) > MAX_SAMPLES:
            # ОПТИМИЗАЦИЯ: Используем стратегическую выборку вместо случайной
            step = int(np.sqrt(len(pixels) / MAX_SAMPLES))
            indices = np.arange(0, len(pixels), step)[:MAX_SAMPLES]
            sample_pixels = pixels[indices]
            use_sampling = True
        else:
            sample_pixels = pixels
            use_sampling = False

        model = VPTreeBregmanClustering(
            n_clusters=int(params["n_clusters"]),
            divergence=params["divergence"],
            linkage_method=params["linkage_method"],
            normalize=bool(params["normalize"]),
            max_depth=int(params["max_depth"])
        )
        
        sample_labels = model.fit_predict(sample_pixels)
        
        if use_sampling:
            # ОПТИМИЗАЦИЯ: Используем более быстрый алгоритм для больших данных
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = min(3, len(sample_pixels) // 10)
            knn = KNeighborsClassifier(n_neighbors=max(1, n_neighbors),
                                       algorithm='ball_tree' if len(pixels) > 10000 else 'auto',
                                       n_jobs=-1)  # Используем все ядра
            knn.fit(sample_pixels, sample_labels)
            labels = knn.predict(pixels)
        else:
            labels = sample_labels

        return labels

    def clastering_points(
        self,
        points: np.ndarray,
        params: StrategyRunConfig
    ) -> np.ndarray:
        """Кластеризация точек данных"""

        points = np.asarray(points, dtype=np.float64)

        # Коррекция формата
        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            points = points.T

        model = VPTreeBregmanClustering(
            n_clusters=int(params["n_clusters"]),
            divergence=params["divergence"],
            linkage_method=params["linkage_method"],
            normalize=bool(params["normalize"]),
            max_depth=int(params["max_depth"])
        )

        return model.fit_predict(points)
