"""
HPStream Clustering Algorithm Implementation
Автор: Балыков Павел [p.balykov@gmail.com]
Последнее обновление: 2025-11-30
"""


import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


class HPStreamClustering:
    """Алгоритм HPStream для проекционной кластеризации потоковых данных"""
    def __init__(self, num_clusters=5, projection_threshold=0.1,
                 window_size=1000, fade_threshold=0.1, normalize=True):

        self.num_clusters = num_clusters
        self.projection_threshold = projection_threshold
        self.window_size = window_size
        self.fade_threshold = fade_threshold
        self.normalize = normalize

        # Результаты кластеризации
        self.labels_ = None
        self.cluster_centers_ = None
        self.dimension_weights_ = None
        self.X_ = None
        self.X_normalized_ = None
        self.scaler_ = None


    def _initialize_weights(self, n_features: int) -> np.ndarray:
        """Инициализация весов размерностей"""
        return np.ones(n_features) / n_features

    def _update_dimension_weights(self, data: np.ndarray, cluster_centers: np.ndarray,
                                labels: np.ndarray) -> np.ndarray:

        """Обновление весов размерностей на основе дисперсии"""
        n_features = data.shape[1]
        new_weights = np.zeros(n_features)

        # Вычисление дисперсии по кластерам
        for k in range(self.num_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                cluster_variance = np.var(cluster_points, axis=0)
                if np.sum(cluster_variance) > 0:
                    cluster_variance = cluster_variance / np.sum(cluster_variance)
                new_weights += cluster_variance

        # Усреднение и применение затухания
        if self.num_clusters > 0:
            new_weights = new_weights / self.num_clusters
        if self.dimension_weights_ is not None:
            new_weights = (1 - self.fade_threshold) * self.dimension_weights_ + self.fade_threshold * new_weights


        # Нормализация весов
        if np.sum(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
        return new_weights



    def _select_projected_dimensions(self, weights: np.ndarray) -> np.ndarray:
        """Выбор значимых размерностей по порогу"""
        return weights >= self.projection_threshold



    def _projected_distance(self, x: np.ndarray, y: np.ndarray, projected_dims: np.ndarray) -> float:
        """Вычисление расстояния только по значимым размерностям"""
        if np.sum(projected_dims) == 0:
            return 0.0
        return np.sqrt(np.sum((x[projected_dims] - y[projected_dims]) ** 2))



    def _initialize_clusters(self, data: np.ndarray) -> np.ndarray:
        """Инициализация центров кластеров методом K-means++"""
        n_samples, n_features = data.shape
        centers = np.zeros((self.num_clusters, n_features))
        centers[0] = data[np.random.randint(n_samples)]
        for i in range(1, self.num_clusters):
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                min_dist = float('inf')
                for k in range(i):
                    dist = np.linalg.norm(data[j] - centers[k])
                    if dist < min_dist:
                        min_dist = dist
                distances[j] = min_dist
            probabilities = distances ** 2
            probabilities /= np.sum(probabilities)
            centers[i] = data[np.random.choice(n_samples, p=probabilities)]
        return centers



    def fit(self, X: np.ndarray, max_iterations: int = 100) -> 'HPStreamClustering':
        """Обучение алгоритма на данных"""
        X = np.asarray(X, dtype=np.float64)
        if len(X) < self.num_clusters:
            raise ValueError(f"n_samples ({len(X)}) < n_clusters ({self.num_clusters})")
        self.X_ = X.copy()
        # Нормализация данных
        if self.normalize:
            self.scaler_ = StandardScaler()
            self.X_normalized_ = self.scaler_.fit_transform(X)
        else:
            self.X_normalized_ = X.copy()
        n_samples, n_features = self.X_normalized_.shape

        # Инициализация весов и центров
        self.dimension_weights_ = self._initialize_weights(n_features)
        self.cluster_centers_ = self._initialize_clusters(self.X_normalized_)

        # Основной цикл кластеризации
        for iteration in range(max_iterations):
            projected_dims = self._select_projected_dimensions(self.dimension_weights_)

            # Назначение точек кластерам
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                min_dist = float('inf')
                for k in range(self.num_clusters):
                    dist = self._projected_distance(self.X_normalized_[i],
                                                    self.cluster_centers_[k],
                                                    projected_dims)
                    if dist < min_dist:
                        min_dist = dist
                        labels[i] = k

            # Обновление центров кластеров
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.num_clusters):
                cluster_points = self.X_normalized_[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]

            # Проверка сходимости
            if np.allclose(new_centers, self.cluster_centers_, rtol=1e-4):
                break

            self.cluster_centers_ = new_centers

            # Обновление весов размерностей
            self.dimension_weights_ = self._update_dimension_weights(self.X_normalized_,
                                                                     self.cluster_centers_,
                                                                     labels)
        self.labels_ = labels
        return self


    def partial_fit(self, X_batch: np.ndarray) -> 'HPStreamClustering':
        """Инкрементальное обучение на новом батче данных"""
        if self.cluster_centers_ is None:
            return self.fit(X_batch)

        X_batch = np.asarray(X_batch, dtype=np.float64)

        if self.normalize and self.scaler_ is not None:
            X_batch_normalized = self.scaler_.transform(X_batch)
        else:
            X_batch_normalized = X_batch


        # Объединение с предыдущими данными
        if hasattr(self, 'X_normalized_') and self.X_normalized_ is not None:
            combined_data = np.vstack([self.X_normalized_[-self.window_size:], X_batch_normalized])
        else:
            combined_data = X_batch_normalized

        self.X_normalized_ = combined_data
        return self.fit(combined_data)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Обучение и возврат меток кластеров"""
        return self.fit(X).labels_


@StrategiesManager.registerStrategy(
    "hpstream",
    "HPStream Projected Clustering",
    "Проекционная кластеризация потоковых данных высоких размерностей"
)

class ConcreteStrategyHPStream(Strategy):
    """Стратегия HPStream для DMM Clustering System"""
    @classmethod
    def _setupParams(cls):
        """Настройка параметров стратегии"""
        cls._addParam(
            "num_clusters",
            "Количество кластеров",
            StrategyParamType.UNumber,
            "Желаемое количество кластеров",
            5
        )

        cls._addParam(
            "projection_threshold",
            "Порог проекции",
            StrategyParamType.UNumber,
            "Порог для определения значимых размерностей",
            0.1
        )

        cls._addParam(
            "window_size",
            "Размер окна",
            StrategyParamType.UNumber,
            "Размер окна для потоковых данных",
            1000
        )

        cls._addParam(
            "fade_threshold",
            "Порог затухания",
            StrategyParamType.UNumber,
            "Коэффициент затухания весов размерностей",
            0.1
        )

        cls._addParam(
            "max_iterations",
            "Макс. итераций",
            StrategyParamType.UNumber,
            "Максимальное количество итераций",
            100
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация изображения.
        ОПТИМИЗИРОВАНО: Добавлена выборка для больших изображений.
        """
        pixels = np.asarray(pixels, dtype=np.float64)

        # Коррекция формата данных
        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        # Ограничение размера данных для предотвращения зависания
        # HPStream может быть медленным на больших изображениях
        MAX_SAMPLES = 5000  # Максимальное количество пикселей для обработки
        
        if len(pixels) > MAX_SAMPLES:
            # ОПТИМИЗАЦИЯ: Используем стратегическую выборку вместо случайной
            step = int(np.sqrt(len(pixels) / MAX_SAMPLES))
            indices = np.arange(0, len(pixels), step)[:MAX_SAMPLES]
            sample_pixels = pixels[indices]
            use_sampling = True
        else:
            sample_pixels = pixels
            use_sampling = False

        model = HPStreamClustering(
            num_clusters=int(params["num_clusters"]),
            projection_threshold=float(params["projection_threshold"]),
            window_size=int(params["window_size"]),
            fade_threshold=float(params["fade_threshold"]),
            normalize=True
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

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """Кластеризация точек"""
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            points = points.T

        model = HPStreamClustering(
            num_clusters=int(params["num_clusters"]),
            projection_threshold=float(params["projection_threshold"]),
            window_size=int(params["window_size"]),
            fade_threshold=float(params["fade_threshold"]),
            normalize=True
        )
        return model.fit_predict(points)
