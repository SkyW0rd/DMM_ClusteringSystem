"""Spherical K-Means для кластеризации направлений и эмбеддингов.

Автор — Кулаков Никита.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy, StrategyParamType, StrategyRunConfig, StrategiesManager,
)
from ClusteringMethods._validation import (
    positive_integer, positive_real, prepare_features,
)


def _unit_rows(features):
    """Нормировать строки, сохраняя нули и избегая переполнения нормы."""
    scale = np.max(np.abs(features), axis=1, keepdims=True)
    scaled = np.divide(features, scale, out=np.zeros_like(features), where=scale != 0)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norms, out=np.zeros_like(scaled), where=norms != 0)


class SphericalKMeans(BaseEstimator):
    """Минимизировать сумму косинусных расстояний до единичных центроидов.

    Вход имеет форму (n_samples, n_features). Нулевые векторы получают
    метку -1 и исключаются из обучения. Каждый запуск использует локальный
    генератор случайных чисел. Расстояния вычисляются блоками batch_size.
    """

    def __init__(self, n_clusters=3, init="k-means++", n_init=10,
                 max_iter=300, tol=0.0001, random_state=42, batch_size=4096):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.batch_size = batch_size

    def _assign(self, features, centers):
        """Найти ближайшие направления без полной матрицы n_samples x k."""
        labels = np.empty(len(features), dtype=np.intp)
        distances = np.empty(len(features), dtype=np.float64)
        for start in range(0, len(features), self.batch_size):
            stop = min(start + self.batch_size, len(features))
            similarities = features[start:stop] @ centers.T
            closest = np.argmax(similarities, axis=1)
            labels[start:stop] = closest
            distances[start:stop] = np.clip(
                1.0 - similarities[np.arange(stop - start), closest], 0.0, 2.0,
            )
        return labels, distances

    def _single_run(self, features, rng):
        """Чередовать косинусное назначение и нормирование сумм векторов."""
        if self.init == "k-means++":
            centers, _ = kmeans_plusplus(features, self.n_clusters, random_state=rng)
        else:
            centers = features[rng.choice(len(features), self.n_clusters, replace=False)].copy()
        labels, distances = self._assign(features, centers)

        for iteration in range(1, self.max_iter + 1):
            sums = np.zeros_like(centers)
            np.add.at(sums, labels, features)
            updated = centers.copy()
            nonzero = np.any(sums != 0, axis=1)
            updated[nonzero] = _unit_rows(sums[nonzero])

            # При нулевой сумме любое единичное направление имеет ту же
            # стоимость. Пустые кластеры переносим к самым далёким объектам.
            empty = np.flatnonzero(np.bincount(labels, minlength=self.n_clusters) == 0)
            if len(empty):
                farthest = np.argsort(-distances, kind="stable")[:len(empty)]
                updated[empty] = features[farthest]

            shift = np.linalg.norm(updated - centers, axis=1).max()
            new_labels, distances = self._assign(features, updated)
            converged = np.array_equal(labels, new_labels) or shift <= self.tol
            centers, labels = updated, new_labels
            if converged:
                break

        return centers, labels, float(distances.sum()), iteration

    def fit(self, data):
        """Обучить модель и сохранить centers, labels и косинусную inertia_."""
        features = prepare_features(data)
        for name in ("n_clusters", "n_init", "max_iter", "batch_size"):
            positive_integer(name, getattr(self, name))
        positive_real("tol", self.tol, allow_zero=True)
        if self.random_state is not None:
            positive_integer("random_state", self.random_state, allow_zero=True)
            if self.random_state >= 2**32:
                raise ValueError("random_state должен быть меньше 4294967296.")
        if self.init not in ("k-means++", "random"):
            raise ValueError("init должен быть k-means++ или random.")

        normalized = _unit_rows(features)
        valid = np.any(normalized != 0, axis=1)
        training = normalized[valid]
        if len(training) < self.n_clusters:
            raise ValueError("n_clusters превышает количество ненулевых векторов.")

        rng = np.random.RandomState(self.random_state)
        best = None
        for _ in range(self.n_init):
            result = self._single_run(training, rng)
            if best is None or result[2] < best[2]:
                best = result

        centers, training_labels, inertia, iterations = best
        labels = np.full(len(features), -1, dtype=np.intp)
        labels[valid] = training_labels
        if len(np.unique(training_labels)) < self.n_clusters:
            warnings.warn(
                "Найдено меньше кластеров, чем n_clusters. Проверьте повторяющиеся направления.",
                ConvergenceWarning, stacklevel=2,
            )

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iterations
        self.n_features_in_ = features.shape[1]
        return self

    def fit_predict(self, data):
        """Обучить модель и вернуть целочисленные метки исходных объектов."""
        return self.fit(data).labels_

    def predict(self, data):
        """Назначить новые векторы ближайшим обученным направлениям."""
        check_is_fitted(self, "cluster_centers_")
        features = prepare_features(data)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("Число признаков должно совпадать с обучающей выборкой.")
        normalized = _unit_rows(features)
        labels, _ = self._assign(normalized, self.cluster_centers_)
        labels[~np.any(normalized != 0, axis=1)] = -1
        return labels


@StrategiesManager.registerStrategy(
    "spherical_kmeans", "Spherical K-Means (NumPy)",
    "Кластеризация по косинусному сходству с единичными центроидами",
)
class ConcreteStrategySphericalKMeans(Strategy):
    """Стратегия Spherical K-Means для точек, эмбеддингов и пикселей."""

    @classmethod
    def _setupParams(cls):
        cls._addParam("n_clusters", "Количество кластеров", StrategyParamType.UNumber,
                      "От 1 до количества ненулевых векторов.", 3)
        cls._addParam("init", "Инициализация центроидов", StrategyParamType.Switch,
                      "k-means++ или случайные ненулевые объекты.", "k-means++",
                      switches=["k-means++", "random"])
        cls._addParam("n_init", "Количество запусков", StrategyParamType.UNumber,
                      "Положительное число запусков. Выбирается минимум косинусной стоимости.", 10)
        cls._addParam("max_iter", "Максимум итераций", StrategyParamType.UNumber,
                      "Положительный предел итераций каждого запуска.", 300)
        cls._addParam("tol", "Допуск смещения центроидов", StrategyParamType.UFloating,
                      "Неотрицательный порог максимального L2-смещения единичного центроида.", 0.0001)
        cls._addParam("random_state", "Состояние случайности", StrategyParamType.UNumber,
                      "Seed локального генератора от 0 до 4294967295.", 42)
        cls._addParam("batch_size", "Размер блока расстояний", StrategyParamType.UNumber,
                      "Положительное число объектов в блоке назначения кластеров.", 4096)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """Принять (n_samples, n_features) и вернуть метку каждой строки."""
        model = SphericalKMeans(**{name: params[name] for name in self.params()})
        return model.fit_predict(points)

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """Кластеризовать плоский массив пикселей по направлению цвета."""
        return self.clastering_points(pixels, params)
