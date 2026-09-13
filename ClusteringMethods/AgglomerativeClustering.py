"""
AgglomerativeClustering Algorithm Implementation
Последнее обновление: 2026-09-13

Иерархическая агломеративная кластеризация на базе
sklearn.cluster.AgglomerativeClustering.

Алгоритм начинает с того, что каждая точка — отдельный кластер,
затем последовательно объединяет ближайшие кластеры. Способ слияния
задаётся параметром linkage (ward, complete, average, single).
Число кластеров можно задать явно через n_clusters либо определить
автоматически по порогу расстояния distance_threshold.

Ограничения:
- linkage='ward' работает только с евклидовой метрикой; при другом
  выборе метрика автоматически заменяется на euclidean.
- Для изображений с числом пикселей больше _IMAGE_MAX_SAMPLES
  выполняется выборка, а оставшиеся точки получают метку ближайшего
  центроида построенных кластеров.

Источник:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
"""

import inspect

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


_WARD_METRICS = {"euclidean", "l2"}
_IMAGE_MAX_SAMPLES = 3000


def _to_compute_full_tree(value) -> bool | str:
    """
    Приводит значение compute_full_tree к типу, который принимает sklearn.

    Parameters:
    -----------
    value : bool | str
        Значение из GUI: True/False либо строки 'auto', 'true', 'false'.

    Returns:
    --------
    bool | str
        True, False или 'auto'.
    """
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return "auto"


def _create_agglomerative_model(
    n_clusters: int,
    n_samples: int,
    linkage: str,
    metric: str,
    compute_full_tree,
    distance_threshold: float,
    compute_distances: bool
) -> AgglomerativeClustering:
    """
    Собирает модель sklearn.cluster.AgglomerativeClustering.

    Учитывает ограничения linkage/metric, порог расстояния и различия
    версий sklearn (параметр metric либо устаревший affinity).

    Parameters:
    -----------
    n_clusters : int
        Желаемое число кластеров. Игнорируется, если distance_threshold > 0.
    n_samples : int
        Число объектов в выборке. Нужно, чтобы n_clusters не превышал его.
    linkage : str
        Метод объединения: 'ward', 'complete', 'average', 'single'.
    metric : str
        Метрика расстояния. Для ward принудительно используется euclidean.
    compute_full_tree : bool | str
        Строить ли полное дерево слияния: 'auto', True или False.
    distance_threshold : float
        Порог остановки слияния. Значение 0 означает использование n_clusters.
    compute_distances : bool
        Сохранять ли расстояния между узлами дерева.

    Returns:
    --------
    sklearn.cluster.AgglomerativeClustering
        Необученная модель, готовая к fit_predict.
    """
    linkage = str(linkage)
    metric = str(metric)

    if linkage == "ward" and metric not in _WARD_METRICS:
        metric = "euclidean"

    use_threshold = distance_threshold is not None and float(distance_threshold) > 0.0
    if use_threshold:
        n_clusters_arg = None
        distance_threshold_arg = float(distance_threshold)
        compute_full_tree_arg = True
    else:
        n_clusters_arg = int(n_clusters) if n_clusters else 2
        n_clusters_arg = max(1, min(n_clusters_arg, max(1, n_samples)))
        distance_threshold_arg = None
        compute_full_tree_arg = _to_compute_full_tree(compute_full_tree)

    kwargs = {
        "n_clusters": n_clusters_arg,
        "linkage": linkage,
        "compute_full_tree": compute_full_tree_arg,
        "distance_threshold": distance_threshold_arg,
        "compute_distances": bool(compute_distances),
    }

    signature = inspect.signature(AgglomerativeClustering.__init__).parameters
    if "metric" in signature:
        kwargs["metric"] = metric
    else:
        kwargs["affinity"] = metric

    return AgglomerativeClustering(**kwargs)


def _prepare_features(data: np.ndarray) -> np.ndarray:
    """
    Приводит входные данные к массиву (n_samples, n_features).

    Parameters:
    -----------
    data : array-like
        Точки или пиксели. Допускается одномерный вектор
        или транспонированный вид (n_features, n_samples),
        если признаков не больше 10.

    Returns:
    --------
    ndarray, shape (n_samples, n_features)
        Массив признаков типа float64.
    """
    features = np.asarray(data, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[0] < features.shape[1] and features.shape[0] <= 10:
        features = features.T
    return features


def _maybe_normalize(features: np.ndarray, normalize: bool) -> np.ndarray:
    """
    Стандартизирует признаки, если включена нормализация.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Исходные признаки.
    normalize : bool
        Если True, применяется StandardScaler.

    Returns:
    --------
    ndarray, shape (n_samples, n_features)
        Нормализованные или исходные признаки.
    """
    if not normalize:
        return features
    return StandardScaler().fit_transform(features)


def _assign_remaining_by_centroids(
    features: np.ndarray,
    sample_indices: np.ndarray,
    sample_labels: np.ndarray
) -> np.ndarray:
    """
    Проставляет метки невыбранным точкам по ближайшему центроиду.

    Используется при кластеризации изображений, когда полная выборка
    слишком велика для агломеративного алгоритма.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Все точки в том же признаковом пространстве, что и выборка.
    sample_indices : ndarray
        Индексы точек, по которым строились кластеры.
    sample_labels : ndarray
        Метки кластеров для выбранных точек.

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метки для всех точек.
    """
    labels = np.empty(len(features), dtype=np.intp)
    labels[sample_indices] = sample_labels

    unique_labels = np.unique(sample_labels)
    sample_features = features[sample_indices]
    centroids = np.vstack([
        sample_features[sample_labels == label].mean(axis=0)
        for label in unique_labels
    ])

    remaining_mask = np.ones(len(features), dtype=bool)
    remaining_mask[sample_indices] = False
    remaining = features[remaining_mask]
    if len(remaining) == 0:
        return labels

    distances = np.linalg.norm(remaining[:, None, :] - centroids[None, :, :], axis=2)
    labels[remaining_mask] = unique_labels[np.argmin(distances, axis=1)]
    return labels


def _fit_predict(features: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
    """
    Запускает AgglomerativeClustering и возвращает метки.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Подготовленные признаки.
    params : StrategyRunConfig
        Параметры стратегии (n_clusters, linkage, metric и др.).

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метка кластера для каждой точки.
    """
    if len(features) == 0:
        return np.array([], dtype=np.intp)
    if len(features) == 1:
        return np.array([0], dtype=np.intp)

    model = _create_agglomerative_model(
        n_clusters=int(params["n_clusters"]),
        n_samples=len(features),
        linkage=params["linkage"],
        metric=params["metric"],
        compute_full_tree=params["compute_full_tree"],
        distance_threshold=float(params["distance_threshold"]),
        compute_distances=bool(params["compute_distances"]),
    )
    return model.fit_predict(features)


@StrategiesManager.registerStrategy(
    "agglomerative_sk",
    "Agglomerative Clustering (SKLearn)",
    "Иерархическая агломеративная кластеризация из sklearn"
)
class ConcreteStrategyAgglomerative_from_SKLEARN(Strategy):
    """
    Стратегия иерархической агломеративной кластеризации из sklearn.

    Parameters:
    -----------
    n_clusters : int, default=3
        Желаемое количество кластеров. Игнорируется, если
        distance_threshold больше 0.

    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Способ объединения кластеров:
        - ward: минимизирует дисперсию (рекомендуется, только euclidean);
        - complete: максимальное расстояние между кластерами;
        - average: среднее расстояние между кластерами;
        - single: минимальное расстояние между кластерами.

    metric : str, default='euclidean'
        Метрика расстояния: euclidean, manhattan, cosine, l1, l2, chebyshev.
        Для linkage='ward' используется только euclidean.

    compute_full_tree : {'auto', 'true', 'false'}, default='auto'
        Строить ли полное дерево слияния. При distance_threshold > 0
        всегда строится полное дерево.

    distance_threshold : float, default=0.0
        Порог расстояния для остановки слияния. 0 означает,
        что используется n_clusters.

    compute_distances : bool, default=False
        Сохранять расстояния между узлами дерева. На метки не влияет.

    normalize : bool, default=True
        Стандартизировать признаки перед кластеризацией.
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров, отображаемых в GUI."""
        cls._addParam(
            "n_clusters",
            "Количество кластеров",
            StrategyParamType.UNumber,
            """
            Желаемое количество кластеров.
            Игнорируется, если задан distance_threshold больше 0.
            Примеры: 2, 3, 4, 5...
            """,
            3
        )

        cls._addParam(
            "linkage",
            "Метод linkage",
            StrategyParamType.Switch,
            """
            Способ объединения кластеров в иерархии:

            - ward: минимизирует дисперсию (рекомендуется, только euclidean)
            - complete: максимальное расстояние между кластерами
            - average: среднее расстояние между кластерами
            - single: минимальное расстояние между кластерами
            """,
            "ward",
            switches=["ward", "complete", "average", "single"]
        )

        cls._addParam(
            "metric",
            "Метрика расстояния",
            StrategyParamType.Switch,
            """
            Метрика для расчёта расстояния между объектами.
            Для linkage=ward автоматически используется euclidean.

            - euclidean: евклидово расстояние (рекомендуется)
            - manhattan: манхэттенское расстояние
            - cosine: косинусное расстояние
            - l1: то же, что manhattan
            - l2: то же, что euclidean
            - chebyshev: максимальная норма
            """,
            "euclidean",
            switches=["euclidean", "manhattan", "cosine", "l1", "l2", "chebyshev"]
        )

        cls._addParam(
            "compute_full_tree",
            "Строить полное дерево",
            StrategyParamType.Switch,
            """
            Останавливать ли построение дерева после получения n_clusters.

            - auto: полное дерево строится при небольшом числе кластеров
            - true: всегда строить полное дерево
            - false: останавливаться после получения n_clusters
            При использовании distance_threshold всегда строится полное дерево.
            """,
            "auto",
            switches=["auto", "true", "false"]
        )

        cls._addParam(
            "distance_threshold",
            "Порог расстояния",
            StrategyParamType.UFloating,
            """
            Порог расстояния для остановки слияния кластеров.
            Если значение больше 0, параметр n_clusters игнорируется,
            а число кластеров определяется автоматически.
            0 означает, что используется n_clusters.
            """,
            0.0
        )

        cls._addParam(
            "compute_distances",
            "Вычислять расстояния",
            StrategyParamType.Bool,
            """
            Вычислять и сохранять расстояния между узлами дерева слияния.
            Нужно только для анализа дендрограммы, на метки кластеров не влияет.
            """,
            False
        )

        cls._addParam(
            "normalize",
            "Нормализовать данные",
            StrategyParamType.Bool,
            """
            Стандартизировать признаки перед кластеризацией.
            Рекомендуется оставить включённым для признаков разного масштаба.
            """,
            True
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация изображения методом AgglomerativeClustering.

        При числе пикселей больше 3000 строится выборка, затем оставшиеся
        пиксели получают метку ближайшего центроида.

        Parameters:
        -----------
        pixels : ndarray
            Пиксели изображения в виде массива признаков.
        params : StrategyRunConfig
            Параметры запуска стратегии.

        Returns:
        --------
        ndarray
            Метка кластера для каждого пикселя.
        """
        pixels = _prepare_features(pixels)
        pixels_proc = _maybe_normalize(pixels, bool(params["normalize"]))

        if len(pixels_proc) > _IMAGE_MAX_SAMPLES:
            step = max(1, int(np.sqrt(len(pixels_proc) / _IMAGE_MAX_SAMPLES)))
            indices = np.arange(0, len(pixels_proc), step)[:_IMAGE_MAX_SAMPLES]
            sample_labels = _fit_predict(pixels_proc[indices], params)
            return _assign_remaining_by_centroids(pixels_proc, indices, sample_labels)

        return _fit_predict(pixels_proc, params)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом AgglomerativeClustering.

        Parameters:
        -----------
        points : ndarray
            Точки данных в виде массива (n_samples, n_features).
        params : StrategyRunConfig
            Параметры запуска стратегии.

        Returns:
        --------
        ndarray
            Метка кластера для каждой точки.
        """
        points = _prepare_features(points)
        points_proc = _maybe_normalize(points, bool(params["normalize"]))
        return _fit_predict(points_proc, params)
