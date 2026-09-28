"""
K-Medians Algorithm Implementation
Автор: Собиров Тельман Темурович [tel9master@mail.ru]
Последнее обновление: 2026-09-26

Кластеризация методом K-Medians на базе pyclustering.cluster.kmedians.

Алгоритм похож на KMeans: точки относятся к ближайшему центру, затем
центры пересчитываются. Отличие в том, что центр кластера считается
не как среднее, а как медиана по каждой координате. Медиана не
"утягивается" отдельными далёкими точками, поэтому метод устойчивее
к выбросам. Медиана минимизирует сумму манхэттенских расстояний,
поэтому метрикой по умолчанию выбрана manhattan.

Ограничения:
- Начальные центры выбираются через sklearn.cluster.kmeans_plusplus:
  pyclustering.cluster.center_initializer не работает с numpy >= 2
  (обращается к удалённому numpy.warnings).
- C++ часть pyclustering (ccore) собрана не для всех платформ
  (например, отсутствует для macOS arm64). Если её не удаётся
  загрузить, используется реализация на Python.

Источник:
https://pyclustering.github.io/docs/0.10.1/html/d0/d7a/classpyclustering_1_1cluster_1_1kmedians_1_1kmedians.html
"""

import numpy as np
from pyclustering.cluster.kmedians import kmedians
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


_METRICS = {
    "manhattan": type_metric.MANHATTAN,
    "euclidean": type_metric.EUCLIDEAN,
    "euclidean_square": type_metric.EUCLIDEAN_SQUARE,
    "chebyshev": type_metric.CHEBYSHEV,
}


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


def _initial_medians(features: np.ndarray, n_clusters: int, init: str, random_state: int) -> np.ndarray:
    """
    Выбирает начальные центры кластеров.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Подготовленные признаки.
    n_clusters : int
        Число кластеров (не больше числа точек).
    init : str
        'k-means++' — центры выбираются далеко друг от друга;
        'random' — случайные точки выборки.
    random_state : int
        Инициализация генератора случайных чисел.

    Returns:
    --------
    ndarray, shape (n_clusters, n_features)
        Начальные центры.
    """
    if init == "random":
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(features), size=n_clusters, replace=False)
        return features[indices]

    centers, _ = kmeans_plusplus(features, n_clusters, random_state=random_state)
    return centers


def _labels_from_clusters(clusters, n_samples: int) -> np.ndarray:
    """
    Преобразует список кластеров pyclustering в массив меток.

    Parameters:
    -----------
    clusters : list of list of int
        Индексы точек каждого кластера.
    n_samples : int
        Общее число точек.

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метка кластера для каждой точки.
    """
    labels = np.zeros(n_samples, dtype=np.intp)
    for cluster_index, cluster in enumerate(clusters):
        labels[cluster] = cluster_index
    return labels


def _fit_predict(features: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
    """
    Запускает K-Medians и возвращает метки.

    Сначала пробует C++ реализацию (если включена), при ошибке её
    загрузки переключается на Python-реализацию.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Подготовленные признаки.
    params : StrategyRunConfig
        Параметры стратегии.

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метка кластера для каждой точки.
    """
    n_samples = len(features)
    if n_samples == 0:
        return np.array([], dtype=np.intp)

    n_clusters = max(1, min(int(params["n_clusters"]), n_samples))
    initial = _initial_medians(
        features,
        n_clusters,
        str(params["init"]),
        int(params["random_state"])
    )
    metric = distance_metric(_METRICS.get(str(params["metric"]), type_metric.MANHATTAN))

    data = features.tolist()
    kwargs = {
        "tolerance": float(params["tolerance"]),
        "metric": metric,
        "itermax": max(1, int(params["itermax"])),
    }

    try:
        model = kmedians(data, initial.tolist(), ccore=bool(params["ccore"]), **kwargs)
        model.process()
    except OSError:
        model = kmedians(data, initial.tolist(), ccore=False, **kwargs)
        model.process()

    return _labels_from_clusters(model.get_clusters(), n_samples)


@StrategiesManager.registerStrategy(
    "kmedians_pyc",
    "K-Medians (PyClustering)",
    "Кластеризация K-Medians из pyclustering"
)
class ConcreteStrategyKMedians_from_PYCLUSTERING(Strategy):
    """
    Стратегия кластеризации K-Medians из pyclustering.

    Parameters:
    -----------
    n_clusters : int, default=3
        Количество кластеров.

    init : {'k-means++', 'random'}, default='k-means++'
        Способ выбора начальных центров.

    metric : {'manhattan', 'euclidean', 'euclidean_square', 'chebyshev'}, default='manhattan'
        Метрика расстояния между точкой и центром.

    tolerance : float, default=0.001
        Алгоритм останавливается, когда центры сдвигаются меньше этого значения.

    itermax : int, default=200
        Максимальное число итераций.

    random_state : int, default=42
        Инициализация генератора случайных чисел.

    normalize : bool, default=True
        Стандартизировать признаки перед кластеризацией.

    ccore : bool, default=True
        Использовать C++ часть pyclustering (с откатом на Python).
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
            Примеры: 2, 3, 4, 5...
            """,
            3
        )

        cls._addParam(
            "init",
            "Метод инициализации",
            StrategyParamType.Switch,
            """
            Способ выбора начальных центров:

            - k-means++: центры выбираются далеко друг от друга (рекомендуется)
            - random: случайные точки выборки
            """,
            "k-means++",
            switches=["k-means++", "random"]
        )

        cls._addParam(
            "metric",
            "Метрика расстояния",
            StrategyParamType.Switch,
            """
            Метрика для расчёта расстояния от точки до центра кластера.

            - manhattan: сумма модулей разностей (рекомендуется для медиан)
            - euclidean: евклидово расстояние
            - euclidean_square: квадрат евклидова расстояния
            - chebyshev: максимальная норма
            """,
            "manhattan",
            switches=["manhattan", "euclidean", "euclidean_square", "chebyshev"]
        )

        cls._addParam(
            "tolerance",
            "Точность остановки",
            StrategyParamType.UFloating,
            """
            Алгоритм останавливается, когда максимальный сдвиг центров
            за итерацию меньше этого значения.
            """,
            0.001
        )

        cls._addParam(
            "itermax",
            "Максимум итераций",
            StrategyParamType.UNumber,
            """
            Максимальное количество итераций алгоритма.
            """,
            200
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
            "normalize",
            "Нормализовать данные",
            StrategyParamType.Bool,
            """
            Стандартизировать признаки перед кластеризацией.
            Рекомендуется оставить включённым для признаков разного масштаба.
            """,
            True
        )

        cls._addParam(
            "ccore",
            "Использовать C++",
            StrategyParamType.Bool,
            """
            Использовать C++ часть библиотеки pyclustering для ускорения.
            Если она недоступна на текущей платформе, автоматически
            используется реализация на Python.
            """,
            True
        )

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация изображения методом K-Medians.

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
        return _fit_predict(pixels_proc, params)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом K-Medians.

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
