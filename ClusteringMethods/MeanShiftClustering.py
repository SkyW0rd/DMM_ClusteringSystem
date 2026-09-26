"""
MeanShift Algorithm Implementation
Последнее обновление: 2026-09-26

Кластеризация методом сдвига среднего (Mean Shift) на базе
sklearn.cluster.MeanShift.

Идея алгоритма: вокруг каждой точки строится "окно" радиуса bandwidth,
затем точка сдвигается в среднее (центр масс) точек, попавших в окно.
Сдвиг повторяется, пока точка не остановится. Так все точки "сползают"
к местам наибольшей плотности данных (модам). Точки, пришедшие в одну
моду, образуют один кластер. Число кластеров определяется автоматически.

Ограничения:
- Сложность алгоритма высокая, поэтому для изображений с числом
  пикселей больше _IMAGE_MAX_SAMPLES модель обучается на выборке,
  а остальные пиксели получают метку ближайшего найденного центра.
- При cluster_all=False точки, не попавшие ни в одно окно, получают
  метку -1 (шум).

Источник:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
"""

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


_IMAGE_MAX_SAMPLES = 3000
_BANDWIDTH_MAX_SAMPLES = 1000


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


def _resolve_bandwidth(features: np.ndarray, bandwidth: float, quantile: float) -> float:
    """
    Определяет радиус окна (bandwidth) для MeanShift.

    Если bandwidth задан (> 0), возвращается как есть. Иначе оценивается
    через sklearn.cluster.estimate_bandwidth по квантилю попарных расстояний.
    Если оценка дала 0 (например, много одинаковых точек), используется
    запасное значение на основе разброса данных.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Подготовленные признаки.
    bandwidth : float
        Радиус окна из GUI. 0 означает автоматическую оценку.
    quantile : float
        Квантиль для estimate_bandwidth, от 0 до 1.

    Returns:
    --------
    float
        Положительный радиус окна.
    """
    if bandwidth > 0.0:
        return float(bandwidth)

    quantile = min(max(float(quantile), 0.01), 1.0)
    estimated = estimate_bandwidth(
        features,
        quantile=quantile,
        n_samples=min(len(features), _BANDWIDTH_MAX_SAMPLES),
        random_state=42
    )
    if estimated > 0.0:
        return float(estimated)

    spread = float(np.max(np.std(features, axis=0)))
    return spread if spread > 0.0 else 1.0


def _create_model(features: np.ndarray, params: StrategyRunConfig) -> MeanShift:
    """
    Собирает модель sklearn.cluster.MeanShift по параметрам из GUI.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Подготовленные признаки (нужны для оценки bandwidth).
    params : StrategyRunConfig
        Параметры стратегии.

    Returns:
    --------
    sklearn.cluster.MeanShift
        Необученная модель.
    """
    bandwidth = _resolve_bandwidth(
        features,
        float(params["bandwidth"]),
        float(params["quantile"])
    )
    return MeanShift(
        bandwidth=bandwidth,
        bin_seeding=bool(params["bin_seeding"]),
        min_bin_freq=max(1, int(params["min_bin_freq"])),
        cluster_all=bool(params["cluster_all"]),
        max_iter=max(1, int(params["max_iter"])),
    )


def _fit_predict(features: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
    """
    Запускает MeanShift и возвращает метки.

    Parameters:
    -----------
    features : ndarray, shape (n_samples, n_features)
        Подготовленные признаки.
    params : StrategyRunConfig
        Параметры стратегии.

    Returns:
    --------
    ndarray, shape (n_samples,)
        Метка кластера для каждой точки (-1 — шум при cluster_all=False).
    """
    if len(features) == 0:
        return np.array([], dtype=np.intp)
    if len(features) == 1:
        return np.array([0], dtype=np.intp)

    return _create_model(features, params).fit_predict(features)


@StrategiesManager.registerStrategy(
    "meanshift_sk",
    "MeanShift (SKLearn)",
    "Кластеризация методом сдвига среднего из sklearn"
)
class ConcreteStrategyMeanShift_from_SKLEARN(Strategy):
    """
    Стратегия кластеризации MeanShift из sklearn.

    Parameters:
    -----------
    bandwidth : float, default=0.0
        Радиус окна. 0 означает автоматическую оценку по quantile.

    quantile : float, default=0.2
        Квантиль попарных расстояний для автоматической оценки bandwidth.
        Меньше значение — меньше окно — больше кластеров.

    bin_seeding : bool, default=False
        Начинать сдвиги не из каждой точки, а из узлов сетки.
        Сильно ускоряет работу на больших данных.

    min_bin_freq : int, default=1
        Минимальное число точек в ячейке сетки, чтобы она стала
        стартовой. Используется только при bin_seeding=True.

    cluster_all : bool, default=True
        Если False, точки вне всех окон получают метку -1 (шум).

    max_iter : int, default=300
        Максимальное число сдвигов для одной стартовой точки.

    normalize : bool, default=True
        Стандартизировать признаки перед кластеризацией.
    """

    @classmethod
    def _setupParams(cls):
        """Инициализация параметров, отображаемых в GUI."""
        cls._addParam(
            "bandwidth",
            "Радиус окна (bandwidth)",
            StrategyParamType.UFloating,
            """
            Радиус окна, в котором считается среднее при сдвиге точки.
            Меньше радиус — больше мелких кластеров, больше радиус — меньше
            крупных кластеров. 0 означает автоматическую оценку по квантилю.
            """,
            0.0
        )

        cls._addParam(
            "quantile",
            "Квантиль для оценки радиуса",
            StrategyParamType.UFloating,
            """
            Используется, если bandwidth равен 0. Радиус оценивается как
            среднее расстояние до соседей, попадающих в данную долю выборки.
            Допустимо от 0 до 1. Рекомендуется: 0.1-0.3.
            """,
            0.2
        )

        cls._addParam(
            "bin_seeding",
            "Стартовые точки по сетке",
            StrategyParamType.Bool,
            """
            Запускать сдвиги не из каждой точки, а из узлов сетки
            с шагом bandwidth. Сильно ускоряет работу на больших данных.
            """,
            False
        )

        cls._addParam(
            "min_bin_freq",
            "Минимум точек в ячейке",
            StrategyParamType.UNumber,
            """
            Минимальное число точек в ячейке сетки, чтобы она стала
            стартовой. Используется только при включённой сетке.
            """,
            1
        )

        cls._addParam(
            "cluster_all",
            "Кластеризовать все точки",
            StrategyParamType.Bool,
            """
            Если включено, каждая точка относится к ближайшему кластеру.
            Если выключено, точки вне всех окон получают метку -1 (шум).
            """,
            True
        )

        cls._addParam(
            "max_iter",
            "Максимум итераций",
            StrategyParamType.UNumber,
            """
            Максимальное число сдвигов для одной стартовой точки.
            Рекомендуется: 300.
            """,
            300
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
        Кластеризация изображения методом MeanShift.

        При числе пикселей больше _IMAGE_MAX_SAMPLES модель обучается
        на равномерной выборке, затем все пиксели получают метку
        ближайшего найденного центра (MeanShift.predict).

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
            step = max(1, len(pixels_proc) // _IMAGE_MAX_SAMPLES)
            sample = pixels_proc[::step][:_IMAGE_MAX_SAMPLES]
            model = _create_model(sample, params)
            model.fit(sample)
            return model.predict(pixels_proc)

        return _fit_predict(pixels_proc, params)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом MeanShift.

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
