"""
WaveClustering Algorithm Implementation
Автор: Левинский Григорий [glevinskiy@gmail.com]
Последнее обновление: 2025-10-21
"""


import numpy as np
from typing import Dict, List, Tuple
import pywt
from scipy import ndimage

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


class WaveClustering:
    """
    Реализация алгоритма WaveClustering

    Parameters:
    -----------
    n_grid : int, default=32
        Количество делений сетки на каждое измерение
    wavelet : str, default='haar'
        Тип вейвлета ('haar', 'db4', 'coif1', и т.д.)
    n_levels : int, default=2
        Количество уровней вейвлет-преобразования (разрешение)
    density_threshold : float, default=0.1
        Порог для обнаружения плотных областей (относительно максимальной плотности)
    """

    def __init__(self, n_grid=32, wavelet='haar', n_levels=2, density_threshold=0.1):
        self.n_grid = n_grid
        self.wavelet = wavelet
        self.n_levels = n_levels
        self.density_threshold = density_threshold
        self.labels_ = None
        self.n_clusters_ = 0

    def _quantize_data(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Шаг 1: Квантование пространства признаков в ячейки сетки

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Входные данные

        Returns:
        --------
        grid : ndarray
            Сетка со значениями плотности
        metadata : dict
            Информация о границах сетки и размерах ячеек
        """
        n_samples, n_features = X.shape

        # Calculate grid bounds
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)

        # Calculate cell sizes
        cell_sizes = (max_vals - min_vals) / self.n_grid

        # Avoid division by zero
        cell_sizes = np.where(cell_sizes == 0, 1, cell_sizes)

        # Create grid based on dimensionality
        if n_features == 2:
            grid = np.zeros((self.n_grid, self.n_grid))
        elif n_features == 3:
            grid = np.zeros((self.n_grid, self.n_grid, self.n_grid))
        else:
            # For higher dimensions, use only first 2 features
            X = X[:, :2]
            n_features = 2
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            cell_sizes = (max_vals - min_vals) / self.n_grid
            cell_sizes = np.where(cell_sizes == 0, 1, cell_sizes)
            grid = np.zeros((self.n_grid, self.n_grid))

        # Assign points to grid cells
        for point in X:
            # Calculate grid indices
            indices = ((point - min_vals) / cell_sizes).astype(int)
            indices = np.clip(indices, 0, self.n_grid - 1)
            grid[tuple(indices)] += 1

        metadata = {
            'min_vals': min_vals,
            'max_vals': max_vals,
            'cell_sizes': cell_sizes,
            'n_features': n_features,
            'original_X': X  # Save modified X if needed
        }

        return grid, metadata

    def _apply_wavelet_transform(self, grid: np.ndarray, level: int = 1) -> np.ndarray:
        """
        Шаг 2: Применение вейвлет-преобразования к сетке

        Parameters:
        -----------
        grid : ndarray
            Квантованное пространство признаков
        level : int
            Количество уровней декомпозиции

        Returns:
        --------
        transformed_grid : ndarray
            Вейвлет-преобразованная сетка (LL поддиапазон)
        """
        if grid.ndim == 2:
            # 2D wavelet transform
            coeffs = pywt.wavedec2(grid, self.wavelet, level=level)
            approx = coeffs[0]
        elif grid.ndim == 3:
            # 3D wavelet transform (apply 1D transform along each axis)
            temp = grid
            for _ in range(level):
                temp = pywt.dwt(temp, self.wavelet, axis=0)[0]
                temp = pywt.dwt(temp, self.wavelet, axis=1)[0]
                temp = pywt.dwt(temp, self.wavelet, axis=2)[0]
            approx = temp
        else:
            raise ValueError(f"Unsupported grid dimensions: {grid.ndim}")

        return approx

    def _find_connected_components(self, grid: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
        """
        Шаг 3: Поиск связных компонент в преобразованном пространстве

        Parameters:
        -----------
        grid : ndarray
            Преобразованная сетка
        threshold : float
            Порог плотности

        Returns:
        --------
        labeled_grid : ndarray
            Сетка с метками кластеров
        n_clusters : int
            Количество найденных кластеров
        """
        # Threshold the grid to find dense regions
        dense_mask = grid > threshold

        # Find connected components
        labeled_grid, n_clusters = ndimage.label(dense_mask)

        return labeled_grid, n_clusters

    def _map_points_to_clusters(self, X: np.ndarray, labeled_grid: np.ndarray,
                                 metadata: Dict, scale_factor: int) -> np.ndarray:
        """
        Шаг 4: Сопоставление исходных точек с метками кластеров

        Parameters:
        -----------
        X : ndarray
            Исходные точки данных
        labeled_grid : ndarray
            Сетка с метками кластеров из преобразованного пространства
        metadata : dict
            Метаданные сетки
        scale_factor : int
            Масштабный коэффициент из-за понижающей дискретизации вейвлета

        Returns:
        --------
        labels : ndarray
            Метки кластеров для каждой точки
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        min_vals = metadata['min_vals']
        cell_sizes = metadata['cell_sizes']

        for i, point in enumerate(X):
            # Calculate grid indices in original space
            indices = ((point - min_vals) / cell_sizes).astype(int)
            indices = np.clip(indices, 0, self.n_grid - 1)

            # Map to transformed space
            transformed_indices = indices // scale_factor
            transformed_indices = np.clip(transformed_indices, 0,
                                         np.array(labeled_grid.shape) - 1)

            # Get cluster label
            labels[i] = labeled_grid[tuple(transformed_indices)]

        return labels

    def fit(self, X: np.ndarray) -> 'WaveClustering':
        """
        Выполнение кластеризации на X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Обучающие экземпляры для кластеризации

        Returns:
        --------
        self : object
            Возвращает сам экземпляр
        """
        X = np.asarray(X)

        # Step 1: Quantize data into grid
        grid, metadata = self._quantize_data(X)

        # Use modified X if available
        if 'original_X' in metadata:
            X = metadata['original_X']

        # Step 2: Apply wavelet transform
        transformed_grid = self._apply_wavelet_transform(grid, level=self.n_levels)

        # Step 3: Find connected components
        # Calculate threshold based on maximum density
        max_density = transformed_grid.max()

        # Handle case when max_density is 0
        if max_density == 0:
            self.labels_ = np.zeros(X.shape[0], dtype=int)
            self.n_clusters_ = 0
            return self

        threshold = max_density * self.density_threshold

        labeled_grid, self.n_clusters_ = self._find_connected_components(
            transformed_grid, threshold)

        # Step 4: Map points to clusters
        scale_factor = 2 ** self.n_levels  # Downsampling factor
        self.labels_ = self._map_points_to_clusters(
            X, labeled_grid, metadata, scale_factor)

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление кластеров и предсказание индекса кластера для каждого образца.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Образцы для кластеризации

        Returns:
        --------
        labels : ndarray, shape (n_samples,)
            Индекс кластера, к которому принадлежит каждый образец
        """
        return self.fit(X).labels_


@StrategiesManager.registerStrategy(
    "waveclustering",
    "WaveClustering",
    "Кластеризация на основе вейвлет-преобразования"
)
class ConcreteStrategyWaveClustering(Strategy):
    """
    WaveClustering strategy
    """

    @classmethod
    def _setupParams(cls):
        cls._addParam("n_grid", "Количество делений сетки", StrategyParamType.UNumber,
                     """
                     Количество делений каждого измерения на ячейки сетки.
                     Большие значения дают более детальное разбиение, но требуют больше памяти.
                     """,
                     32)

        cls._addParam("wavelet", "Тип вейвлета", StrategyParamType.Switch,
                     """
                     Тип вейвлет-функции для преобразования.
                     haar - самый простой и быстрый
                     db4 - Добеши 4-го порядка
                     coif1 - Койфлет 1-го порядка
                     """,
                     "haar",
                     switches=["haar", "db4", "db2", "db6", "coif1"])

        cls._addParam("n_levels", "Количество уровней", StrategyParamType.UNumber,
                     """
                     Количество уровней вейвлет-разложения (масштабы кластеров).
                     Большие значения дают более крупные кластеры.
                     """,
                     2)

        cls._addParam("density_threshold", "Порог плотности", StrategyParamType.UFloating,
                     """
                     Порог для определения плотных областей (доля от максимальной плотности).
                     Значения от 0 до 1. Меньшие значения дают больше кластеров.
                     """,
                     0.1)

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация изображения методом WaveClustering

        Parameters:
        -----------
        pixels : ndarray
            Массив пикселей (может быть в формате фреймворка)
        params : StrategyRunConfig
            Параметры алгоритма

        Returns:
        --------
        labels : ndarray
            Метки кластеров
        """
        # Проверка и коррекция формата данных
        pixels = np.asarray(pixels)

        # Если данные в формате (n_features, n_samples), транспонируем
        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            pixels = pixels.T

        model = WaveClustering(
            n_grid=int(params["n_grid"]),
            wavelet=params["wavelet"],
            n_levels=int(params["n_levels"]),
            density_threshold=float(params["density_threshold"])
        )
        return model.fit_predict(pixels)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом WaveClustering

        Parameters:
        -----------
        points : ndarray
            Массив точек (может быть в формате фреймворка)
        params : StrategyRunConfig
            Параметры алгоритма

        Returns:
        --------
        labels : ndarray
            Метки кластеров
        """
        # Проверка и коррекция формата данных
        points = np.asarray(points)

        # Если данные в формате (n_features, n_samples), транспонируем
        # Предполагаем, что n_features обычно меньше, чем n_samples
        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            points = points.T

        model = WaveClustering(
            n_grid=int(params["n_grid"]),
            wavelet=params["wavelet"],
            n_levels=int(params["n_levels"]),
            density_threshold=float(params["density_threshold"])
        )
        return model.fit_predict(points)
