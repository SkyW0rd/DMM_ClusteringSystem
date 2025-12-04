"""
DUCStream Clustering Algorithm Implementation
Алгоритм кластеризации потоковых данных с неопределенностью и вейвлет-декомпозицией
Основан на DenStream с добавлением вейвлет-преобразования
Автор: Мысин Юрий [yuriymysin@yandex.ru]
Последнее обновление: 2025-12-04
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pywt  # PyWavelets library for wavelet transforms
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


class DUCtreamClustering:
    """
    Реализация алгоритма DUCtream (DenStream-based Uncertain Clustering Stream)
    с вейвлет-декомпозицией для обработки потоковых данных с неопределенностью
    
    Parameters:
    -----------
    num_clusters : int, default=5
        Ожидаемое количество кластеров
    lambda_param : float, default=0.25
        Параметр затухания для потоковых данных
    epsilon : float, default=0.1
        Радиус окрестности для микрокластеров
    beta : float, default=0.2
        Порог для определения outlier микрокластеров
    mu : int, default=2
        Минимальное количество точек в микрокластере
    wavelet : str, default='haar'
        Тип вейвлета для декомпозиции
    n_levels : int, default=2
        Количество уровней вейвлет-декомпозиции
    normalize : bool, default=True
        Нормализация данных перед обработкой
    """
    
    def __init__(self, num_clusters=5, lambda_param=0.25, epsilon=0.1, 
                 beta=0.2, mu=2, wavelet='haar', n_levels=2, normalize=True):
        self.num_clusters = num_clusters
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.wavelet = wavelet
        self.n_levels = n_levels
        self.normalize = normalize
        
        # Результаты кластеризации
        self.labels_ = None
        self.cluster_centers_ = None
        self.X_ = None
        self.X_normalized_ = None
        self.scaler_ = None
        self.microclusters_ = []
        self.wavelet_coeffs_ = None
        
    def _apply_wavelet_decomposition(self, X: np.ndarray) -> np.ndarray:
        """
        Применение вейвлет-декомпозиции к данным для снижения размерности
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Входные данные
            
        Returns:
        --------
        X_decomposed : ndarray
            Данные после вейвлет-декомпозиции
        """
        n_samples, n_features = X.shape
        
        # Применяем вейвлет-преобразование к каждой размерности отдельно
        decomposed_features = []
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            
            # Применяем 1D вейвлет-преобразование
            coeffs = pywt.wavedec(feature_data, self.wavelet, level=self.n_levels)
            
            # Используем только аппроксимационные коэффициенты (низкочастотная часть)
            approx_coeffs = coeffs[0]
            
            # Если коэффициентов меньше, чем образцов, интерполируем
            if len(approx_coeffs) < n_samples:
                # Повторяем последний коэффициент для заполнения
                repeat_factor = n_samples // len(approx_coeffs) + 1
                approx_coeffs = np.tile(approx_coeffs, repeat_factor)[:n_samples]
            elif len(approx_coeffs) > n_samples:
                # Обрезаем до нужного размера
                approx_coeffs = approx_coeffs[:n_samples]
            
            decomposed_features.append(approx_coeffs)
        
        # Объединяем все размерности
        X_decomposed = np.column_stack(decomposed_features)
        
        return X_decomposed
    
    def _create_microcluster(self, center: np.ndarray, weight: float, 
                            timestamp: int) -> Dict:
        """
        Создание микрокластера
        
        Parameters:
        -----------
        center : ndarray
            Центр микрокластера
        weight : float
            Вес микрокластера
        timestamp : int
            Временная метка создания
            
        Returns:
        --------
        microcluster : dict
            Словарь с параметрами микрокластера
        """
        return {
            'center': center.copy(),
            'weight': weight,
            'timestamp': timestamp,
            'points_count': 1
        }
    
    def _update_microcluster(self, microcluster: Dict, point: np.ndarray, 
                            weight: float, timestamp: int):
        """
        Обновление микрокластера новым пикселем
        
        Parameters:
        -----------
        microcluster : dict
            Микрокластер для обновления
        point : ndarray
            Новая точка
        weight : float
            Вес новой точки
        timestamp : int
            Текущая временная метка
        """
        # Затухание веса
        time_diff = timestamp - microcluster['timestamp']
        microcluster['weight'] *= (self.lambda_param ** time_diff)
        
        # Обновление центра (взвешенное среднее)
        total_weight = microcluster['weight'] + weight
        if total_weight > 0:
            microcluster['center'] = (
                microcluster['center'] * microcluster['weight'] + 
                point * weight
            ) / total_weight
        
        microcluster['weight'] += weight
        microcluster['timestamp'] = timestamp
        microcluster['points_count'] += 1
    
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Вычисление евклидова расстояния"""
        return np.linalg.norm(x - y)
    
    def _find_nearest_microcluster(self, point: np.ndarray, 
                                  microclusters: List[Dict]) -> Optional[int]:
        """
        Поиск ближайшего микрокластера к точке
        
        Parameters:
        -----------
        point : ndarray
            Точка данных
        microclusters : List[Dict]
            Список микрокластеров
            
        Returns:
        --------
        index : int or None
            Индекс ближайшего микрокластера или None, если расстояние > epsilon
        """
        min_dist = float('inf')
        nearest_idx = None
        
        for idx, mc in enumerate(microclusters):
            dist = self._distance(point, mc['center'])
            if dist < self.epsilon and dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        return nearest_idx
    
    def _remove_weak_microclusters(self, microclusters: List[Dict], 
                                   timestamp: int):
        """
        Удаление слабых микрокластеров (outlier микрокластеры)
        
        Parameters:
        -----------
        microclusters : List[Dict]
            Список микрокластеров
        timestamp : int
            Текущая временная метка
        """
        to_remove = []
        
        for idx, mc in enumerate(microclusters):
            time_diff = timestamp - mc['timestamp']
            decayed_weight = mc['weight'] * (self.lambda_param ** time_diff)
            
            # Удаляем микрокластеры с весом меньше beta * mu
            if decayed_weight < self.beta * self.mu:
                to_remove.append(idx)
        
        # Удаляем в обратном порядке, чтобы не нарушить индексы
        for idx in reversed(to_remove):
            microclusters.pop(idx)
    
    def _merge_microclusters(self, microclusters: List[Dict]) -> List[Dict]:
        """
        Объединение близких микрокластеров
        
        Parameters:
        -----------
        microclusters : List[Dict]
            Список микрокластеров
            
        Returns:
        --------
        merged : List[Dict]
            Список объединенных микрокластеров
        """
        if len(microclusters) == 0:
            return []
        
        merged = []
        used = set()
        
        for i, mc1 in enumerate(microclusters):
            if i in used:
                continue
            
            cluster = mc1.copy()
            used.add(i)
            
            # Ищем близкие микрокластеры для объединения
            for j, mc2 in enumerate(microclusters[i+1:], start=i+1):
                if j in used:
                    continue
                
                dist = self._distance(mc1['center'], mc2['center'])
                if dist < self.epsilon:
                    # Объединяем микрокластеры
                    total_weight = cluster['weight'] + mc2['weight']
                    if total_weight > 0:
                        cluster['center'] = (
                            cluster['center'] * cluster['weight'] +
                            mc2['center'] * mc2['weight']
                        ) / total_weight
                    
                    cluster['weight'] += mc2['weight']
                    cluster['points_count'] += mc2['points_count']
                    cluster['timestamp'] = max(cluster['timestamp'], mc2['timestamp'])
                    used.add(j)
            
            merged.append(cluster)
        
        return merged
    
    def _cluster_microclusters(self, microclusters: List[Dict]) -> np.ndarray:
        """
        Кластеризация микрокластеров в финальные кластеры
        
        Parameters:
        -----------
        microclusters : List[Dict]
            Список микрокластеров
            
        Returns:
        --------
        labels : ndarray
            Метки кластеров для каждой точки
        """
        if len(microclusters) == 0:
            return np.zeros(self.X_.shape[0], dtype=int)
        
        # Фильтруем только значимые микрокластеры (вес >= mu)
        significant_mcs = [mc for mc in microclusters if mc['weight'] >= self.mu]
        
        if len(significant_mcs) == 0:
            return np.zeros(self.X_.shape[0], dtype=int)
        
        # Если количество значимых микрокластеров <= num_clusters, 
        # используем их как финальные кластеры
        if len(significant_mcs) <= self.num_clusters:
            # Создаем центры кластеров из микрокластеров
            centers = np.array([mc['center'] for mc in significant_mcs])
        else:
            # Используем K-means для группировки микрокластеров
            mc_centers = np.array([mc['center'] for mc in significant_mcs])
            mc_weights = np.array([mc['weight'] for mc in significant_mcs])
            
            # Инициализация центров K-means++
            centers = np.zeros((self.num_clusters, mc_centers.shape[1]))
            centers[0] = mc_centers[np.random.choice(len(mc_centers), p=mc_weights/mc_weights.sum())]
            
            for i in range(1, self.num_clusters):
                distances = np.array([
                    min([np.linalg.norm(mc_centers[j] - centers[k]) 
                         for k in range(i)]) 
                    for j in range(len(mc_centers))
                ])
                probabilities = (distances ** 2) * mc_weights
                probabilities /= probabilities.sum()
                centers[i] = mc_centers[np.random.choice(len(mc_centers), p=probabilities)]
            
            # Итерации K-means
            for _ in range(50):
                # Назначение микрокластеров к центрам
                assignments = np.array([
                    np.argmin([np.linalg.norm(mc_centers[i] - centers[j]) 
                               for j in range(self.num_clusters)])
                    for i in range(len(mc_centers))
                ])
                
                # Обновление центров
                new_centers = np.zeros_like(centers)
                for k in range(self.num_clusters):
                    mask = assignments == k
                    if np.any(mask):
                        weighted_sum = np.sum(mc_centers[mask] * mc_weights[mask, np.newaxis], axis=0)
                        total_weight = np.sum(mc_weights[mask])
                        if total_weight > 0:
                            new_centers[k] = weighted_sum / total_weight
                        else:
                            new_centers[k] = centers[k]
                    else:
                        new_centers[k] = centers[k]
                
                if np.allclose(centers, new_centers, rtol=1e-4):
                    break
                centers = new_centers
        
        # Назначаем точки к ближайшим центрам кластеров
        labels = np.zeros(self.X_.shape[0], dtype=int)
        for i, point in enumerate(self.X_normalized_):
            distances = [np.linalg.norm(point - center) for center in centers]
            labels[i] = np.argmin(distances)
        
        self.cluster_centers_ = centers
        return labels
    
    def fit(self, X: np.ndarray) -> 'DUCtreamClustering':
        """
        Выполнение кластеризации на X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Входные данные
            
        Returns:
        --------
        self : object
            Возвращает сам экземпляр
        """
        X = np.asarray(X, dtype=np.float64)
        self.X_ = X.copy()
        
        # Нормализация данных
        if self.normalize:
            self.scaler_ = StandardScaler()
            self.X_normalized_ = self.scaler_.fit_transform(X)
        else:
            self.X_normalized_ = X.copy()
        
        # Применение вейвлет-декомпозиции
        X_decomposed = self._apply_wavelet_decomposition(self.X_normalized_)
        
        # Инициализация микрокластеров
        self.microclusters_ = []
        timestamp = 0
        
        # Обработка данных в потоковом режиме
        for i, point in enumerate(X_decomposed):
            timestamp += 1
            
            # Затухание весов существующих микрокластеров
            for mc in self.microclusters_:
                time_diff = timestamp - mc['timestamp']
                mc['weight'] *= (self.lambda_param ** time_diff)
            
            # Поиск ближайшего микрокластера
            nearest_idx = self._find_nearest_microcluster(point, self.microclusters_)
            
            if nearest_idx is not None:
                # Обновление существующего микрокластера
                self._update_microcluster(
                    self.microclusters_[nearest_idx], 
                    point, 
                    1.0, 
                    timestamp
                )
            else:
                # Создание нового микрокластера
                new_mc = self._create_microcluster(point, 1.0, timestamp)
                self.microclusters_.append(new_mc)
            
            # Периодическая очистка слабых микрокластеров
            if timestamp % 100 == 0:
                self._remove_weak_microclusters(self.microclusters_, timestamp)
                # Объединение близких микрокластеров
                self.microclusters_ = self._merge_microclusters(self.microclusters_)
        
        # Финальная очистка и объединение
        self._remove_weak_microclusters(self.microclusters_, timestamp)
        self.microclusters_ = self._merge_microclusters(self.microclusters_)
        
        # Кластеризация микрокластеров в финальные кластеры
        self.labels_ = self._cluster_microclusters(self.microclusters_)
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление кластеров и предсказание индекса кластера для каждого образца
        
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


# Интеграция в существующую структуру проекта
@StrategiesManager.registerStrategy(
    "ductream",
    "DUCtream Clustering",
    "Кластеризация потоковых данных с неопределенностью и вейвлет-декомпозицией"
)
class ConcreteStrategyDUCtream(Strategy):
    """
    DUCtream strategy for the DMM Clustering System
    """
    
    @classmethod
    def _setupParams(cls):
        cls._addParam("num_clusters", "Количество кластеров", StrategyParamType.UNumber,
                     """
                     Ожидаемое количество кластеров для выделения.
                     """,
                     5)
        
        cls._addParam("lambda_param", "Параметр затухания", StrategyParamType.UFloating,
                     """
                     Параметр затухания для потоковых данных (0-1).
                     Меньшие значения быстрее забывают старые данные.
                     """,
                     0.25)
        
        cls._addParam("epsilon", "Радиус окрестности", StrategyParamType.UFloating,
                     """
                     Радиус окрестности для микрокластеров.
                     Определяет, насколько близко должны быть точки для объединения.
                     """,
                     0.1)
        
        cls._addParam("beta", "Порог outlier", StrategyParamType.UFloating,
                     """
                     Порог для определения outlier микрокластеров.
                     Микрокластеры с весом меньше beta * mu удаляются.
                     """,
                     0.2)
        
        cls._addParam("mu", "Минимум точек", StrategyParamType.UNumber,
                     """
                     Минимальное количество точек в значимом микрокластере.
                     """,
                     2)
        
        cls._addParam("wavelet", "Тип вейвлета", StrategyParamType.Switch,
                     """
                     Тип вейвлет-функции для декомпозиции.
                     haar - самый простой и быстрый
                     db4 - Добеши 4-го порядка
                     coif1 - Койфлет 1-го порядка
                     """,
                     "haar",
                     switches=["haar", "db4", "db2", "db6", "coif1"])
        
        cls._addParam("n_levels", "Количество уровней", StrategyParamType.UNumber,
                     """
                     Количество уровней вейвлет-разложения.
                     Большие значения дают более агрессивное сжатие данных.
                     """,
                     2)
    
    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация изображения методом DUCtream
        
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
        pixels = np.asarray(pixels, dtype=np.float64)
        
        # Если данные в формате (n_features, n_samples), транспонируем
        if pixels.shape[0] < pixels.shape[1] and pixels.shape[0] <= 10:
            print(f"🔄 Транспонирование данных: {pixels.shape} -> {pixels.T.shape}")
            pixels = pixels.T
        
        # Ограничение размера данных для предотвращения зависания
        MAX_SAMPLES = 10000  # Максимальное количество пикселей для обработки
        
        if len(pixels) > MAX_SAMPLES:
            # Используем стратегическую выборку
            step = int(np.sqrt(len(pixels) / MAX_SAMPLES))
            indices = np.arange(0, len(pixels), step)[:MAX_SAMPLES]
            sample_pixels = pixels[indices]
            use_sampling = True
        else:
            sample_pixels = pixels
            use_sampling = False
        
        model = DUCtreamClustering(
            num_clusters=int(params["num_clusters"]),
            lambda_param=float(params["lambda_param"]),
            epsilon=float(params["epsilon"]),
            beta=float(params["beta"]),
            mu=int(params["mu"]),
            wavelet=params["wavelet"],
            n_levels=int(params["n_levels"]),
            normalize=True
        )
        
        sample_labels = model.fit_predict(sample_pixels)
        
        if use_sampling:
            # Применяем модель ко всем пикселям через ближайшие соседи
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = min(3, len(sample_pixels) // 10)
            knn = KNeighborsClassifier(n_neighbors=max(1, n_neighbors),
                                       algorithm='ball_tree' if len(pixels) > 10000 else 'auto',
                                       n_jobs=-1)
            knn.fit(sample_pixels, sample_labels)
            labels = knn.predict(pixels)
        else:
            labels = sample_labels
        
        return labels
    
    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        """
        Кластеризация точек методом DUCtream
        
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
        points = np.asarray(points, dtype=np.float64)
        
        # Если данные в формате (n_features, n_samples), транспонируем
        if points.shape[0] < points.shape[1] and points.shape[0] <= 10:
            print(f"🔄 Транспонирование данных: {points.shape} -> {points.T.shape}")
            points = points.T
        
        model = DUCtreamClustering(
            num_clusters=int(params["num_clusters"]),
            lambda_param=float(params["lambda_param"]),
            epsilon=float(params["epsilon"]),
            beta=float(params["beta"]),
            mu=int(params["mu"]),
            wavelet=params["wavelet"],
            n_levels=int(params["n_levels"]),
            normalize=True
        )
        
        return model.fit_predict(points)

