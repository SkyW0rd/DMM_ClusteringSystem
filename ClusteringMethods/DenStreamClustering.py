"""
DenStream Clustering Algorithm Implementation
Стратегия кластеризации на основе алгоритма denstream из библиотеки pyDenStream

DenStream (Density-Based Clustering over an Evolving Data Stream with Noise) —
Алгоритм плотностной кластеризации развивающихся потоков данных с шумом
Работает в двух фазах: online-phase, offline-phase
online-phase:
    сжатие точек в статистические структуры - микрокластеры
offline-phase:
    объединение микрокластеров в выходные кластеры на основе DBSCAN
Алгоритм способен находить кластеры произвольной формы
В отличие от CluStream алгоритм выделяет два типа микрокластеров: p-micro-clusters, o-micro-clusters.
    p-micro-clusters (Potential): Потенциальные кластеры. Это группы, в которых плотность (суммарный вес точек) выше определенного порога
    o-micro-clusters (Outlier): Микрокластеры выбросов. Сюда попадают новые или изолированные точки.

Алгоритм реализует модель затухания. С течением времени вес точки внутри микрокластера уменьшается.
Если суммарный вес o-micro-cluster ниже определенного порога, то микрокластер удаляется из памяти
"""

import numpy as np
from denstream import DenStream
import sklearn.cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy,
    StrategyParamType,
    StrategyRunConfig,
    StrategiesManager
)


class DenStreamClustering:
    def __init__(self, epsilon: float = 0.8, beta: float = 0.75, mu: int = 8, lambd: float = 0.001, offline_option: float = 1.7):
        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.lambd = lambd
        self.min_samples = mu
        self.offline_option = offline_option

        self.model = DenStream(
            epsilon=self.epsilon, 
            beta=self.beta, 
            mu=self.mu,
            lambd=self.lambd, 
            min_samples=self.min_samples
        )

        self.labels = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)

        for i, row in enumerate(X):
            self.model.partial_fit(row.reshape(1, -1), time=i)

        labels = np.full(n_samples, -1, dtype=np.int32)

        if len(self.model.p_micro_clusters) == 0:
            self.labels = labels
            return self

        centers = np.concatenate([mc.center for mc in self.model.p_micro_clusters], axis=0)
        weights = np.array(
            [mc.weight.item() for mc in self.model.p_micro_clusters], dtype=np.float64
        )

        dbscan = sklearn.cluster.DBSCAN(
            eps=self.epsilon*self.offline_option,
            min_samples=self.min_samples,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1,
        )
        micro_labels = dbscan.fit_predict(centers, sample_weight=weights)

        for mc_idx, mc in enumerate(self.model.p_micro_clusters):
            macro_id = int(micro_labels[mc_idx])
            if macro_id == -1:
                continue
            labels[mc.time_array.flatten()] = macro_id
  
        self.labels = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels



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


@StrategiesManager.registerStrategy(
    "denstream",
    "DENSTREAM",
    "Density-Based Clustering over an Evolving Data Stream with Noise (pyDenStream)"
)
class ConcreteStrategyDENSTREAM(Strategy):
    """Метод кластеризации с использованием DENSTREAM из pyDenStream
    DENSTREAM распределяет точки по микрокластерам если в eps-окрестности есть микрокластеры, иначе создает новый o-micro-cluster.
    При уменьшении eps данные дробятся на множество отдельных микрокластеров, при увеличении - микрокластеры сливаются в один микрокластер
    При увеличении λ алгоритм быстрее забывает данные полученные ранее, адаптируется к изменениям в потоке
    При увеличении μ растет плотность p-микрокластеров и сокращается их количество
    При увеличении β повышается устойчивость алгоритма к шуму
    """

    @classmethod
    def _setupParams(cls):
        cls._addParam(
            "lambd",
            "Коэффициент затухания",
            StrategyParamType.UFloating,
            """(Decaying factor) Параметр, контролирующий важность исторических данных для текущего кластера. 
            Коэффициента затухания должен быть строго больше 0.""",
            0.001
        )
        cls._addParam(
            "beta",
            "Коэффициент толерантности к шуму",
            StrategyParamType.UFloating,
            """(Beta factor) Параметр, регулирующий границу между потенциальными микрокластерами и шумом. 
            Значение бета должно быть в интеравале (0,1]. Чем больше beta, тем меньше число ложноположительных кластеров.
            """,
            0.75
        )
        cls._addParam(
            "mu",
            "Параметр минимальной плотности потенциального микрокластера",
            StrategyParamType.UNumber,
            """(mu) Параметр, определяющий порог выбросов (o-micro-cluster) 
            относительно потенциального микрокластера (p-micro-clusters).
            Должен находится в интервале (1/beta, inf].
            Чем больше, тем меньше будет выделено потенциальных микрокластеров и тем больше будет их плотность
            """,
            8
        )
        cls._addParam(
            "epsilon",
            "Радиус окрестности микрокластера",
            StrategyParamType.UFloating,
            """(epsilon) Определяет epsilon-окрестность микрокластеров.
            Чем больше, тем более грубой будет финальная кластеризация, p-micro-clustres станут крупнее.
            """,
            0.8
        )
        cls._addParam(
            "offline_option",
            "Множитель для epsilon на оффлайн фазе",
            StrategyParamType.UFloating,
            """(offline_option) Расстояние между центрами p-micro-clusters <= epsilon*offline_option
            Чем больше, тем более грубой будет финальная кластеризация, кластеры станут крупнее, более отдаленные p-micro-clusters будут объединяться в один кластер.
            """,
            1.7
        )
        

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        
        pixels = _prepare_features(pixels)
        pixels_proc = StandardScaler().fit_transform(pixels)

        _IMAGE_MAX_SAMPLES = 3000
        if len(pixels_proc) > _IMAGE_MAX_SAMPLES:

            flat_stride = len(pixels_proc) / _IMAGE_MAX_SAMPLES
            indices = np.round(np.arange(_IMAGE_MAX_SAMPLES) * flat_stride).astype(int)
            
            sample_pixels = pixels_proc[indices]

            model = DenStreamClustering(
                epsilon=float(params["epsilon"]),
                beta=float(params["beta"]),
                mu=int(params["mu"]),
                lambd=float(params["lambd"]),
                offline_option=float(params["offline_option"])
            )

            sample_labels = model.fit_predict(sample_pixels)
            knn = KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                algorithm='ball_tree' if len(pixels_proc) > 10000 else 'auto',
                n_jobs=-1
            )
            knn.fit(sample_pixels, sample_labels)
            labels = knn.predict(pixels_proc)
        else:
            model = DenStreamClustering(
                epsilon=float(params["epsilon"]),
                beta=float(params["beta"]),
                mu=int(params["mu"]),
                lambd=float(params["lambd"]),
                offline_option=float(params["offline_option"])
            )
            labels = model.fit_predict(pixels_proc)

        return labels

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        points = _prepare_features(points)
        points_proc = StandardScaler().fit_transform(points)
        
        model = DenStreamClustering(
            epsilon=float(params["epsilon"]),
            beta=float(params["beta"]),
            mu=int(params["mu"]),
            lambd=float(params["lambd"]),
            offline_option=float(params["offline_option"])
        )

        return model.fit_predict(points_proc)




    
    