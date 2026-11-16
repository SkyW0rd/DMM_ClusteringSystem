"""
PROCLUS Algorithm Implementation
Автор: Мизюрин С. В. [mizyurinsv@gmail.com]
Последнее обновление: 2026-11-04

PROCLUS (PROjected CLUStering) - проекционная кластеризация на основе медоидов.
Каждый из k кластеров описывается медоидом и набором признаков D_i, в котором 
точки кластера лежат близко друг к другу. Средний размер наборов признаков задаёт 
пользователь (l); всего выбирается k*l признаков, не меньше двух на кластер. 
Точки вне всех кластеров - выбросы.

Источники:
- Aggarwal C. C., Procopiuc C., Wolf J. L., Yu P. S., Park J. S.
  Fast Algorithms for Projected Clustering. SIGMOD 1999.
  https://doi.org/10.1145/304181.304188
- ELKI: elki.clustering.subspace.PROCLUS (Java), https://elki-project.github.io/
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from ClusteringMethods.ClasteringAlgorithms import (
    Strategy, StrategyParamType, StrategyRunConfig, StrategiesManager,
)
from ClusteringMethods._validation import (
    prepare_features, positive_integer, positive_real,
)


class PROCLUS:
    """Проекционная кластеризация: fit_predict(X).

    После fit доступны:
        labels_ - метка каждой точки, -1 означает выброс;
        medoid_indices_ - индексы медоидов в X;
        medoids_ - координаты медоидов;
        dimensions_ - список массивов номеров признаков D_i каждого кластера;
        objective_ - значение целевой функции лучшего набора медоидов;
        n_iter_ - число итераций второй фазы в лучшем запуске.

    n_init - число независимых запусков с разными сидами:
    выбирается запуск с наименьшим значением целевой функции;
    это помогает снизить риск застрять в плохом наборе
    медоидов (например, когда два медоида попали в один кластер).
    """

    def __init__(self, n_clusters=3, avg_dimensions=2, sample_factor=30,
                 medoid_factor=10, min_deviation=0.1, max_no_improve=30,
                 max_iter=500, n_init=3, detect_outliers=True, normalize=False,
                 random_state=0):
        self.n_clusters = n_clusters
        self.avg_dimensions = avg_dimensions
        self.sample_factor = sample_factor
        self.medoid_factor = medoid_factor
        self.min_deviation = min_deviation
        self.max_no_improve = max_no_improve
        self.max_iter = max_iter
        self.n_init = n_init
        self.detect_outliers = detect_outliers
        self.normalize = normalize
        self.random_state = random_state
        self._l = avg_dimensions      # в fit ограничивается числом непостоянных признаков

    def _validate(self, n, d):
        positive_integer("n_clusters", self.n_clusters)
        positive_integer("avg_dimensions", self.avg_dimensions)
        positive_integer("sample_factor", self.sample_factor)
        positive_integer("medoid_factor", self.medoid_factor)
        positive_integer("max_no_improve", self.max_no_improve)
        positive_integer("max_iter", self.max_iter)
        positive_integer("n_init", self.n_init)
        positive_integer("random_state", self.random_state, allow_zero=True)
        positive_real("min_deviation", self.min_deviation, allow_zero=True)
        if self.min_deviation >= 1:
            raise ValueError("min_deviation должен быть меньше 1.")
        if self.n_clusters < 2:
            raise ValueError("n_clusters должен быть не меньше 2.")
        if self.avg_dimensions < 2:
            raise ValueError("avg_dimensions должен быть не меньше 2.")
        if d < 2:
            raise ValueError("PROCLUS требует не меньше двух признаков.")
        if self.avg_dimensions > d:
            raise ValueError(f"avg_dimensions ({self.avg_dimensions}) больше числа признаков ({d}).")
        if self.n_clusters >= n:
            raise ValueError(f"n_clusters ({self.n_clusters}) должен быть меньше числа точек ({n}).")

    def fit(self, data) -> "PROCLUS":
        X_input = prepare_features(data)
        n, d = X_input.shape
        self._validate(n, d)
        active = np.flatnonzero(np.ptp(X_input, axis=0) > 0)
        if active.size < 2:
            raise ValueError("PROCLUS требует не меньше двух непостоянных признаков.")
        self._l = min(self.avg_dimensions, active.size)
        X = X_input[:, active]
        if self.normalize:
            X = StandardScaler().fit_transform(X)
        seeds = np.random.SeedSequence(self.random_state).spawn(self.n_init)
        runs = [self._fit_once(X, np.random.default_rng(s)) for s in seeds]
        best_objective, best, best_labels, n_iter = min(runs, key=lambda r: r[0])
        k = self.n_clusters

        # уточнение признаков по кластерам, переназначение, выбросы.
        clusters = [np.flatnonzero(best_labels == i) for i in range(k)]
        dims = self._find_dimensions(X, best, clusters)
        labels = self._assign(X, X[best], dims)
        if self.detect_outliers:
            labels = self._mark_outliers(X, best, dims, labels)

        self.medoid_indices_ = best
        self.medoids_ = X_input[best]
        self.dimensions_ = [active[D] for D in dims]    # номера исходных признаков
        self.objective_ = best_objective
        self.n_iter_ = n_iter
        self.labels_ = _relabel(labels)
        return self

    def fit_predict(self, data) -> np.ndarray:
        return self.fit(data).labels_

    def _fit_once(self, X, rng):
        n = len(X)
        k = self.n_clusters

        # выборка A*k точек и жадный выбор B*k кандидатов в медоиды.
        sample = rng.choice(n, size=min(n, self.sample_factor * k), replace=False)
        candidates = self._greedy(X, sample, min(sample.size, self.medoid_factor * k), rng)
        if candidates.size < k:
            raise ValueError("Слишком мало различных точек для выбора медоидов.")

        # итеративный поиск лучшего набора медоидов.
        current = rng.choice(candidates, size=k, replace=False)
        best = None
        best_objective = np.inf
        bad = np.array([], dtype=int)
        no_improve = 0
        n_iter = 0
        while no_improve < self.max_no_improve and n_iter < self.max_iter:
            n_iter += 1
            localities = self._localities(X, current)
            dims = self._find_dimensions(X, current, localities)
            labels = self._assign(X, X[current], dims)
            objective = self._evaluate(X, labels, dims)
            if objective < best_objective:
                best_objective, best, best_labels = objective, current.copy(), labels
                bad = self._bad_medoids(labels, n)
                no_improve = 0
            else:
                no_improve += 1
            current = self._replace_bad(best, bad, candidates, rng)
        return best_objective, best, best_labels, n_iter


    @staticmethod
    def _greedy(X, sample, m, rng):
        pts = X[sample]
        chosen = [int(rng.integers(sample.size))]
        dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
        for _ in range(1, m):
            nxt = int(np.argmax(dist))
            if dist[nxt] == 0:          # остались только совпадающие точки
                break
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
        return sample[np.array(chosen)]

    @staticmethod
    def _localities(X, medoids):
        M = X[medoids]
        between = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
        np.fill_diagonal(between, np.inf)
        delta = between.min(axis=1)
        return [np.flatnonzero(np.linalg.norm(X - M[i], axis=1) <= delta[i])
                for i in range(len(medoids))]

    def _find_dimensions(self, X, medoids, groups):
        k, d = len(medoids), X.shape[1]
        Xij = np.empty((k, d))
        for i, (m, group) in enumerate(zip(medoids, groups)):
            pts = X[group] if group.size else X[[m]]
            Xij[i] = np.abs(pts - X[m]).mean(axis=0)
        Y = Xij.mean(axis=1, keepdims=True)
        sigma = Xij.std(axis=1, ddof=1, keepdims=True)
        Z = np.divide(Xij - Y, sigma, out=np.zeros_like(Xij), where=sigma > 0)

        chosen = np.zeros((k, d), dtype=bool)
        order = np.argsort(Z, axis=1, kind="stable")
        chosen[np.arange(k)[:, None], order[:, :2]] = True
        remaining = k * self._l - 2 * k
        if remaining > 0:
            rest = np.where(chosen, np.inf, Z).ravel()
            pick = np.argsort(rest, kind="stable")[:remaining]
            chosen.ravel()[pick] = True
        return [np.flatnonzero(row) for row in chosen]

    @staticmethod
    def _segmental(X, centers, dims):
        return np.column_stack([np.abs(X[:, D] - c[D]).mean(axis=1) for c, D in zip(centers, dims)])

    def _assign(self, X, centers, dims):
        return self._segmental(X, centers, dims).argmin(axis=1)

    @staticmethod
    def _evaluate(X, labels, dims):
        total = 0.0
        for i, D in enumerate(dims):
            pts = X[labels == i]
            if pts.size == 0:
                continue
            w = np.abs(pts[:, D] - pts[:, D].mean(axis=0)).mean()
            total += len(pts) * w
        return total / len(X)

    def _bad_medoids(self, labels, n):
        sizes = np.bincount(labels, minlength=self.n_clusters)
        bad = set(np.flatnonzero(sizes < n / self.n_clusters * self.min_deviation).tolist())
        bad.add(int(np.argmin(sizes)))
        return np.array(sorted(bad), dtype=int)

    @staticmethod
    def _replace_bad(best, bad, candidates, rng):
        current = best.copy()
        pool = np.setdiff1d(candidates, best)
        if pool.size == 0 or bad.size == 0:
            return current
        replace = rng.choice(pool, size=min(bad.size, pool.size), replace=False)
        current[bad[:replace.size]] = replace
        return current

    def _mark_outliers(self, X, medoids, dims, labels):
        M = X[medoids]
        Delta = np.array([min(np.abs(M[j, D] - M[i, D]).mean() for j in range(len(medoids)) if j != i)
                          for i, D in enumerate(dims)])
        dist = self._segmental(X, M, dims)
        outlier = np.all(dist > Delta[None, :], axis=1)
        labels = labels.copy()
        labels[outlier] = -1
        return labels


def _relabel(labels):
    result = np.full(labels.shape, -1, dtype=np.int64)
    used = np.unique(labels[labels >= 0])
    for new, old in enumerate(used):
        result[labels == old] = new
    return result


@StrategiesManager.registerStrategy(
    "proclus", "PROCLUS", "Проекционная кластеризация: k медоидов, у каждого свой набор признаков",
)
class ConcreteStrategyPROCLUS(Strategy):
    @classmethod
    def _setupParams(cls):
        cls._addParam("n_clusters", "Число кластеров (k)", StrategyParamType.UNumber,
                      "Число кластеров, не меньше 2.", 3)
        cls._addParam("avg_dimensions", "Среднее число признаков (l)", StrategyParamType.UNumber,
                      "Средний размер набора признаков кластера; всего выбирается k*l, "
                      "не меньше 2 на кластер. Не больше числа признаков данных.", 2)
        cls._addParam("sample_factor", "Множитель выборки (A)", StrategyParamType.UNumber,
                      "Размер случайной выборки A*k для поиска кандидатов в медоиды.", 30)
        cls._addParam("medoid_factor", "Множитель кандидатов (B)", StrategyParamType.UNumber,
                      "Число кандидатов в медоиды B*k, выбираемых жадно из выборки.", 10)
        cls._addParam("min_deviation", "Мин. доля кластера", StrategyParamType.UFloating,
                      "Медоид плохой, если в его кластере меньше N/k * min_deviation точек.", 0.1)
        cls._addParam("max_no_improve", "Итераций без улучшения", StrategyParamType.UNumber,
                      "Остановка второй фазы после стольких итераций без улучшения.", 30)
        cls._addParam("n_init", "Число запусков", StrategyParamType.UNumber,
                      "Независимые запуски с разными сидами; берётся запуск с лучшей "
                      "целевой функцией (как n_init в KMeans).", 3)
        cls._addParam("detect_outliers", "Выделять выбросы", StrategyParamType.Bool,
                      "Точки, далёкие от всех медоидов в их признаках, получают метку -1.", True)
        cls._addParam("normalize", "Нормализовать данные", StrategyParamType.Bool,
                      "StandardScaler перед кластеризацией (признаки в разных единицах).", False)
        cls._addParam("random_state", "Seed", StrategyParamType.UNumber,
                      "сид генератора случайных чисел.", 0)

    def _cluster(self, data, params):
        values = {name: params[name] for name in self.params()}
        values["min_deviation"] = float(values["min_deviation"])
        return PROCLUS(**values).fit_predict(data)

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        return self._cluster(points, params)

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        return self._cluster(pixels, params)
