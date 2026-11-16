"""
Регрессионные тесты PROCLUS (ClusteringMethods/PROCLUSClustering.py)
Автор: Мизюрин С. В. [mizyurinsv@gmail.com]

Запуск из корня проекта:
    python -m unittest TestMethods.test_proclus_regression -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ClusteringMethods.ClasteringAlgorithms import Context, StrategiesManager  # noqa: E402
from ClusteringMethods.PROCLUSClustering import PROCLUS, ConcreteStrategyPROCLUS  # noqa: E402


def projected_data(n=3000, d=20, k=5, l=5, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 100, (n, d))
    y = rng.integers(0, k, n)
    dims = []
    for i in range(k):
        D = np.sort(rng.choice(d, l, replace=False))
        dims.append(D)
        X[np.ix_(y == i, D)] = rng.uniform(10, 90, l) + rng.normal(0, 3, ((y == i).sum(), l))
    m = int(n * noise)
    X[:m] = rng.uniform(0, 100, (m, d))
    y[:m] = -1
    return X, y, dims


def configuration(**values):
    config = StrategiesManager.getStrategyRunConfigById("proclus")
    for key, value in values.items():
        config[key] = value
    return config


class ProjectedClusteringTests(unittest.TestCase):
    def test_recovers_clusters_and_dimensions(self):
        X, y, dims = projected_data()
        model = PROCLUS(n_clusters=5, avg_dimensions=5).fit(X)
        inliers = y >= 0
        # отдельные точки кластеров на краю могут уйти в выбросы
        self.assertGreater(adjusted_rand_score(y[inliers], model.labels_[inliers]), 0.99)
        found = {tuple(D) for D in model.dimensions_}
        self.assertEqual(found, {tuple(D) for D in dims})

    def test_dimension_budget(self):
        X, _, _ = projected_data(d=30, l=6, seed=1)
        model = PROCLUS(n_clusters=5, avg_dimensions=6).fit(X)
        sizes = [len(D) for D in model.dimensions_]
        self.assertEqual(sum(sizes), 30)
        self.assertGreaterEqual(min(sizes), 2)

    def test_outliers(self):
        X, y, _ = projected_data()
        labels = PROCLUS(n_clusters=5, avg_dimensions=5).fit_predict(X)
        self.assertGreater((labels[y == -1] == -1).mean(), 0.1)   # часть шума отделена
        self.assertLess((labels[y >= 0] == -1).mean(), 0.01)      # кластеры почти не теряют точек
        no_outliers = PROCLUS(n_clusters=5, avg_dimensions=5, detect_outliers=False).fit_predict(X)
        self.assertFalse((no_outliers == -1).any())


class StepTests(unittest.TestCase):
    def setUp(self):
        self.model = PROCLUS(n_clusters=2, avg_dimensions=2)

    def test_segmental_distance(self):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 4.0, 10.0]])
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        dims = [np.array([0, 1]), np.array([0, 1, 2])]
        dist = self.model._segmental(X, centers, dims)
        np.testing.assert_allclose(dist, [[0.0, 1.0], [2.5, (0 + 3 + 9) / 3]])

    def test_find_dimensions_uses_smallest_z(self):
        rng = np.random.default_rng(0)
        a = np.column_stack([rng.normal(0, 0.1, (50, 2)), rng.uniform(-10, 10, (50, 2))])
        b = np.column_stack([rng.uniform(-10, 10, (50, 2)), rng.normal(0, 0.1, (50, 2))])
        X = np.vstack([a, b])
        dims = self.model._find_dimensions(X, np.array([0, 50]), [np.arange(50), np.arange(50, 100)])
        self.assertEqual([list(D) for D in dims], [[0, 1], [2, 3]])

    def test_objective(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [10.0, 14.0]])
        labels = np.array([0, 0, 1, 1])
        value = self.model._evaluate(X, labels, [np.array([0]), np.array([0, 1])])
        self.assertAlmostEqual(value, (2 * 1.0 + 2 * (0 + 2) / 2) / 4)

    def test_bad_medoids(self):
        labels = np.array([0] * 70 + [1] * 22 + [2] * 8)
        # N/k * min_deviation = 100/3 * 0.3 = 10: кластер 2 мал (и самый маленький)
        self.assertEqual(PROCLUS(n_clusters=3, min_deviation=0.3)._bad_medoids(labels, 100).tolist(), [2])
        # при пороге 100/3 * 0.9 = 30 плохи кластеры 1 и 2
        self.assertEqual(PROCLUS(n_clusters=3, min_deviation=0.9)._bad_medoids(labels, 100).tolist(), [1, 2])
        # без малых кластеров плохим остаётся только самый маленький
        self.assertEqual(PROCLUS(n_clusters=3, min_deviation=0.0)._bad_medoids(labels, 100).tolist(), [2])

    def test_restarts_choose_lowest_objective(self):
        X, _, _ = projected_data(n=1000)
        combined = PROCLUS(n_clusters=5, avg_dimensions=5, n_init=4, random_state=3).fit(X)
        single = []
        for s in np.random.SeedSequence(3).spawn(4):
            single.append(PROCLUS(n_clusters=5, avg_dimensions=5)._fit_once(X, np.random.default_rng(s))[0])
        self.assertAlmostEqual(combined.objective_, min(single))


class ResultTests(unittest.TestCase):
    def test_medoids_are_data_points_and_labels(self):
        X, _, _ = projected_data(n=1000)
        model = PROCLUS(n_clusters=5, avg_dimensions=5).fit(X)
        np.testing.assert_array_equal(model.medoids_, X[model.medoid_indices_])
        self.assertEqual(model.labels_.shape, (1000,))
        self.assertTrue(np.issubdtype(model.labels_.dtype, np.integer))
        used = np.unique(model.labels_[model.labels_ >= 0])
        np.testing.assert_array_equal(used, np.arange(used.size))

    def test_reproducible_with_seed(self):
        X, _, _ = projected_data(n=1000)
        first = PROCLUS(n_clusters=5, avg_dimensions=5, random_state=7).fit_predict(X)
        second = PROCLUS(n_clusters=5, avg_dimensions=5, random_state=7).fit_predict(X)
        np.testing.assert_array_equal(first, second)

    def test_normalize_removes_unit_dependence(self):
        X, _, _ = projected_data(n=1000)
        scaled = X.copy()
        scaled[:, 0] *= 1000
        a = PROCLUS(n_clusters=5, avg_dimensions=5, normalize=True).fit_predict(X)
        b = PROCLUS(n_clusters=5, avg_dimensions=5, normalize=True).fit_predict(scaled)
        np.testing.assert_array_equal(a, b)

    def test_constant_features_are_ignored(self):
        X, y = make_blobs(600, centers=[[-4, 0], [0, 5], [4, 0]], cluster_std=0.6, random_state=6767)
        padded = np.column_stack([X[:, 0], np.zeros(len(X)), X[:, 1]])
        plain = PROCLUS(detect_outliers=False).fit(X)
        model = PROCLUS(detect_outliers=False).fit(padded)
        np.testing.assert_array_equal(plain.labels_, model.labels_)
        self.assertEqual(adjusted_rand_score(y, model.labels_), 1.0)
        for D in model.dimensions_:                       # номера исходных признаков
            self.assertEqual(D.tolist(), [0, 2])
        PROCLUS(avg_dimensions=3, detect_outliers=False).fit(padded)
        with self.assertRaises(ValueError):
            PROCLUS().fit(np.column_stack([X[:, 0], np.zeros(len(X))]))

    def test_invalid_parameters(self):
        X = np.random.default_rng(0).normal(size=(50, 4))
        for kwargs in ({"n_clusters": 1}, {"avg_dimensions": 1}, {"avg_dimensions": 5},
                       {"min_deviation": 1.0}, {"min_deviation": -0.1}, {"sample_factor": 0},
                       {"medoid_factor": 0}, {"max_no_improve": 0}, {"n_clusters": 2.5},
                       {"n_clusters": 50}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                PROCLUS(**kwargs).fit(X)
        with self.assertRaises(ValueError):
            PROCLUS().fit(np.zeros((10, 1)))        # меньше двух признаков
        with self.assertRaises(ValueError):
            PROCLUS().fit([[np.inf, 1.0]] * 5)


class StrategyTests(unittest.TestCase):
    def test_registered_with_defaults(self):
        self.assertIn("proclus", StrategiesManager.strategies())
        config = configuration()
        self.assertEqual(config["n_clusters"], 3)
        self.assertEqual(config["avg_dimensions"], 2)

    def test_context_matches_direct_call(self):
        X, y, _ = projected_data(n=800, d=10, k=3, l=3)
        config = configuration(avg_dimensions=3)
        direct = ConcreteStrategyPROCLUS().clastering_points(X, config)
        via_context = Context(ConcreteStrategyPROCLUS()).do_some_clustering_points(X.T.tolist(), config)
        np.testing.assert_array_equal(direct, via_context)
        self.assertGreater(adjusted_rand_score(y[y >= 0], direct[y >= 0]), 0.95)

    def test_image_pixels(self):
        rng = np.random.default_rng(6)
        pixels = np.vstack([rng.normal(40, 4, (400, 3)), rng.normal(200, 4, (400, 3)),
                            rng.normal([200, 40, 120], 4, (400, 3))]).clip(0, 255)
        labels = ConcreteStrategyPROCLUS().clastering_image(pixels, configuration(detect_outliers=False))
        self.assertEqual(adjusted_rand_score(np.repeat([0, 1, 2], 400), labels), 1.0)


if __name__ == "__main__":
    unittest.main()
