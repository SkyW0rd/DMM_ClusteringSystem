"""Проверки и визуализации Spherical K-Means. Автор — Кулаков Никита.

python TestMethods/test_spherical_kmeans.py --no-show
python -m unittest TestMethods.test_spherical_kmeans -v
"""

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ClusteringMethods.ClasteringAlgorithms import Context, StrategiesManager
from ClusteringMethods.SphericalKMeansClustering import (
    ConcreteStrategySphericalKMeans, SphericalKMeans,
)


def directional_data(n_features=2, n_per_cluster=40):
    """Три направления с независимыми длинами векторов и малым угловым шумом."""
    rng = np.random.default_rng(42)
    if n_features == 2:
        angles = np.arange(3) * 2 * np.pi / 3
        centers = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        centers = np.eye(3, n_features)
    truth = np.repeat(np.arange(3), n_per_cluster)
    points = centers[truth] + rng.normal(0, 0.025, (len(truth), n_features))
    points *= rng.uniform(0.2, 8, (len(truth), 1))
    return points, truth


def configuration(**values):
    config = StrategiesManager.getStrategyRunConfigById("spherical_kmeans")
    for name, value in values.items():
        config[name] = value
    return config


class SphericalKMeansTests(unittest.TestCase):
    def test_basic_2d_clustering(self):
        points, truth = directional_data()
        for init in ("k-means++", "random"):
            with self.subTest(init=init):
                model = SphericalKMeans(init=init).fit(points)
                self.assertGreater(adjusted_rand_score(truth, model.labels_), 0.99)
                assert_allclose(np.linalg.norm(model.cluster_centers_, axis=1), 1)

    def test_centers_are_normalized_sums(self):
        points, _ = directional_data()
        model = SphericalKMeans(tol=0).fit(points)
        unit = points / np.linalg.norm(points, axis=1, keepdims=True)
        for index, center in enumerate(model.cluster_centers_):
            total = unit[model.labels_ == index].sum(axis=0)
            assert_allclose(center, total / np.linalg.norm(total), atol=1e-12)

    def test_one_iteration_uses_spherical_assignment(self):
        angles = np.deg2rad([-125.53, -49.65, 36.56, -43.41, -169.24,
                             60.05, 80.04, -66.2, 99.02, -74.17])
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        model = SphericalKMeans(n_clusters=2, init="random", n_init=1,
                                max_iter=1, random_state=42).fit(points)
        # Ручной результат назначения после нормирования обновлённых центров.
        assert_array_equal(model.labels_, [1, 1, 0, 1, 1, 0, 0, 1, 0, 1])

    def test_positive_row_scaling_invariance(self):
        points, _ = directional_data()
        scales = np.geomspace(1e-150, 1e150, len(points))[:, None]
        first = SphericalKMeans().fit(points)
        second = SphericalKMeans().fit(points * scales)
        self.assertEqual(adjusted_rand_score(first.labels_, second.labels_), 1)
        assert_allclose(first.inertia_, second.inertia_, atol=1e-12)

    def test_extreme_finite_magnitudes(self):
        points = [[1e300, 1e300], [1e-300, 1e-300], [-1e300, -1e300], [-1e-300, -1e-300]]
        with np.errstate(all="raise"):
            model = SphericalKMeans(n_clusters=2).fit(points)
        self.assertEqual(adjusted_rand_score([0, 0, 1, 1], model.labels_), 1)
        assert_allclose(model.inertia_, 0, atol=1e-12)

    def test_zero_vectors_are_excluded(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [0, 3], [0, 0]])
        model = SphericalKMeans(n_clusters=2).fit(points)
        assert_array_equal(model.labels_[[0, 5]], [-1, -1])
        self.assertEqual(adjusted_rand_score([0, 0, 1, 1], model.labels_[1:5]), 1)
        assert_array_equal(model.predict(points), model.labels_)
        assert_array_equal(model.predict([[0, 0]]), [-1])

    def test_duplicate_directions_and_empty_clusters(self):
        for init in ("random", "k-means++"):
            with self.subTest(init=init), self.assertWarns(ConvergenceWarning):
                model = SphericalKMeans(n_clusters=3, init=init).fit(np.ones((8, 2)))
            self.assertTrue(np.isfinite(model.cluster_centers_).all())
            assert_allclose(np.linalg.norm(model.cluster_centers_, axis=1), 1)
            assert_array_equal(model.predict(np.ones((8, 2))), model.labels_)

    def test_cancelling_cluster_keeps_unit_center(self):
        model = SphericalKMeans(n_clusters=1).fit([[1, 0], [-1, 0]])
        assert_allclose(np.linalg.norm(model.cluster_centers_, axis=1), 1)
        self.assertAlmostEqual(model.inertia_, 2)

    def test_one_vector_one_feature_and_wide_embeddings(self):
        assert_array_equal(SphericalKMeans(n_clusters=1).fit_predict([[3, 4, 5]]), [0])
        self.assertEqual(SphericalKMeans(n_clusters=2).fit_predict([1, 2, -1, -2]).shape, (4,))
        points = np.eye(4, 128)
        labels = SphericalKMeans(n_clusters=4).fit_predict(points)
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(len(np.unique(labels)), 4)

    def test_high_dimensional_embeddings(self):
        points, truth = directional_data(n_features=128)
        labels = SphericalKMeans(batch_size=7).fit_predict(points)
        self.assertGreater(adjusted_rand_score(truth, labels), 0.99)

    def test_3d_clustering(self):
        points, truth = directional_data(n_features=3)
        labels = ConcreteStrategySphericalKMeans().clastering_points(points, configuration())
        self.assertGreater(adjusted_rand_score(truth, labels), 0.99)

    def test_different_datasets(self):
        points, truth = directional_data()
        for data in (points, points / np.linalg.norm(points, axis=1, keepdims=True)):
            labels = ConcreteStrategySphericalKMeans().clastering_points(data, configuration())
            self.assertGreater(adjusted_rand_score(truth, labels), 0.99)

    def test_parameter_sensitivity(self):
        points, _ = directional_data()
        for k in (2, 3, 5):
            labels = ConcreteStrategySphericalKMeans().clastering_points(points, configuration(n_clusters=k))
            self.assertEqual(len(np.unique(labels)), k)

    def test_iteration_limit_keeps_predictions_and_cost_consistent(self):
        points = np.random.default_rng(17).normal(size=(81, 9))
        model = SphericalKMeans(max_iter=1, n_init=1, tol=0).fit(points)
        self.assertEqual(model.n_iter_, 1)
        assert_array_equal(model.labels_, model.predict(points))
        unit = points / np.linalg.norm(points, axis=1, keepdims=True)
        distances = 1 - np.sum(unit * model.cluster_centers_[model.labels_], axis=1)
        self.assertAlmostEqual(model.inertia_, distances.sum())

    def test_restarts_select_lowest_cosine_cost(self):
        points = np.random.default_rng(15).normal(size=(60, 5))
        first = SphericalKMeans(n_init=1, init="random").fit(points)
        multiple = SphericalKMeans(n_init=7, init="random").fit(points)
        self.assertLessEqual(multiple.inertia_, first.inertia_ + 1e-12)

    def test_batches_and_reproducibility(self):
        points, _ = directional_data()
        first = SphericalKMeans(batch_size=1).fit(points)
        second = SphericalKMeans(batch_size=4096).fit(points)
        assert_array_equal(first.labels_, second.labels_)
        assert_allclose(first.cluster_centers_, second.cluster_centers_, atol=1e-12)
        assert_array_equal(first.labels_, SphericalKMeans(batch_size=1).fit_predict(points))

    def test_input_and_global_rng_are_preserved(self):
        points, _ = directional_data()
        before = points.copy()
        state = np.random.get_state()
        SphericalKMeans().fit(points)
        after = np.random.get_state()
        assert_array_equal(points, before)
        assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_unseeded_fit(self):
        points, _ = directional_data()
        self.assertEqual(SphericalKMeans(random_state=None).fit_predict(points).shape, (len(points),))

    def test_invalid_parameters(self):
        invalid = [("n_clusters", 0), ("n_clusters", 5), ("n_clusters", True),
                   ("n_clusters", 1.5), ("n_init", 0), ("n_init", 1.5),
                   ("max_iter", 0), ("batch_size", 0), ("batch_size", False),
                   ("tol", -1), ("tol", np.nan), ("tol", np.inf), ("tol", True),
                   ("random_state", -1), ("random_state", 2**32),
                   ("random_state", 1.5), ("random_state", False), ("init", "other")]
        for name, value in invalid:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                SphericalKMeans(**{name: value}).fit([[1, 0], [0, 1], [-1, 0], [0, -1]])

    def test_invalid_data_and_insufficient_nonzero_rows(self):
        for points in ([], np.empty((2, 0)), [[[1]]], [[np.nan]], [[np.inf]],
                       [["x"]], [[0, 0]], [[1, 0], [0, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                SphericalKMeans(n_clusters=2).fit(points)

    def test_prediction_validation(self):
        with self.assertRaises(NotFittedError):
            SphericalKMeans().predict([[1, 0]])
        model = SphericalKMeans(n_clusters=1).fit([[1, 0]])
        for points in ([[1, 0, 0]], [[np.nan, 0]], []):
            with self.subTest(points=points), self.assertRaises(ValueError):
                model.predict(points)


class SphericalStrategyTests(unittest.TestCase):
    def test_registry_and_independent_configuration(self):
        strategy = StrategiesManager.createStrategyById("spherical_kmeans")
        self.assertIsInstance(strategy, ConcreteStrategySphericalKMeans)
        self.assertEqual(len(configuration().params), 7)
        self.assertNotIn("batch_size", StrategiesManager.getStrategyRunConfigById("kmeans_sk").params)
        config = configuration(n_clusters=2)
        self.assertEqual(configuration()["n_clusters"], 3)
        self.assertEqual(config["n_clusters"], 2)

    def test_strategy_integration(self):
        pixels = np.repeat([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]], 9, axis=0)
        strategy = StrategiesManager.createStrategyById("spherical_kmeans")
        config = configuration()
        direct = strategy.clastering_points(pixels, config)
        context = Context(strategy)
        assert_array_equal(direct, context.do_some_clustering_points(pixels.T, config))
        assert_array_equal(direct, context.do_some_clustering_image(pixels.reshape(6, 6, 3), config, 1))
        assert_array_equal(direct, context.do_some_clustering_image(pixels, config, 0))
        self.assertEqual(direct.shape, (36,))
        self.assertTrue(np.issubdtype(direct.dtype, np.integer))
        assert_array_equal(direct[-9:], -np.ones(9, dtype=int))

    def test_strategy_parameters_reach_model(self):
        points, _ = directional_data()
        values = dict(n_clusters=2, init="random", n_init=2, max_iter=1,
                      tol=0.0, random_state=7, batch_size=11)
        direct = SphericalKMeans(**values).fit_predict(points)
        labels = ConcreteStrategySphericalKMeans().clastering_points(points, configuration(**values))
        assert_array_equal(direct, labels)


def create_figures(show):
    """Сохранить пять визуальных проверок по соглашению TestMethods."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(__file__).parent / "Images" / "SphericalKMeans"
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=120, bbox_inches="tight")
        if not show:
            plt.close(fig)

    def cluster(data, **options):
        return ConcreteStrategySphericalKMeans().clastering_points(data, configuration(**options))

    points, truth = directional_data()
    model = SphericalKMeans().fit(points)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, labels, title in zip(axes, (truth, cluster(points)), ("Reference directions", "Spherical K-Means")):
        ax.scatter(*points.T, c=labels, cmap="viridis", s=15)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    save(fig, "test_basic_2d.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, k in zip(axes, (2, 3, 5)):
        labels = cluster(points, n_clusters=k)
        ax.scatter(*points.T, c=labels, cmap="viridis", s=15)
        ax.set(title=f"n_clusters = {k}", xlabel="x", ylabel="y", aspect="equal")
    save(fig, "test_parameters.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    datasets = [points, points / np.linalg.norm(points, axis=1, keepdims=True),
                np.vstack([points, np.zeros((10, 2))])]
    for ax, data, title in zip(axes, datasets, ("Variable length", "Unit vectors", "Zero vectors (-1)")):
        labels = cluster(data)
        ax.scatter(*data.T, c=labels, cmap="viridis", s=15)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    save(fig, "test_datasets.png")

    points3d, _ = directional_data(n_features=3)
    labels = cluster(points3d)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*points3d.T, c=labels, cmap="viridis", s=15)
    ax.set(title="Spherical K-Means in 3D", xlabel="x", ylabel="y", zlabel="z")
    save(fig, "test_3d.png")

    labels = Context(ConcreteStrategySphericalKMeans()).do_some_clustering_points(points.T, configuration())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(*points.T, c=labels, cmap="viridis", s=15)
    ax.quiver(np.zeros(3), np.zeros(3), *model.cluster_centers_.T,
              angles="xy", scale_units="xy", scale=0.2, color="black")
    ax.set(title="Strategy / Context and centroid directions", xlabel="x", ylabel="y", aspect="equal")
    save(fig, "test_strategy.png")
    print(f"Визуализации сохранены в {output}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Сохранить графики без открытия окон")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        sys.exit(1)
    create_figures(show=not args.no_show)
