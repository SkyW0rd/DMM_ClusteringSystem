"""
Тестовый скрипт для проверки PROCLUS с визуализацией
Автор: Мизюрин С. В. [mizyurinsv@gmail.com]

Запуск из корня проекта:
    python TestMethods/test_proclus.py --no-show
"""

import argparse
from pathlib import Path
import sys

import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ClusteringMethods.ClasteringAlgorithms import Context, StrategiesManager  # noqa: E402
from ClusteringMethods.PROCLUSClustering import PROCLUS, ConcreteStrategyPROCLUS  # noqa: E402

ALGORITHM_NAME = "PROCLUS"
STRATEGY_ID = "proclus"
IMAGES_DIR = Path(__file__).parent / "Images" / ALGORITHM_NAME


def generate_test_data_2d(dataset_type="blobs", n_samples=600):
    if dataset_type == "blobs":
        return make_blobs(n_samples=n_samples, centers=[[-4, 0], [0, 5], [4, 0]],
                          cluster_std=0.6, random_state=6767)
    if dataset_type == "moons":
        return make_moons(n_samples=n_samples, noise=0.05, random_state=6767)
    if dataset_type == "circles":
        return make_circles(n_samples=n_samples, factor=0.45, noise=0.03, random_state=6767)
    raise ValueError(dataset_type)


def generate_projected_3d(n_samples=900, seed=6767):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (n_samples, 3))
    y = np.repeat([0, 1, 2], n_samples // 3)
    for label, dims, center in ((0, [0, 1], [2, 2]), (1, [1, 2], [8, 3]), (2, [0, 2], [8, 8])):
        X[np.ix_(y == label, dims)] = rng.normal(center, 0.4, ((y == label).sum(), 2))
    return X, y


def run_strategy(points, **overrides):
    config = StrategiesManager.getStrategyRunConfigById(STRATEGY_ID)
    for name, value in overrides.items():
        config[name] = value
    labels = ConcreteStrategyPROCLUS().clastering_points(points, config)
    assert labels.shape == (len(points),)
    assert np.issubdtype(labels.dtype, np.integer)
    clusters = len(set(labels) - {-1})
    print(f"{ALGORITHM_NAME}: {len(points)} точек, {clusters} кластеров, "
          f"{np.count_nonzero(labels == -1)} выбросов")
    return labels


def save_figure(fig, filename):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {IMAGES_DIR / filename}")


def scatter(ax, points, labels, title):
    noise = labels == -1
    ax.scatter(*points[noise].T, c="lightgray", s=6)
    ax.scatter(*points[~noise].T, c=labels[~noise], cmap="tab10", s=10)
    if points.shape[1] == 3:
        ax.set_zlabel("X3")
    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, alpha=0.3)


def test_basic_2d_clustering():
    points, truth = generate_test_data_2d()
    labels = run_strategy(points, detect_outliers=False)
    score = adjusted_rand_score(truth, labels)
    print(f"ARI: {score:.4f}")
    assert score > 0.95
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scatter(axes[0], points, truth, "Истинные кластеры")
    scatter(axes[1], points, labels, f"{ALGORITHM_NAME}; ARI={score:.3f}")
    save_figure(fig, "test_basic_2d.png")


def test_parameter_sensitivity():
    points, truth = generate_test_data_2d()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (k, outliers) in zip(axes.ravel(), ((2, False), (3, False), (5, False), (3, True))):
        labels = run_strategy(points, n_clusters=k, detect_outliers=outliers)
        scatter(ax, points, labels, f"k={k}, выбросы={'да' if outliers else 'нет'}; "
                                    f"ARI={adjusted_rand_score(truth, labels):.2f}")
    save_figure(fig, "test_parameters.png")


def test_different_datasets():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, kind in zip(axes, ("blobs", "moons", "circles")):
        points, truth = generate_test_data_2d(kind)
        k = 3 if kind == "blobs" else 2
        labels = run_strategy(points, n_clusters=k, detect_outliers=False)
        scatter(ax, points, labels, f"{kind}; ARI={adjusted_rand_score(truth, labels):.2f}")
    save_figure(fig, "test_datasets.png")


def test_3d_clustering():
    points, truth = generate_projected_3d()
    model = PROCLUS(n_clusters=3, avg_dimensions=2, detect_outliers=False).fit(points)
    labels = model.labels_
    score = adjusted_rand_score(truth, labels)
    print(f"ARI: {score:.4f}; признаки кластеров: {[D.tolist() for D in model.dimensions_]}")
    assert score > 0.95
    assert {tuple(D) for D in model.dimensions_} == {(0, 1), (1, 2), (0, 2)}
    fig = plt.figure(figsize=(11, 5))
    scatter(fig.add_subplot(121, projection="3d"), points, truth, "Истинные кластеры")
    dims_text = ", ".join(str(tuple(D.tolist())) for D in model.dimensions_)
    scatter(fig.add_subplot(122, projection="3d"), points, labels, f"{ALGORITHM_NAME}; признаки: {dims_text}")
    save_figure(fig, "test_3d.png")


def test_strategy_integration():
    points, _ = generate_test_data_2d(n_samples=300)
    config = StrategiesManager.getStrategyRunConfigById(STRATEGY_ID)
    strategy = ConcreteStrategyPROCLUS()
    direct = strategy.clastering_points(points, config)
    contextual = Context(strategy).do_some_clustering_points(points.T, config)
    np.testing.assert_array_equal(direct, contextual)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scatter(axes[0], points, direct, "Strategy")
    scatter(axes[1], points, np.asarray(contextual), "Context")
    save_figure(fig, "test_strategy.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Сохранить PNG без графических окон")
    args = parser.parse_args()
    for test in (test_basic_2d_clustering, test_parameter_sensitivity,
                 test_different_datasets, test_3d_clustering, test_strategy_integration):
        test()
    print(f"Все 5 проверок {ALGORITHM_NAME} пройдены. Результаты: {IMAGES_DIR}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
