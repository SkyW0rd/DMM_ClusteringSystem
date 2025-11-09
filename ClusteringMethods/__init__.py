"""
ClusteringMethods Package - UPDATED with Correct BallTree v2.0

Автоматическая загрузка всех алгоритмов кластеризации
"""

# Импорт базовых классов и алгоритмов
from ClusteringMethods.ClasteringAlgorithms import *

# Импорт WaveClustering
try:
    from ClusteringMethods.WaveClusteringAlgorithm import (
        ConcreteStrategyWaveClustering,
        WaveClustering
    )
    print("✅ WaveClustering успешно загружен и зарегистрирован!")
except ImportError as e:
    print(f"⚠️ WaveClustering не загружен: {e}")
    print("   Убедитесь, что установлены: pip install PyWavelets scipy")

# Импорт BallTree Clustering (ИСПРАВЛЕННАЯ ВЕРСИЯ v2.0)
try:
    from ClusteringMethods.BallTreeClustering import (
        ConcreteStrategyBallTree,
        BallTreeClustering
    )
    print("✅ BallTree Clustering успешно загружен и зарегистрирован!")
except ImportError as e:
    print(f"⚠️ BallTree Clustering не загружен: {e}")
    print("   Убедитесь, что установлены: pip install scikit-learn scipy numpy")

# Экспортируем все
__all__ = [
    'Strategy',
    'StrategiesManager',
    'Context',
    'ConcreteStrategyWaveClustering',
    'WaveClustering',
    'ConcreteStrategyBallTree',
    'BallTreeClustering',
]