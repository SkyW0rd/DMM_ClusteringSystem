"""
ClusteringMethods Package - UPDATED with VP-Tree Bregman (БЕЗ pyBregman)

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

# Импорт BallTree Clustering
try:
    from ClusteringMethods.BallTreeClustering import (
        ConcreteStrategyBallTree,
        BallTreeClustering
    )
    print("✅ BallTree Clustering успешно загружен и зарегистрирован!")
except ImportError as e:
    print(f"⚠️ BallTree Clustering не загружен: {e}")
    print("   Убедитесь, что установлены: pip install scikit-learn scipy numpy")

# ============================================================================
# Импорт VP-Tree Bregman Clustering (БЕЗ pyBregman!)
# ============================================================================
try:
    from ClusteringMethods.VPTreeClustering import (
        ConcreteStrategyVPTreeBregman,
        VPTreeBregmanClustering
    )
    print("✅ VP-Tree Bregman Clustering успешно загружен и зарегистрирован!")
except ImportError as e:
    print(f"⚠️ VP-Tree Bregman Clustering не загружен: {e}")
    print("   Убедитесь, что установлены: pip install scipy scikit-learn numpy")

try:
    from ClusteringMethods.HPStreamClasteringAlgorithm import (
        ConcreteStrategyHPStream,
        HPStreamClustering
    )
    print("✅ HPStreamClustering успешно загружен и зарегистрирован!")
except ImportError as e:
    print(f"⚠️  HPStreamClustering не загружен: {e}")
    print("   Убедитесь, что установлены: pip install numpy scikit-learn")

# Импорт KMeans с PCA
try:
    from ClusteringMethods.KMeansClustering import (
        ConcreteStrategyKMeans_from_SKLEARN
    )
    print("✅ KMeans (SKLearn) успешно загружен и зарегистрирован!")
except ImportError as e:
    print(f"⚠️  KMeans (SKLearn) не загружен: {e}")
    print("   Убедитесь, что установлены: pip install scikit-learn numpy")

# Экспортируем все
__all__ = [
    'Strategy',
    'StrategiesManager',
    'Context',
    'ConcreteStrategyWaveClustering',
    'WaveClustering',
    'ConcreteStrategyBallTree',
    'BallTreeClustering',
    'ConcreteStrategyVPTreeBregman',
    'VPTreeBregmanClustering',
    'ConcreteStrategyHPStream',
    'HPStreamClustering',
    'ConcreteStrategyKMeans_from_SKLEARN',
]
