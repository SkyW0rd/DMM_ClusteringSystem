"""
ClusteringMethods Package
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
    print(f"⚠️  WaveClustering не загружен: {e}")
    print("   Убедитесь, что установлены: pip install PyWavelets scipy")

# Экспортируем все
__all__ = [
    'Strategy',
    'StrategiesManager',
    'Context',
    'ConcreteStrategyWaveClustering',
    'WaveClustering'
]
