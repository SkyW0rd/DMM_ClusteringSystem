# Изменено:
  - Добалена более обширная документация для всех функций


# Документация по методам кластеризации

## Авторы
- **Mineev S. A.** [mineeff20@yandex.ru]
- **Meshkova O. V.** [oxn.lar5@yandex.ru]
- **Kozlov A. I.** [alex_kozlov15@mail.ru]

Ниже представлены методы кластеризации, используемые в проекте версии v1.1. Включены как подробные описания параметров, так и практические примеры использования, чтобы сделать работу с кодом более понятной и легкой для дальнейшего улучшения.

### 1. BIRCH из библиотеки scikit-learn

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import Birch
```

**Описание метода:**
Метод BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) из библиотеки scikit-learn используется для кластеризации больших наборов данных, работая с ограниченными вычислительными ресурсами.

**Сигнатура:**
```python
class sklearn.cluster.Birch(
    *, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True
)
```

**Параметры:**
- **threshold** (*float*, по умолчанию 0.5): Радиус подкластера, полученный путем слияния нового образца и ближайшего подкластерного центра. Если радиус превышает значение порога, создается новый подкластер. Установка низкого значения способствует более дробному делению кластеров.

- **branching_factor** (*int*, по умолчанию 50): Максимальное количество подкластеров (CF) в каждом узле. Если количество подкластеров превышает значение этого параметра, узел разделяется на два, что приводит к перераспределению подкластеров между ними.

- **n_clusters** (*int, экземпляр модели sklearn.cluster или None*, по умолчанию 3): Количество кластеров на завершающем этапе. Листовые подгруппы рассматриваются как новые точки данных.

- **compute_labels** (*bool*, по умолчанию True): Определяет, вычислять ли метки для каждого объекта данных.

- **copy** (*bool*, по умолчанию True): Определяет, следует ли создавать копию входных данных. Если False, исходные данные будут изменены.


**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)

---

### 2. BIRCH из библиотеки pyclustering

**Установка:**
```bash
python -m pip install pyclustering
```
**Импорт:**
```python
from pyclustering.cluster.birch import birch
from pyclustering.container.cftree import measurement_type
```

**Описание метода:**
Метод BIRCH из библиотеки pyclustering используется для создания CF-дерева и дальнейшей кластеризации данных с использованием заданных параметров.

**Сигнатура:**
```python
birch(
    data, number_clusters=3, branching_factor=50, max_node_entries=3,
    diameter=0.5, type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
    entry_size_limit=500, diameter_multiplier=1.5, ccore=True
)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек, каждая из которых является списком координат.

- **number_clusters** (*int*, по умолчанию 3): Количество кластеров, которые нужно выделить.

- **branching_factor** (*int*, по умолчанию 50): Максимальное количество подгрупп (CF) в каждом нелистовом узле CF-дерева.

- **max_node_entries** (*int*, по умолчанию 200): Максимальное количество элементов, которые могут содержаться в каждом листовом узле CF-дерева.

- **diameter** (*float*, по умолчанию 0.5): Диаметр CF-входного узла, используемый для построения дерева. Может увеличиваться, если превышен лимит элементов в узле.

- **type_measurement** (*measurement_type*, по умолчанию `measurement_type.CENTROID_EUCLIDEAN_DISTANCE`): Тип метрики, используемой для вычисления расстояний.

- **entry_size_limit** (*int*, по умолчанию 500): Максимальное количество записей, которое может быть в дереве. Если лимит превышен, диаметр увеличивается, и дерево перестраивается.

- **diameter_multiplier** (*float*, по умолчанию 1.5): Множитель для увеличения диаметра при превышении лимита записей.

- **ccore** (*bool*, по умолчанию True): Если True, то для обработки используется часть библиотеки C++.


**Источник:** [Документация pyclustering](https://pyclustering.github.io/docs/0.10.1/html/d6/d00/classpyclustering_1_1cluster_1_1birch_1_1birch.html)

---

### 3. CURE из библиотеки pyclustering

**Установка:**
```bash
python -m pip install pyclustering
```
**Импорт:**
```python
from pyclustering.cluster.cure import cure
```

**Описание метода:**
Метод CURE (Clustering Using Representatives) используется для уменьшения размеров кластеров путем сжатия выбранных репрезентативных точек.

**Сигнатура:**
```python
cure(data, number_clusters=3, number_represent_points=5, compression=0.5, ccore=True)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек (объектов), где каждая точка - это список координат.

- **number_clusters** (*int*, по умолчанию 3): Количество кластеров, которые должны быть выделены.

- **number_represent_points** (*int*, по умолчанию 5): Количество репрезентативных точек для каждого кластера.

- **compression** (*float*, по умолчанию 0.5): Коэффициент, определяющий уровень сжатия точек представления к центроиду нового кластера после объединения.

- **ccore** (*bool*, по умолчанию True): Если True, то для обработки используется часть библиотеки C++.


**Источник:** [Документация pyclustering](https://pyclustering.github.io/docs/0.10.0/html/dc/d6d/classpyclustering_1_1cluster_1_1cure_1_1cure.html)

---

### 4. ROCK из библиотеки pyclustering

**Установка:**
```bash
python -m pip install pyclustering
```
**Импорт:**
```python
from pyclustering.cluster.rock import rock
```

**Описание метода:**
Метод ROCK (Robust Clustering using Links) используется для кластеризации категориальных данных и основан на количестве "связей" между точками.

**Сигнатура:**
```python
rock(data, eps, number_clusters=3, threshold=0.5, ccore=True)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек, где каждая точка - это список координат.

- **eps** (*float*): Радиус связности. Точки являются соседними, если расстояние между ними меньше заданного радиуса.

- **number_clusters** (*int*, по умолчанию 3): Количество кластеров, которые должны быть выделены.

- **threshold** (*float*, по умолчанию 0.5): Порог нормализации, влияющий на выбор кластеров для объединения.

- **ccore** (*bool*, по умолчанию True): Определяет, должен ли CCORE (библиотека C ++ для пикластеринга) использоваться вместо кода Python или нет.


**Источник:** [Документация pyclustering](https://pyclustering.github.io/docs/0.8.2/html/d8/dde/classpyclustering_1_1cluster_1_1rock_1_1rock.html)

---


### 5. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import DBSCAN
```

**Описание метода:**
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) — это алгоритм кластеризации, основанный на плотности, который выделяет кластеры по плотности точек и позволяет эффективно работать с шумовыми данными.

**Сигнатура:**
```python
class sklearn.cluster.DBSCAN(
    eps=0.5, min_samples=5, metric='euclidean',
    algorithm='auto', leaf_size=30, p=2, n_jobs=None
)
```

**Параметры:**
- **eps** (*float*, по умолчанию 0.5): Максимальное расстояние между точками, чтобы считать их соседями.
- **min_samples** (*int*, по умолчанию 5): Минимальное число точек для формирования плотного кластера.
- **metric** (*str*, по умолчанию 'euclidean'): Метрика расстояния между точками.
- **algorithm** (*str*, по умолчанию 'auto'): Алгоритм расчета: 'auto', 'ball_tree', 'kd_tree', 'brute'.
- **leaf_size** (*int*, по умолчанию 30): Параметр для оптимизации скорости деревьев.
- **p** (*float*, по умолчанию 2): Степень метрики Minkowski.
- **n_jobs** (*int*, по умолчанию None): Число потоков (-1 для использования всех доступных).

**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

---

### 6. HDBSCAN (Hierarchical DBSCAN)

**Установка:**
```bash
python -m pip install hdbscan
```
**Импорт:**
```python
from hdbscan import HDBSCAN
```

**Описание метода:**
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) расширяет DBSCAN для работы с иерархическими структурами кластеров и автоматического определения количества кластеров.

**Сигнатура:**
```python
class hdbscan.HDBSCAN(
    min_cluster_size=5, metric='euclidean', cluster_selection_method='eom'
)
```

**Параметры:**
- **min_cluster_size** (*int*, по умолчанию 5): Минимальное количество точек в кластере.
- **metric** (*str*, по умолчанию 'euclidean'): Метрика для расчета расстояний.
- **cluster_selection_method** (*str*, по умолчанию 'eom'): Метод выбора кластеров ('eom' или 'leaf').

**Источник:** [Документация HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)

---

### 7. GMM (Gaussian Mixture Model)

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.mixture import GaussianMixture
```

**Описание метода:**
GMM (Gaussian Mixture Model) использует подходы вероятностного моделирования для определения кластеров, предполагая, что данные распределены как смесь нескольких гауссианов.

**Сигнатура:**
```python
class sklearn.mixture.GaussianMixture(
    n_components=1, covariance_type='full', tol=0.001, max_iter=100
)
```

**Параметры:**
- **n_components** (*int*, по умолчанию 1): Количество гауссианов.
- **covariance_type** (*str*, по умолчанию 'full'): Тип ковариации ('full', 'tied', 'diag', 'spherical').
- **tol** (*float*, по умолчанию 0.001): Порог для завершения итераций.
- **max_iter** (*int*, по умолчанию 100): Максимальное количество итераций.

**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

---

### 8. OPTICS (Ordering Points To Identify the Clustering Structure)

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import OPTICS
```

**Описание метода:**
OPTICS — это алгоритм кластеризации, который упорядочивает точки так, чтобы можно было определить иерархические кластеры.

**Сигнатура:**
```python
class sklearn.cluster.OPTICS(
    min_samples=5, max_eps=np.inf, metric='minkowski',
    p=2, cluster_method='xi', eps=np.inf
)
```

**Параметры:**
- **min_samples** (*int*, по умолчанию 5): Минимальное число точек в кластере.
- **max_eps** (*float*, по умолчанию np.inf): Максимальное расстояние для кластеризации.
- **metric** (*str*, по умолчанию 'minkowski'): Метрика для расчета расстояния.
- **p** (*float*, по умолчанию 2): Степень метрики Minkowski.
- **cluster_method** (*str*, по умолчанию 'xi'): Метод кластеризации ('xi' или 'dbscan').
- **eps** (*float*, по умолчанию np.inf): Максимальное расстояние между точками для включения в кластер.

**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)

---

### 9. Spectral Biclustering

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import SpectralBiclustering
```

**Описание метода:**
Spectral Biclustering — это метод, который находит кластеры как по строкам, так и по столбцам матрицы, выделяя скрытые структуры данных.

**Сигнатура:**
```python
class sklearn.cluster.SpectralBiclustering(
    n_clusters=3, method='bistochastic', n_components=6
)
```

**Параметры:**
- **n_clusters** (*int*, по умолчанию 3): Количество кластеров.
- **method** (*str*, по умолчанию 'bistochastic'): Метод обработки ('bistochastic', 'scale', 'log').
- **n_components** (*int*, по умолчанию 6): Количество компонент для анализа.

**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html)

---

### 10. BANG (Bounding And Nesting Grid)

**Установка:**
```bash
python -m pip install pyclustering
```
**Импорт:**
```python
from pyclustering.cluster.bang import bang
```

**Описание метода:**
BANG использует иерархическую сеточную структуру для быстрого поиска и выделения кластеров в многомерных данных.

**Сигнатура:**
```python
bang(data, levels=15)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек.
- **levels** (*int*, по умолчанию 15): Глубина иерархии сетки.

**Источник:** [Документация pyclustering](https://pyclustering.github.io/docs/0.10.1/html/da/d85/classpyclustering_1_1cluster_1_1bang_1_1bang.html)

---

### 11. Fuzzy C-means

**Установка:**
```bash
python -m pip install scikit-fuzzy
```
**Импорт:**
```python
from skfuzzy.cluster import cmeans
```

**Описание метода:**
Метод нечёткой кластеризации C-средних позволяет разбить имеющееся множество элементов мощностью *N* на заданное число нечётких множеств *k*.

**Сигнатура:**
```python
cmeans(data, 
        c=2,
        m=2.0,
        error=0.005,
        maxiter=1000)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек.
- **c** (*int*, по умолчанию 2): Желаемое количество кластеров.
- **m**(*float*, по умолчанию 2.0): Фактор фаззификации.
- **error**(*float*, по умолчанию 0.005): Критерий остановки; остановка досрочная, если норма (u[p] - u[p-1]) < ошибка.
- **maxiter**(*int*, по умолчанию 1000): Максимальное разрешенное количество итераций.

**Источник:** [Документация scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html#cmeans)


---

### 12. Spectral Clustering

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import SpectralClustering
```

**Описание метода:**
Spectral Сlustering —  это метод кластеризации, который использует методы спектральной теории графов для выделения кластеров в данных. 

**Сигнатура:**
```python
class sklearn.cluster.SpectralСlustering(
   n_clusters=3, affinity='rbf', gamma=1.0, n_neighbors=10, eigen_solver=None, random_state=0
)
```

**Параметры:**
- **n_clusters** (*int*, по умолчанию 3): Количество кластеров.
- **affinity** (*str*, по умолчанию 'rbf'): Функция сходства ('rbf', 'nearest_neighbors', 'precomputed', 'precomputed_nearest_neighbors').
- **gamma** (*float*, по умолчанию 1.0): Параметр gamma для ядра RBF. Используется, если affinity='rbf'.
- **n_neighbors** (*int*, по умолчанию 10): Количество соседей для affinity='nearest_neighbors'.
- **eigen_solver** (*str*, по умолчанию None): Решатель собственных значений ('None','arpack', 'lobpcg', 'amg').
- **random_state** (*int*, по умолчанию 0): Инициализация генератора случайных чисел..

**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

---

### 13. WaveClustering (Кластеризация на основе вейвлет-преобразования)

**Установка:**
```bash
python -m pip install PyWavelets scipy
```
**Импорт:**
```python
from ClusteringMethods.WaveClusteringAlgorithm import WaveClustering
```

**Описание метода:**
WaveClustering использует вейвлет-преобразование для анализа пространства признаков на разных масштабах. Алгоритм квантует данные в сетку, применяет вейвлет-преобразование для выделения плотных областей на различных уровнях разрешения, а затем находит связные компоненты в преобразованном пространстве. Это позволяет обнаруживать кластеры различного размера и формы, используя многомасштабный анализ.

**Сигнатура:**
```python
class WaveClustering(
    n_grid=32, wavelet='haar', n_levels=2, density_threshold=0.1
)
```

**Параметры:**
- **n_grid** (*int*, по умолчанию 32): Количество делений каждого измерения на ячейки сетки. Большие значения дают более детальное разбиение, но требуют больше памяти.

- **wavelet** (*str*, по умолчанию 'haar'): Тип вейвлет-функции для преобразования. Доступные варианты: 'haar' (самый простой и быстрый), 'db2', 'db4', 'db6' (Добеши различных порядков), 'coif1' (Койфлет 1-го порядка).

- **n_levels** (*int*, по умолчанию 2): Количество уровней вейвлет-разложения (масштабы кластеров). Большие значения дают более крупные кластеры.

- **density_threshold** (*float*, по умолчанию 0.1): Порог для определения плотных областей (доля от максимальной плотности). Значения от 0 до 1. Меньшие значения дают больше кластеров.

**Источник:** `ClusteringMethods/WaveClusteringAlgorithm.py`

---

### 14. BallTree Hierarchical Clustering (Иерархическая кластеризация с использованием BallTree)

**Установка:**
```bash
python -m pip install scikit-learn scipy
```
**Импорт:**
```python
from ClusteringMethods.BallTreeClustering import BallTreeClustering
```

**Описание метода:**
Метод объединяет структуру данных BallTree с иерархической кластеризацией на основе матрицы попарных расстояний. Алгоритм вычисляет полную матрицу расстояний между всеми точками, строит иерархическое дерево кластеров с использованием методов linkage (ward, complete, average, single), а затем разрезает дерево на заданное количество кластеров. Нормализация данных является критически важной для корректной работы алгоритма.

**Сигнатура:**
```python
class BallTreeClustering(
    n_neighbors=5, leaf_size=40, metric='euclidean',
    linkage_method='ward', n_clusters=4, normalize=True
)
```

**Параметры:**
- **n_clusters** (*int*, по умолчанию 4): Желаемое количество кластеров. В отличие от DBSCAN, здесь обязательно указывать количество.

- **linkage_method** (*str*, по умолчанию 'ward'): Метод объединения кластеров в иерархии. Доступные варианты: 'ward' (минимизирует дисперсию, рекомендуется), 'complete' (максимальное расстояние), 'average' (среднее расстояние), 'single' (минимальное расстояние).

- **metric** (*str*, по умолчанию 'euclidean'): Метрика для расчета расстояния. Доступные варианты: 'euclidean' (евклидово, рекомендуется), 'manhattan' (манхэттенское), 'chebyshev' (максимальная норма).

- **n_neighbors** (*int*, по умолчанию 5): Зарезервировано для совместимости. В текущей версии не используется (используется полная матрица расстояний).

- **leaf_size** (*int*, по умолчанию 40): Зарезервировано для совместимости. В текущей версии не используется.

- **normalize** (*bool*, по умолчанию True): Нормализировать данные перед кластеризацией. Критически важно для корректной работы алгоритма.

**Источник:** `ClusteringMethods/BallTreeClustering.py`

---

### 15. HPStream (Проекционная кластеризация потоковых данных)

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from ClusteringMethods.HPStreamClasteringAlgorithm import HPStreamClustering
```

**Описание метода:**
HPStream (High-dimensional Projected Stream clustering) предназначен для кластеризации высокомерных потоковых данных. Алгоритм автоматически определяет значимые размерности для каждого кластера, используя адаптивные веса признаков, и выполняет кластеризацию только в подпространствах релевантных признаков. Метод поддерживает инкрементальное обучение на новых батчах данных с использованием скользящего окна, что делает его пригодным для потоковой обработки.

**Сигнатура:**
```python
class HPStreamClustering(
    num_clusters=5, projection_threshold=0.1,
    window_size=1000, fade_threshold=0.1, normalize=True
)
```

**Параметры:**
- **num_clusters** (*int*, по умолчанию 5): Желаемое количество кластеров.

- **projection_threshold** (*float*, по умолчанию 0.1): Порог для определения значимых размерностей. Размерности с весом выше этого порога считаются значимыми для кластеризации.

- **window_size** (*int*, по умолчанию 1000): Размер окна для потоковых данных. Определяет количество последних точек, которые учитываются при обновлении модели.

- **fade_threshold** (*float*, по умолчанию 0.1): Коэффициент затухания весов размерностей. Контролирует скорость адаптации весов к новым данным.

- **max_iterations** (*int*, по умолчанию 100): Максимальное количество итераций алгоритма кластеризации.

- **normalize** (*bool*, по умолчанию True): Нормализировать входные данные перед кластеризацией.

**Источник:** `ClusteringMethods/HPStreamClasteringAlgorithm.py`

---

### 16. KMeans (KMeans с опциональной PCA декомпозицией)

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

**Описание метода:**
Реализация классического алгоритма KMeans из библиотеки scikit-learn с дополнительной возможностью применения PCA (Principal Component Analysis) для снижения размерности данных перед кластеризацией. Алгоритм итеративно обновляет центроиды кластеров, минимизируя сумму квадратов расстояний от точек до ближайших центров. PCA позволяет работать с данными высокой размерности, сохраняя основную вариативность данных.

**Сигнатура:**
```python
class sklearn.cluster.KMeans(
    n_clusters=3, init='k-means++', n_init=10, max_iter=300,
    random_state=42
)
```

**Параметры:**
- **n_clusters** (*int*, по умолчанию 3): Желаемое количество кластеров для KMeans. Примеры: 2, 3, 4, 5...

- **init** (*str*, по умолчанию 'k-means++'): Метод инициализации центроидов. Доступные варианты: 'k-means++' (умная инициализация, рекомендуется), 'random' (случайная инициализация).

- **n_init** (*int*, по умолчанию 10): Количество различных инициализаций для выбора лучшего результата. Больше значений = лучше результат, но медленнее. Рекомендуется: 10-20.

- **max_iter** (*int*, по умолчанию 300): Максимальное количество итераций алгоритма KMeans. Рекомендуется: 300-500.

- **random_state** (*int*, по умолчанию 42): Инициализация генератора случайных чисел для воспроизводимости результатов.

- **use_pca** (*bool*, по умолчанию True): Включить/выключить PCA декомпозицию перед кластеризацией. Если False, используется только стандартизация данных.

- **n_components** (*int*, по умолчанию 0): Количество главных компонент для PCA декомпозиции. Если 0, используется автоматический выбор (min(n_samples, n_features)). Рекомендуется: 2-3 для визуализации, больше для сохранения информации.

**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) и `ClusteringMethods/KMeansClustering.py`

---

### 17. MiniBatchKMeans (Мини-батч версия KMeans)

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

**Описание метода:**
MiniBatchKMeans является оптимизированной версией KMeans, которая обрабатывает данные небольшими батчами вместо использования всего набора данных на каждой итерации. Это значительно ускоряет обучение, особенно на больших наборах данных, при сохранении приемлемого качества кластеризации. Алгоритм обновляет центроиды на основе мини-батчей, что делает его пригодным для потоковой обработки и работы с данными, которые не помещаются в память.

**Сигнатура:**
```python
class sklearn.cluster.MiniBatchKMeans(
    n_clusters=3, init='k-means++', max_iter=100, batch_size=100,
    verbose=False, compute_labels=True, random_state=42, reassignment_ratio=0.01
)
```

**Параметры:**
- **n_clusters** (*int*, по умолчанию 3): Желаемое количество кластеров для MiniBatchKMeans. Примеры: 2, 3, 4, 5...

- **init** (*str*, по умолчанию 'k-means++'): Метод инициализации центроидов. Доступные варианты: 'k-means++' (умная инициализация, рекомендуется), 'random' (случайная инициализация).

- **max_iter** (*int*, по умолчанию 100): Максимальное количество итераций алгоритма MiniBatchKMeans. Рекомендуется: 100-300.

- **batch_size** (*int*, по умолчанию 100): Размер мини-батча для обработки данных. Большие значения дают более точные результаты, но медленнее. Рекомендуется: 100-1000.

- **verbose** (*bool*, по умолчанию False): Включить/выключить подробный вывод процесса кластеризации.

- **compute_labels** (*bool*, по умолчанию True): Вычислять метки для всех точек после обучения. Если False, метки будут вычисляться только при необходимости.

- **random_state** (*int*, по умолчанию 42): Инициализация генератора случайных чисел для воспроизводимости результатов.

- **reassignment_ratio** (*float*, по умолчанию 0.01): Коэффициент перераспределения центроидов. Контролирует частоту перераспределения центров кластеров. Рекомендуется: 0.01-0.1.

- **use_pca** (*bool*, по умолчанию True): Включить/выключить PCA декомпозицию перед кластеризацией. Если False, используется только стандартизация данных.

- **n_components** (*int*, по умолчанию 0): Количество главных компонент для PCA декомпозиции. Если 0, используется автоматический выбор (min(n_samples, n_features)). Рекомендуется: 2-3 для визуализации, больше для сохранения информации.

**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) и `ClusteringMethods/MiniBatchKMeansClustering.py`

---

### 18. VP-Tree Bregman Divergence Clustering (Кластеризация на основе VP-tree с дивергенциями)

**Установка:**
```bash
python -m pip install scikit-learn scipy
```
**Импорт:**
```python
from ClusteringMethods.VPTreeClustering import VPTreeBregmanClustering
```

**Описание метода:**
Метод объединяет концепцию VP-tree (Vantage Point Tree) с иерархической кластеризацией, используя различные типы дивергенций и метрик расстояния для вычисления матрицы попарных расстояний. Алгоритм поддерживает евклидову дивергенцию, дивергенции Хеллингера и Бхаттачарьи (полезны для вероятностных данных), а также метрики Манхэттена, Чебышёва и косинусное расстояние. На основе вычисленной матрицы строится иерархическое дерево кластеров.

**Сигнатура:**
```python
class VPTreeBregmanClustering(
    divergence='euclidean', linkage_method='ward', n_clusters=4,
    normalize=True, max_depth=10
)
```

**Параметры:**
- **n_clusters** (*int*, по умолчанию 4): Желаемое количество кластеров для разбиения. Примеры: 2, 3, 4, 5...

- **divergence** (*str*, по умолчанию 'euclidean'): Выбор метрики/дивергенции для измерения расстояния. Доступные варианты: 'euclidean' (евклидова дивергенция, рекомендуется), 'hellinger' (дивергенция Хеллингера для вероятностей), 'bhattacharyya' (дивергенция Бхаттачарьи, статистическое расстояние), 'manhattan' (манхэттенское расстояние), 'chebyshev' (максимальная норма), 'cosine' (косинусное расстояние).

- **linkage_method** (*str*, по умолчанию 'ward'): Метод для связывания кластеров в иерархии. Доступные варианты: 'ward' (минимизирует дисперсию, рекомендуется), 'complete' (максимальное расстояние между кластерами), 'average' (среднее расстояние между кластерами), 'single' (минимальное расстояние между кластерами).

- **normalize** (*bool*, по умолчанию True): Нормализировать входные данные перед кластеризацией.

- **max_depth** (*int*, по умолчанию 10): Максимальная глубина дерева VP-tree. Зарезервировано для будущих версий.

**Источник:** `ClusteringMethods/VPTreeClustering.py`

---
