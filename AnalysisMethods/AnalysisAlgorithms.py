# Authors:
#       Nikolaev M. A. [Misha.via@yandex.ru]
#       Fedorov A. V. [alexis.sasis7@gmail.com]
#       Griban M. S. [gribanms007@gmail.com]

from typing import List

import numpy as np
from scipy.spatial import distance

def converter_to_c(points, labels) -> List:
    """
    @brief Преобразует список точек и меток в список кластеров.

    Параметры:
    points (List): Список точек данных.
    labels (List): Список меток кластеров для каждой точки.

    Возвращает:
    List: Список кластеров, где каждый кластер содержит свои точки.

    ИСПРАВЛЕНИЕ: Теперь корректно обрабатывает любые значения меток,
    включая отрицательные (шум) и с пропусками в нумерации.
    """
    # Преобразуем в numpy для удобства
    labels = np.array(labels)
    points = np.array(points)

    # Находим уникальные метки (включая шум/выбросы)
    unique_labels = np.unique(labels)

    # Создаем словарь для сопоставления меток с индексами
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Создаем список кластеров
    C = [[] for _ in range(len(unique_labels))]

    # Распределяем точки по кластерам
    for index, label in enumerate(labels):
        cluster_idx = label_to_index[label]
        C[cluster_idx].append(points[index])

    # Преобразуем списки в numpy массивы для удобства
    C = [np.array(cluster) if len(cluster) > 0 else np.array([]) for cluster in C]

    return C

#------------------------------------------------------------#

def MinInterCluster(C, i, j):
    """
    Вычисляет минимальное расстояние между кластерами i и j.

    Параметры:
        C (List): Список кластеров.
        i (int): Индекс первого кластера.
        j (int): Индекс второго кластера.

    Возвращает:
        float: Минимальное расстояние между кластерами i и j.
    """
    # Преобразуем в numpy массивы для векторизации
    cluster_i = np.asarray(C[i])
    cluster_j = np.asarray(C[j])
    
    # Если кластеры пустые, возвращаем большое значение
    if len(cluster_i) == 0 or len(cluster_j) == 0:
        return 100000.0
    
    # Вычисляем все попарные расстояния между кластерами
    # Используем broadcasting для эффективного вычисления
    # Расстояние между каждой точкой в cluster_i и каждой точкой в cluster_j
    distances = distance.cdist(cluster_i, cluster_j, metric='euclidean')
    
    # Находим минимальное расстояние
    mind = np.min(distances)
    return float(mind)


def MaxIntraCluster(C, i):
    """
    Вычисляет максимальное расстояние между точками внутри кластера i.

    Параметры:
        C (List): Список кластеров.
        i (int): Индекс кластера.

    Возвращает:
        float: Максимальное внутрикластерное расстояние.
    """    
    if len(C[i]) <= 1:
        return 0

    maxd = 0
    points = C[i]
    n_points = len(points)

    # Сравниваем только уникальные пары по индексам
    for idx1 in range(n_points):
        for idx2 in range(idx1 + 1, n_points):
            i1 = points[idx1]
            j1 = points[idx2]
            temp = distance.euclidean(i1, j1)
            if temp > maxd:
                maxd = temp
    return maxd


def DunnIndex(C):
    """
    Вычисляет индекс Данна для заданного набора кластеров.

    Параметры:
        C (List): Список кластеров.

    Возвращает:
        float: Значение индекса Данна.
    """
    mind = 100000
    maxd = 0
    for i in range (0, len(C), 1):
        for j in range (i+1, len(C), 1):
                temp = MinInterCluster(C, i, j)
                if mind > temp:
                    mind = temp
    for i in range (0, len(C), 1):
        temp = MaxIntraCluster(C,i)
        if temp > maxd:
            maxd = temp

    return(mind/maxd)

#------------------------------------------------------------#

def MeanInterclusterDistance(C,i,j):
    """
    Вычисляет среднее расстояние между кластерами i и j.

    Параметры:
        C (List): Список кластеров.
        i (int): Индекс первого кластера.
        j (int): Индекс второго кластера.

    Возвращает:
        float: Среднее межкластерное расстояние.
    """    
    # Преобразуем в numpy массивы для векторизации
    cluster_i = np.asarray(C[i])
    cluster_j = np.asarray(C[j])
    
    # Если кластеры пустые, возвращаем 0
    if len(cluster_i) == 0 or len(cluster_j) == 0:
        return 0.0
    
    # Вычисляем все попарные расстояния между кластерами
    distances = distance.cdist(cluster_i, cluster_j, metric='euclidean')
    
    # Вычисляем среднее расстояние
    MInter = np.mean(distances)
    return float(MInter)

def DunnIndexMean(C):
    """
    Вычисляет модифицированный индекс Данна с использованием среднего межкластерного расстояния.

    Параметры:
        C (List): Список кластеров.

    Возвращает:
        float: Значение модифицированного индекса Данна.
    """    
    mind = 100000
    maxd = 0
    for i in range (0, len(C), 1):
        for j in range (i+1, len(C), 1):
                temp = MeanInterclusterDistance(C,i,j)
                if mind > temp:
                    mind = temp
    for i in range (0, len(C), 1):
        temp = MaxIntraCluster(C,i)
        if temp > maxd:
            maxd = temp

    return(mind/maxd)

#------------------------------------------------------------#

# DBi
def normp(p, u, v):
    """
    Вычисляет норму порядка p между векторами u и v.

    Параметры:
        p (int): Порядок нормы.
        u (List): Первый вектор.
        v (List): Второй вектор.

    Возвращает:
        float: Значение нормы.
    """
    # Преобразуем в numpy массивы
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Вычисляем разность и норму через векторизацию
    diff = np.abs(u - v)
    sum_p = np.sum(diff ** p)
    return float(sum_p ** (1 / p))


def Mi(C, i):
    """
    Вычисляет центроид кластера i.

    Параметры:
        C (List): Список кластеров.
        i (int): Индекс кластера.

    Возвращает:
        List: Координаты центроида.
    """    
    # Преобразуем в numpy массив
    cluster = np.asarray(C[i])
    
    # Если кластер пустой, возвращаем нулевой центроид
    if len(cluster) == 0:
        return [0.0] * (len(C[i][0]) if len(C[i]) > 0 else 1)
    
    # Вычисляем среднее по каждой координате (центроид)
    mi = np.mean(cluster, axis=0)
    return mi.tolist()


def IntraclusterSeparation(C, i, p, q):
    """
    Вычисляет внутрикластерное рассеяние для кластера i.

    Параметры:
        C (List): Список кластеров.
        i (int): Индекс кластера.
        p (int): Порядок нормы для расстояния.
        q (int): Порядок нормы для суммирования.

    Возвращает:
        float: Внутрикластерное рассеяние.
    """    
    Norm = 1 / len(C[i])
    sum = 0

    M = Mi(C, i)
    for i1 in C[i]:
        sum += normp(p, i1, M) ** q
    sum *= Norm
    sum = sum ** (1 / q)
    return (sum)


def InterclusterSeparation(C, l, k, p):
    """
    Вычисляет расстояние между центроидами кластеров l и k.

    Параметры:
        C (List): Список кластеров.
        l (int): Индекс первого кластера.
        k (int): Индекс второго кластера.
        p (int): Порядок нормы.

    Возвращает:
        float: Межкластерное расстояние.
    """    
    u = Mi(C, l)
    v = Mi(C, k)
    res = normp(p, u, v)
    return (res)

def DBi(C,l,k,p,q):
    """
    Вычисляет индекс Дэвиса-Болдина между кластерами l и k.

    Параметры:
        C (List): Список кластеров.
        l (int): Индекс первого кластера.
        k (int): Индекс второго кластера.
        p (int): Порядок нормы для расстояния.
        q (int): Порядок нормы для суммирования.

    Возвращает:
        float: Значение индекса Дэвиса-Болдина.
    """    
    return((IntraclusterSeparation(C,l,p,q) + IntraclusterSeparation(C,k,p,q))/InterclusterSeparation(C,l,k,p))

# default:  DBi(C,0,1,1,1);

#------------------------------------------------------------#

