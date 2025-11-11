# Руководство по использованию системы кластеризации данных

**Автор:** Manee S. A. [mineeff20@yandex.ru]

**Последняя редакция:** Mysin Y.A. [yuriymysin@yandex.ru]

## 1 Установка

1. Скачайте архив с GitHub по ссылке:  
   https://github.com/SkyW0rd/DMM_ClusteringSystem.git
2. Распакуйте архив в выбранную директорию
3. Откройте проект в IDE
4. Создайте папку `env` на основном уровне проекта для окружения
5. Разверните окружение проекта
6. Установите зависимости
7. Запустите сборку проекта

## 2 Предварительное ознакомление с интерфейсом программы. Изменение настроек

Интерфейс (рис. 2.1) состоит из:
- Панели настроек кластеризации
- Области отображения результатов кластеризации

![2.1](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_2_1.png)

*Рис. 2.1 — Расположение основных областей интерфейса программы*

### Панель настроек

![2.2](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_2_2.png)

*Рис. 2.2 — Демонстрация расположения и названия окон на панели настроек*

### Заголовок интерфейса

В верхнем заголовке расположены:
- Иконка программы
- Имя программы
- Основные компоненты настройки интерфейса

![2.3](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_2_3.png)

*Рис. 2.3 — Основные компоненты программы*

### Настройки темы

В заголовке также имеется вкладка “Инструменты”, которая содержит активность “Настройки”, предназначенную для изменения настроек темы приложения.

![2.4](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_2_4.png)

*Рис. 2.4 — Выбор вкладки "Инструменты"*

В программе имеется две темы. Одну можно настроить в качестве светлой темы, а вторую в качестве темной. Настройки хранятся в Frameworks_interface/qss.

![2.5](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_2_5.png)

*Рис. 2.5 — Демонстрация параметров тем*

## 3 Генерация и кластеризация данных на основе распределений

1. Для генерации данных на основе распределений выберите в окне загрузки данных пункт “1. Сгенерировать и кластеризовать данные”.

![3.1](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_1.png)

*Рис. 3.1 — Выбор пункта “1. Сгенерировать и кластеризовать данные”

2. В окне "Генерация данных" выберите "1. Генерация распределений"

![3.2](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_2.png)

*Рис. 3.2 — Выбор типа генерации*

3. Задайте количество точек для генерации. В качестве примера на рис. 3.3 задается 100 точек.

![3.3](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_3.png)

*Рис. 3.3 — Указание количества точек*

4. Определите количество фич (метрик, осей), размерность которых будут иметь точки. На рис. 3.4 представлено в качестве примера три фичи.

![3.4](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_4.png)

*Рис. 3.4 — Указание размерности данных*

5. Для удобства мы уменьшим размеры окна и отделим окно “Генерация данных” (рис. 3.5).

![3.5](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_5.png)

*Рис. 3.5 — Отделение окна “Генерация данных”

6. Для каждой фичи задайте:
   - Тип распределения
   - Параметры распределения
   - Seed - фиксация генеративных данных (опционально)

На рис. 3.5–3.7 приводится пример установки для всех трех фич (начальная фича-1, шаг итерации-1, конечная фича-3) типа распределения “Нормальное”, без seed, с параметрами распределения: 
- loc(среднее, центр распределения) - 0.1
- scale(Стандартное отклонение) – 0.5

![3.6](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_6.png)

*Рис. 3.6 — Настройка параметров распределения*

![3.7](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_7.png)

*Рис. 3.7 — Продолжение настройки*

![3.8](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_8.png)

*Рис. 3.8 — Завершение настройки*

6. Генерация данных

После того, как будут заданы параметры распределений для всех фитч и добавлены в таблицу появится кнопка “Сгенерировать данные”. Для изменения параметров достаточно указать новые значения в полях, последовательно нажимая на “Далее” и зафиксировать в таблицу, нажав “Добавить запись”.

После того, как вы нажмете на кнопку “Сгенерировать данные”. Будут сгенерированы точки и отображены по первым трем измерениям в окне “Генерация данных”.

![3.9](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_9.png)

*Рис. 3.9 — Результат сгенерированных данных*

### Кластеризация данных

Затем в окне “Кластеризация данных” необходимо выбрать методы кластеризации, установив галочку на пересечении соответствующего метода и поля таблицы “0| used”, а также пожеланию выбрать у тех методов, где это возможно пункт “11| ccore”, который позволит задействовать реализацию вычисления с++ для ускорения процесса кластеризации.

![3.10](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_10.png)

*Рис. 3.10 — Установка параметров кластеризации*

Стоит отметить, что не все поля можно изменять. Неизменяемые поля отмечены серым и не подаются редактированию. Изменяемые поля (рис. 3.11) отмечены черным и могут быть отредактированы.

![3.11](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_11.png)

*Рис. 3.11 — Демонстрация изменяемых и неизменяемых полей таблицы задания параметров кластеризации*

Затем нажмите на кнопку “Провести кластеризацию”. На рис. 3.12 приводится результат откластеризованных данных на основе распределений.

![3.12](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_12.png)

*Рис. 3.12 — Демонстрация откластеризованных данных*

Вы можете открыть в отдельных подокнах результатов кластеризации методов развернуть под панель “параметры” и сравнить параметры (рис. 3.13).

![3.13](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_13.png)

*Рис. 3.13 — Демонстрация скрытых параметров*

![3.14](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_14.png)

*Рис. 3.14 — Демонстрация развернутых параметров*

Для отдельных методов программа позволяет сохранить результаты кластеризации в отдельные файлы. При этом стоит отметить, что сохранение 3D модели будет происходить в том состоянии, в котором вы повернули модель. При успешном сохранении появится статус “Ок” на соответствующей панели.

*Рис. 3.15 — Демонстрация сохранения данных для метода ROCK*

Выбор “Перестройки подокон” (рис. 3.16).

![3.16](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_16.png)

*Рис. 3.16 — Кнопка "перестроить подокна"*

![3.17](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_17.png)

*Рис. 3.17 – Демонстрация перестроенных подокон по принципу каскада*

На рис. 3.18–3.20 приводятся примеры отделения графических компонентов для более детального сравнения.

![3.18](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_18.png)

*Рис. 3.18 – Отделения 2D-модели откластеризованных данных*

![3.19](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_19.png)

*Рис. 3.19 – Отделения 3D-модели откластеризованных данных*

![3.20](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_3_20.png)

*Рис. 3.20 – Отделения таблиц для сравнения параметров откластеризованных данных*
## 4  Генерация и кластеризация данных на основе make-функций

1. Выберите "1. Сгенерировать и кластеризовать данные" (рис. 4.1)

![4.1](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_1.png)

*Рис. 4.1 — Выбор генерации данных*

2. В панели "Генерация данных" выберите "2. Генерация изображений" (рис. 4.2)

![4.2](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_2.png)

*Рис. 4.2 — Выбор типа генерации*

3. Далее на панели “Генерация данных” пункта “2. Генерация изображений” нажмите на кнопку “Добавить запись” для добавления make-функции и ее параметров в таблицу (рис 4.3).

![4.3](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_3.png)

*Рис. 4.3 — Кнопка "Добавить запись"*


Затем выберите в первом столбце записи make-функцию. В качестве нее вы можете выбрать следующие функции:

- **make_blobs** – позволяет сгенерировать данные в виде разбросанных точек.

- **make_circle** – позволяет сгенерировать данные в виде точек выстроенных в две окружности, одна из которых включена в другую.

- **make_moons** – позволяет сгенерировать данные в виде точек распределенных по контуру натянутого лука(лук, которым стреляют).

- **make_dna** – позволяет сгенерировать данные в виде точек спирали.

- **make_spheres** - позволяет сгенерировать данные в виде точек сферы.

![4.4](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_4.png)

*Рис. 4.4 — Результаты генерации данных с помощью make-функций*

![4.5](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_5.png)

*Рис. 4.5 — Выбор make-функций*

Для удаления записи из таблицы достаточно одним или двойным щелчком мыши выбрать строку(запись) таблицы, которую требуется удалить и нажав соответствующую кнопку.

![4.6](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_6.png)

*Рис. 4.6 — Выделение второй строки (записи)*

![4.7](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_7.png)

*Рис. 4.7 — Удаление записи*

![4.8](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_8.png)

*Рис. 4.8 — Результат удаления make-функции (второй записи)*

Следует отметить, что одно изображение является результатом комбинации данных, сгенерированных на основе таблицы make-функций. Так же отметим, что make-функции делятся на 2D-make-функции и 3D-make-функции. К 2D-make-функциям относятся: make_blobs, make_circles и  make_moons. Остальные относятся к 3D-make-функциям. Если задать сначала 2D-make-функцию, а потом, 3D-make-функцию, то последняя перекроек данные предыдущей функции по причине реализации в коде перезаписи, которая в свою очередь сделана по причине особенности работы np.array.

![4.9](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_9.png)

*Рис. 4.9 – Перекрытие данных 2D-make-функции 3D-make-функцией*

Для решения возникшей проблемы достаточно сначала задать 3D-make-функции, а затем 2D-make-функции.

![4.10](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_10.png)

*Рис. 4.10 – Успешная кластеризация комбинации 2D и 3D make-функций*

Те или иные параметры становятся изменяемыми/неизменяемыми, в зависимости от выбранных make-функций в таблице. Неизменяемые выделены серым (рис. 4.11). Изменяемые Черным.

![4.11](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_11.png)

*Рис. 4.11 – Изменяемые и неизменяемые параметры функций*

![4.12](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_4_12.png)

*Рис. 4.12 – Результат кластеризации данных*

## 5 Загрузка и кластеризация изображений

1. В панели "Загрузка данных" выберите "2. Загрузить данные" (рис 5.1)

![5.1](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_1.png)

*Рис. 5.1 – Выбор загрузки данных*

2. Нажмите "Выбрать" (рис. 5.2) и выберите изображение (рис. 5.3)

![5.2](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_2.png)

*Рис. 5.2 — Кнопка "Выбрать"*

Стоит отметить, что в зависимости от качества и размера изображения скорость кластеризации может меняться.

![5.3](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_3.png)

*Рис. 5.3 — Загрузка изображения*

![5.4](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_4.png)

*Рис. 5.4 — Кнопка "Открыть" для использвания изображения в программе*

3. Установите параметры кластеризации (рис. 5.5)

![5.5](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_5.png)

*Рис. 5.5 — Установка параметров кластеризации изображения*

![5.6](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_6.png)

*Рис. 5.6 — Выбора типа конвертации изображения*

![5.7](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_7.png)

*Рис. 5.7 — Результаты кластеризации изображения*

![5.8](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_5_8.png)

*Рис. 5.8 — Результаты кластеризации изображения в каскадном стиле подокон*

## 6 Удаление программы

1. Удалите директорию с проектом
2. Очистите реестр Windows:
   - Нажмите Win+R
   - Введите `regedit`
   - Нажмите "Ок"
   - Перейдите по пути: `HKEY_CURRENT_USER\SOFTWARE\RTU_MIREA` (рис. 6.2)
   - Удалите папку `ClustSystem`

![6.1](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_6_1.png)

*Рис. 6.1 — Вход в реестр* на ОС windows

![6.2](https://github.com/SkyW0rd/DMM_ClusteringSystem/blob/docs-update/Documentation/Programming_documentation/docs_images/uses_cases_6_2.png)

*Рис. 6.2 — Расположение данных проекта в реестре*