"""Итерационные методы для линейных систем."""

import math

import numpy as np


def backward_substitution(upper, d):
    """Решает верхнюю треугольную систему ux=d.

    Args:
        upper (numpy.ndarray): верхнетреугольная матрица.
        d (numpy.ndarray): значения d.

    Returns:
        x (numpy.ndarray): решение линейной системы.
    """
    [n, m] = upper.shape
    b = d.astype(float)

    if n != m:
        raise ValueError("'upper' должна быть квадратной матрицей.")

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if upper[i, i] == 0:
            raise ValueError("'upper' - вырожденная матрица.")

        x[i] = b[i] / upper[i, i]
        b[0:i] = b[0:i] - upper[0:i, i] * x[i]

    return x


def forward_substitution(lower, c):
    """Решает нижнюю треугольную систему lx=c.

    Args:
        lower (numpy.ndarray): нижнетреугольная матрица.
        c (numpy.ndarray): значения c.

    Returns:
        x (numpy.ndarray): решение линейной системы.
    """
    [n, m] = lower.shape
    b = c.astype(float)

    if n != m:
        raise ValueError("'lower' должна быть квадратной матрицей.")

    x = np.zeros(n)

    for i in range(0, n):
        if lower[i, i] == 0:
            raise ValueError("'lower' - вырожденная матрица.")

        x[i] = b[i] / lower[i, i]
        b[i + 1:n] = b[i + 1:n] - lower[i + 1:n, i] * x[i]

    return x


def gauss_elimination_pp(a, b):
    """Метод Гаусса с выбором главного элемента.

    Вычисляет верхнетреугольную матрицу из системы Ax=b (делает строковое
    преобразование).

    Args:
        a (numpy.ndarray): матрица A из системы Ax=b.
        b (numpy.ndarray): значения b.

    Returns:
        a (numpy.ndarray): расширенная верхнетреугольная матрица.
    """
    [n, m] = a.shape

    if n != m:
        raise ValueError("'a' должна быть квадратной матрицей.")

    # Создаем расширенную матрицу
    a = np.concatenate((a, b[:, None]), axis=1).astype(float)

    # Начинаем процесс устранения
    for i in range(0, n - 1):
        p = i

        # Сравниваем, чтобы выбрать главный элемент
        for j in range(i + 1, n):
            if math.fabs(a[j, i]) > math.fabs(a[i, i]):
                # Меняем строки местами
                a[[i, j]] = a[[j, i]]

        # Проверяем на нулевость главных элементов
        while p < n and a[p, i] == 0:
            p += 1

        if p == n:
            print("Информация: Нет уникального решения.")
        else:
            if p != i:
                # Меняем строки местами
                a[[i, p]] = a[[p, i]]

        for j in range(i + 1, n):
            a[j, :] = a[j, :] - a[i, :] * (a[j, i] / a[i, i])

    # Проверяем на ненулевость последней записи
    if a[n - 1, n - 1] == 0:
        print("Информация: Нет уникального решения.")

    return a
