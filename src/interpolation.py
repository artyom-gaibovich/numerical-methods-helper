"""Методы интерполяции."""

import numpy as np


def lagrange(x, y, x_int):
    """Интерполирует значение с использованием полинома Лагранжа.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.
        x_int (float): значение для интерполяции.

    Returns:
        y_int (float): интерполированное значение.
    """
    m = x.size
    y_int = 0

    for i in range(0, m):
        p = y[i]
        for j in range(0, m):
            if i != j:
                p = p * (x_int - x[j]) / (x[i] - x[j])
        y_int = y_int + p

    return y_int


def newton(x, y, x_int):
    """Интерполирует значение с использованием полинома Ньютона.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.
        x_int (float): значение для интерполяции.

    Returns:
        y_int (float): интерполированное значение.
    """
    m = x.size
    del_y = y.copy()

    # Вычисляем разделенные разности
    for k in range(1, m):
        for i in range(m - 1, k - 1, -1):
            del_y[i] = (del_y[i] - del_y[i - 1]) / (x[i] - x[i - k])

    # Оцениваем полином методом Горнера
    y_int = del_y[-1]
    for i in range(m - 2, -1, -1):
        y_int = y_int * (x_int - x[i]) + del_y[i]

    return y_int


def gregory_newton(x, y, x_int):
    """Интерполирует значение с использованием полинома Грегори-Ньютона.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.
        x_int (float): значение для интерполяции.

    Returns:
        y_int (float): интерполированное значение.
    """
    m = x.size
    del_y = y.copy()

    # Вычисляем конечные разности
    for k in range(1, m):
        for i in range(m - 1, k - 1, -1):
            del_y[i] = del_y[i] - del_y[i - 1]

    # Оцениваем полином методом Горнера
    u = (x_int - x[0]) / (x[1] - x[0])
    y_int = del_y[-1]
    for i in range(m - 2, -1, -1):
        y_int = y_int * (u - i) / (i + 1) + del_y[i]

    return y_int


def neville(x, y, x_int):
    """Интерполирует значение с использованием полинома Невиля.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.
        x_int (float): значение для интерполяции.

    Returns:
        y_int (float): интерполированное значение.
        q (numpy.ndarray): матрица коэффициентов.
    """
    n = x.size
    q = np.zeros((n, n - 1))

    # Вставляем 'y' в первый столбец матрицы 'q'
    q = np.concatenate((y[:, None], q), axis=1)

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = ((x_int - x[i - j]) * q[i, j - 1] -
                       (x_int - x[i]) * q[i - 1, j - 1]) / (x[i] - x[i - j])

    y_int = q[n - 1, n - 1]
    return y_int, q
