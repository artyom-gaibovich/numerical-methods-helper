"""Численное дифференцирование."""

import numpy as np


def backward_difference(x, y):
    """Вычисляет первую производную.

    Все значения в 'x' должны быть равномерно распределены.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.

    Returns:
        dy (numpy.ndarray): значения первой производной.
    """
    if x.size < 2 or y.size < 2:
        raise ValueError("Массивы 'x' и 'y' должны содержать как минимум 2 значения.")

    if x.size != y.size:
        raise ValueError("Массивы 'x' и 'y' должны иметь одинаковый размер.")

    def dy_difference(h, y0, y1):
        return (y1 - y0) / h

    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i == n - 1:
            hx = x[i] - x[i - 1]
            dy[i] = dy_difference(-hx, y[i], y[i - 1])
        else:
            hx = x[i + 1] - x[i]
            dy[i] = dy_difference(hx, y[i], y[i + 1])

    return dy


def three_point(x, y):
    """Вычисляет первую производную.

    Все значения в 'x' должны быть равномерно распределены.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.

    Returns:
        dy (numpy.ndarray): значения первой производной.
    """
    if x.size < 3 or y.size < 3:
        raise ValueError("Массивы 'x' и 'y' должны содержать как минимум 3 значения.")

    if x.size != y.size:
        raise ValueError("Массивы 'x' и 'y' должны иметь одинаковый размер.")

    def dy_mid(h, y0, y2):
        return (1 / (2 * h)) * (y2 - y0)

    def dy_end(h, y0, y1, y2):
        return (1 / (2 * h)) * (-3 * y0 + 4 * y1 - y2)

    hx = x[1] - x[0]
    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i == 0:
            dy[i] = dy_end(hx, y[i], y[i + 1], y[i + 2])
        elif i == n - 1:
            dy[i] = dy_end(-hx, y[i], y[i - 1], y[i - 2])
        else:
            dy[i] = dy_mid(hx, y[i - 1], y[i + 1])

    return dy


def five_point(x, y):
    """Вычисляет первую производную.

    Все значения в 'x' должны быть равномерно распределены.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.

    Returns:
        dy (numpy.ndarray): значения первой производной.
    """
    if x.size < 6 or y.size < 6:
        raise ValueError("Массивы 'x' и 'y' должны содержать как минимум 6 значений.")

    if x.size != y.size:
        raise ValueError("Массивы 'x' и 'y' должны иметь одинаковый размер.")

    def dy_mid(h, y0, y1, y3, y4):
        return (1 / (12 * h)) * (y0 - 8 * y1 + 8 * y3 - y4)

    def dy_end(h, y0, y1, y2, y3, y4):
        return (1 / (12 * h)) * \
            (-25 * y0 + 48 * y1 - 36 * y2 + 16 * y3 - 3 * y4)

    hx = x[1] - x[0]
    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i in (0, 1):
            dy[i] = dy_end(hx, y[i], y[i + 1], y[i + 2], y[i + 3], y[i + 4])
        elif i in (n - 1, n - 2):
            dy[i] = dy_end(-hx, y[i], y[i - 1], y[i - 2], y[i - 3], y[i - 4])
        else:
            dy[i] = dy_mid(hx, y[i - 2], y[i - 1], y[i + 1], y[i + 2])

    return dy
