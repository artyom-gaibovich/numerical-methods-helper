"""Методы для решения линейных систем."""

import numpy as np


def jacobi(a, b, x0, toler, iter_max):
    """Метод Якоби: решение Ax = b с начальным приближением x0.

    Args:
        a (numpy.ndarray): матрица A из системы Ax=b.
        b (numpy.ndarray): значения b.
        x0 (numpy.ndarray): начальное приближение решения.
        toler (float): допуск (критерий остановки).
        iter_max (int): максимальное число итераций (критерий остановки).

    Returns:
        x (numpy.ndarray): решение линейной системы.
        iter (int): число итераций, использованных методом.
    """
    # Матрицы D и M
    d = np.diag(np.diag(a))
    m = a - d

    # Итерационный процесс
    x = None
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(d, (b - np.dot(m, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= toler:
            break
        x0 = x.copy()

    return x, i


def gauss_seidel(a, b, x0, toler, iter_max):
    """Метод Гаусса-Зейделя: решение Ax = b с начальным приближением x0.

    Args:
        a (numpy.ndarray): матрица A из системы Ax=b.
        b (numpy.ndarray): значения b.
        x0 (numpy.ndarray): начальное приближение решения.
        toler (float): допуск (критерий остановки).
        iter_max (int): максимальное число итераций (критерий остановки).

    Returns:
        x (numpy.ndarray): решение линейной системы.
        iter (int): число итераций, использованных методом.
    """
    # Матрицы L и U
    lower = np.tril(a)
    upper = a - lower

    # Итерационный процесс
    x = None
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(lower, (b - np.dot(upper, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= toler:
            break
        x0 = x.copy()

    return x, i
