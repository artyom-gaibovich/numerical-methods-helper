"""Методы численного интегрирования."""

import numpy as np


def simpson(f, a, b, n):
    """Вычисляет интеграл методом Симпсона.

    Args:
        f (function): функция f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.

    Returns:
        xi (float): численное приближение определенного интеграла.
    """
    h = (b - a) / n

    sum_odd = 0
    sum_even = 0

    for i in range(0, n - 1):
        x = a + (i + 1) * h
        if (i + 1) % 2 == 0:
            sum_even += f(x)
        else:
            sum_odd += f(x)

    xi = h / 3 * (f(a) + 2 * sum_even + 4 * sum_odd + f(b))
    return xi


def trapezoidal(f, a, b, n):
    """Вычисляет интеграл методом трапеций.

    Args:
        f (function): функция f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.

    Returns:
        xi (float): численное приближение определенного интеграла.
    """
    h = (b - a) / n

    sum_x = 0

    for i in range(0, n - 1):
        x = a + (i + 1) * h
        sum_x += f(x)

    xi = h / 2 * (f(a) + 2 * sum_x + f(b))
    return xi


def simpson_array(x, y):
    """Вычисляет интеграл методом Симпсона.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.

    Returns:
        xi (float): численное приближение определенного интеграла.
    """
    if x.size != y.size:
        raise ValueError("Массивы 'x' и 'y' должны иметь одинаковый размер.")

    h = x[1] - x[0]
    n = x.size

    sum_odd = 0
    sum_even = 0

    for i in range(1, n - 1):
        if (i + 1) % 2 == 0:
            sum_even += y[i]
        else:
            sum_odd += y[i]

    xi = h / 3 * (y[0] + 2 * sum_even + 4 * sum_odd + y[n - 1])
    return xi


def trapezoidal_array(x, y):
    """Вычисляет интеграл методом трапеций.

    Args:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.

    Returns:
        xi (float): численное приближение определенного интеграла.
    """
    if x.size != y.size:
        raise ValueError("Массивы 'x' и 'y' должны иметь одинаковый размер.")

    h = x[1] - x[0]
    n = x.size

    sum_x = 0

    for i in range(1, n - 1):
        sum_x += y[i]

    xi = h / 2 * (y[0] + 2 * sum_x + y[n - 1])
    return xi


def romberg(f, a, b, n):
    """Вычисляет интеграл методом Ромберга.

    Args:
        f (function): функция f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.

    Returns:
        xi (float): численное приближение определенного интеграла.
    """
    # Инициализация таблицы интегрирования Ромберга
    r = np.zeros((n, n))

    # Вычисление правила трапеций для первого столбца (h = b - a)
    h = b - a
    r[0, 0] = 0.5 * h * (f(a) + f(b))

    # Итерация для каждого уровня уточнения
    for i in range(1, n):
        h = 0.5 * h  # Уменьшаем шаг вдвое
        # Вычисление составного правила трапеций
        sum_f = 0
        for j in range(1, 2**i, 2):
            x = a + j * h
            sum_f += f(x)
        r[i, 0] = 0.5 * r[i - 1, 0] + h * sum_f

        # Экстраполяция Ричардсона для более высоких порядков аппроксимации
        for k in range(1, i + 1):
            r[i, k] = r[i, k - 1] + \
                (r[i, k - 1] - r[i - 1, k - 1]) / ((4**k) - 1)

    return float(r[n - 1, n - 1])
