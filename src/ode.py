"""Методы для обыкновенных дифференциальных уравнений."""

import numpy as np


def euler(f, a, b, n, ya):
    """Вычислить решение задачи Коши (IVP).

    Решить задачу Коши методом Эйлера.

    Args:
        f (function): уравнение f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.
        ya (numpy.ndarray): начальные значения.

    Returns:
        vx (numpy.ndarray): значения x.
        vy (numpy.ndarray): значения y (решение IVP).
    """
    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    vx[0] = x
    vy[0] = y

    fxy = f(x, y)
    print(f"i = 000,\tx = {x:+.4f},\ty = {y:+.4f}")

    for i in range(0, n):
        x = a + (i + 1) * h
        y += h * fxy

        fxy = f(x, y)
        print(f"i = {(i+1):03d},\tx = {x:+.4f},\ty = {y:+.4f}")
        vx[i] = x
        vy[i] = y

    return vx, vy


def taylor2(f, df1, a, b, n, ya):
    """Вычислить решение задачи Коши (IVP).

    Решить задачу Коши методом Тейлора (второго порядка).

    Args:
        f (function): уравнение f(x).
        df1 (function): первая производная уравнения f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.
        ya (numpy.ndarray): начальные значения.

    Returns:
        vx (numpy.ndarray): значения x.
        vy (numpy.ndarray): значения y (решение IVP).
    """
    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    vx[0] = x
    vy[0] = y

    print(f"i = 000,\tx = {x:+.4f},\ty = {y:+.4f}")

    for i in range(0, n):
        y += h * (f(x, y) + 0.5 * h * df1(x, y))
        x = a + (i + 1) * h

        print(f"i = {(i + 1):03d},\tx = {x:+.4f},\ty = {y:+.4f}")
        vx[i] = x
        vy[i] = y

    return vx, vy


def taylor4(f, df1, df2, df3, a, b, n, ya):
    """Вычислить решение задачи Коши (IVP).

    Решить задачу Коши методом Тейлора (четвертого порядка).

    Args:
        f (function): уравнение f(x).
        df1 (function): первая производная уравнения f(x).
        df2 (function): вторая производная уравнения f(x).
        df3 (function): третья производная уравнения f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.
        ya (numpy.ndarray): начальные значения.

    Returns:
        vx (numpy.ndarray): значения x.
        vy (numpy.ndarray): значения y (решение IVP).
    """
    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    vx[0] = x
    vy[0] = y

    print(f"i = 000,\tx = {x:+.4f},\ty = {y:+.4f}")

    for i in range(0, n):
        y += h * (f(x, y) + 0.5 * h * df1(x, y) + (h ** 2 / 6) * df2(x, y) +
                  (h ** 3 / 24) * df3(x, y))
        x = a + (i + 1) * h

        print(f"i = {(i + 1):03d},\tx = {x:+.4f},\ty = {y:+.4f}")
        vx[i] = x
        vy[i] = y

    return vx, vy


def rk4(f, a, b, n, ya):
    """Вычислить решение задачи Коши (IVP).

    Решить задачу Коши методом Рунге-Кутта (четвертого порядка).

    Args:
        f (function): уравнение f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.
        ya (numpy.ndarray): начальные значения.

    Returns:
        vx (numpy.ndarray): значения x.
        vy (numpy.ndarray): значения y (решение IVP).
    """
    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    k = np.zeros(4)

    vx[0] = x
    vy[0] = y

    print(f"i = 000,\tx = {x:+.4f},\ty = {y:+.4f}")

    for i in range(0, n):
        k[0] = h * f(x, y)
        k[1] = h * f(x + h / 2, y + k[0] / 2)
        k[2] = h * f(x + h / 2, y + k[1] / 2)
        k[3] = h * f(x + h, y + k[2])

        x = a + (i + 1) * h
        y += (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6

        print(f"i = {(i+1):03d},\tx = {x:+.4f},\ty = {y:+.4f}")
        vx[i] = x
        vy[i] = y

    return vx, vy


def rk4_system(f, a, b, n, ya):
    """Вычислить решение систем дифференциальных уравнений.

    Решить с помощью метода Рунге-Кутта (четвертого порядка).

    Args:
        f (numpy.ndarray): уравнения f(x).
        a (float): начальная точка.
        b (float): конечная точка.
        n (int): количество интервалов.
        ya (numpy.ndarray): начальные значения.

    Returns:
        vx (numpy.ndarray): значения x.
        vy (numpy.ndarray): значения y (решение IVP).
    """
    m = len(f)

    k = [np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)]

    vx = np.zeros(n + 1)
    vy = np.zeros((m, n + 1))

    h = (b - a) / n

    x = a
    y = ya

    vx[0] = x
    vy[:, 0] = y

    for i in range(0, n):

        for j in range(0, m):
            k[0][j] = h * f[j](x, y)

        for j in range(0, m):
            k[1][j] = h * f[j](x + h / 2, y + k[0] / 2)

        for j in range(0, m):
            k[2][j] = h * f[j](x + h / 2, y + k[1] / 2)

        for j in range(0, m):
            k[3][j] = h * f[j](x + h, y + k[2])

        x = a + i * h
        y = y + (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6

        vx[i + 1] = x
        vy[:, i + 1] = y

    return vx, vy
