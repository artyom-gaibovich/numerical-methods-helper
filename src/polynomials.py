"""Методы для полиномов."""

import math
import numpy as np

def briot_ruffini(a, root):
    """Деление полинома на другой полином.

    Формат: P(x) = Q(x) * (x - root) + остаток.

    Аргументы:
        a (numpy.ndarray): коэффициенты входного полинома.
        root (float): один из корней полинома.

    Возвращает:
        b (numpy.ndarray): коэффициенты выходного полинома.
        rest (float): остаток при делении полинома.
    """
    n = a.size - 1
    b = np.zeros(n)

    b[0] = a[0]

    for i in range(1, n):
        b[i] = b[i - 1] * root + a[i]

    rest = b[n - 1] * root + a[n]

    return b, rest

def newton_divided_difference(x, y):
    """Нахождение коэффициентов разделенных разностей Ньютона.

    Также, нахождение полинома Ньютона.

    Аргументы:
        x (numpy.ndarray): значения x.
        y (numpy.ndarray): значения y.

    Возвращает:
        f (numpy.ndarray): коэффициенты разделенных разностей Ньютона.
    """
    n = x.size
    q = np.zeros((n, n - 1))

    # Вставка 'y' в первый столбец матрицы 'q'
    q = np.concatenate((y[:, None], q), axis=1)

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = (q[i, j - 1] - q[i - 1, j - 1]) / (x[i] - x[i - j])

    # Копирование диагональных значений матрицы q в вектор f
    f = np.zeros(n)
    for i in range(0, n):
        f[i] = q[i, i]

    # Печать полинома
    print("Полином:")
    print(f"p(x)={f[0]:+.3f}", end="")
    for i in range(1, n):
        print(f"{f[i]:+.3f}", end="")
        for j in range(1, i + 1):
            print(f"(x{(x[j] * -1):+.3f})", end="")
    print("")

    return f

def root_limits(c):
    """Нахождение границ действительных корней полиномиального уравнения.

    Используя теорему Лагранжа, доказательство которой приведено Демидовичем и Мароном.

    Аргументы:
        c (numpy.ndarray): коэффициенты полинома.

    Возвращает:
        lim (numpy.ndarray): нижние и верхние границы положительных и
            отрицательных корней соответственно.
    """
    lim = np.zeros(4)
    n = len(c) - 1
    c = np.concatenate(([0], c))
    c = np.concatenate((c, [0]))

    if c[1] == 0:
        raise ValueError("Первый коэффициент равен нулю.")

    t = n + 1
    c[t + 1] = 0

    # Если c[t+1] равно нулю, тогда полином приведен.
    while True:
        if c[t] != 0:
            break
        t -= 1

    # Вычисление четырех границ действительных корней.
    for i in range(0, 4):
        if i in (1, 3):
            # Инверсия порядка коэффициентов.
            for j in range(1, t // 2 + 1):
                c[j], c[t - j + 1] = c[t - j + 1], c[j]
        else:
            if i == 2:
                # Переинверсия порядка и обмен
                # знаков коэффициентов.
                for j in range(1, t // 2 + 1):
                    c[j], c[t - j + 1] = c[t - j + 1], c[j]
                for j in range(t - 1, 0, -2):
                    c[j] = -c[j]

        # Если c[1] отрицательно, тогда все коэффициенты меняются знаками.
        if c[1] < 0:
            for j in range(1, t + 1):
                c[j] = -c[j]

        # Вычисление 'k', наибольшего индекса отрицательных коэффициентов.
        k = 2
        while True:
            if c[k] < 0 or k > t:
                break
            k += 1

        # Вычисление 'b', наибольшего отрицательного коэффициента по модулю.
        if k <= t:
            b = 0
            for j in range(2, t + 1):
                if c[j] < 0 and math.fabs(c[j]) > b:
                    b = math.fabs(c[j])

            # Граница положительных корней 'P(x) = 0' и вспомогательных уравнений.
            lim[i] = 1 + (b / c[1]) ** (1 / (k - 1))
        else:
            lim[i] = 10 ** 100

    # Граница положительных и отрицательных корней 'P(x) = 0'.
    lim[0], lim[1], lim[2], lim[3] = 1 / lim[1], lim[0], -lim[2], -1 / lim[3]

    return lim
