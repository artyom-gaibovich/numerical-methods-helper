"""Методы для решений уравнений."""

import math

def bisection(f, a, b, toler, iter_max):
    """Вычисление корня уравнения методом бисекции.

    Аргументы:
        f (function): уравнение f(x).
        a (float): нижний предел.
        b (float): верхний предел.
        toler (float): допустимая погрешность (критерий остановки).
        iter_max (int): максимальное количество итераций (критерий остановки).

    Возвращает:
        root (float): значение корня.
        iter (int): количество использованных итераций методом.
        converged (boolean): флаг, указывающий, найден ли корень.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Функция не меняет знак на концах данного интервала.")

    delta_x = math.fabs(b - a) / 2

    x = 0
    converged = False
    for i in range(0, iter_max + 1):
        x = (a + b) / 2
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if delta_x <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        if fa * fx > 0:
            a = x
            fa = fx
        else:
            b = x

        delta_x = delta_x / 2

    root = x
    return root, i, converged

def secant(f, a, b, toler, iter_max):
    """Вычисление корня уравнения методом секущих.

    Аргументы:
        f (function): уравнение f(x).
        a (float): нижний предел.
        b (float): верхний предел.
        toler (float): допустимая погрешность (критерий остановки).
        iter_max (int): максимальное количество итераций (критерий остановки).

    Возвращает:
        root (float): значение корня.
        iter (int): количество использованных итераций методом.
        converged (boolean): флаг, указывающий, найден ли корень.
    """
    fa = f(a)
    fb = f(b)

    if fb - fa == 0:
        raise ValueError("f(b)-f(a) должно быть ненулевым.")

    if b - a == 0:
        raise ValueError("b-a должно быть ненулевым.")

    if math.fabs(fa) < math.fabs(fb):
        a, b = b, a
        fa, fb = fb, fa

    x = b
    fx = fb

    converged = False
    for i in range(0, iter_max + 1):
        delta_x = -fx / (fb - fa) * (b - a)
        x += delta_x
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        a, b = b, x
        fa, fb = fb, fx

    root = x
    return root, i, converged

def regula_falsi(f, a, b, toler, iter_max):
    """Вычисление корня уравнения методом Регулы Фальси.

    Аргументы:
        f (function): уравнение f(x).
        a (float): нижний предел.
        b (float): верхний предел.
        toler (float): допустимая погрешность (критерий остановки).
        iter_max (int): максимальное количество итераций (критерий остановки).

    Возвращает:
        root (float): значение корня.
        iter (int): количество использованных итераций методом.
        converged (boolean): флаг, указывающий, найден ли корень.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Функция не меняет знак на концах данного интервала.")

    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    x = b
    fx = fb

    converged = False
    for i in range(0, iter_max + 1):
        delta_x = -fx / (fb - fa) * (b - a)
        x += delta_x
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        if fx < 0:
            a = x
            fa = fx
        else:
            b = x
            fb = fx

    root = x
    return root, i, converged

def pegasus(f, a, b, toler, iter_max):
    """Вычисление корня уравнения методом Пегаса.

    Аргументы:
        f (function): уравнение f(x).
        a (float): нижний предел.
        b (float): верхний предел.
        toler (float): допустимая погрешность (критерий остановки).
        iter_max (int): максимальное количество итераций (критерий остановки).

    Возвращает:
        root (float): значение корня.
        iter (int): количество использованных итераций методом.
        converged (boolean): флаг, указывающий, найден ли корень.
    """
    fa = f(a)
    fb = f(b)
    x = b
    fx = fb

    converged = False
    for i in range(0, iter_max + 1):
        delta_x = -fx / (fb - fa) * (b - a)
        x += delta_x
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        if fx * fb < 0:
            a = b
            fa = fb
        else:
            fa = fa * fb / (fb + fx)

        b = x
        fb = fx

    root = x
    return root, i, converged

def muller(f, a, c, toler, iter_max):
    """Вычисление корня уравнения методом Мюллера.

    Аргументы:
        f (function): уравнение f(x).
        a (float): нижний предел.
        c (float): верхний предел.
        toler (float): допустимая погрешность (критерий остановки).
        iter_max (int): максимальное количество итераций (критерий остановки).

    Возвращает:
        root (float): значение корня.
        iter (int): количество использованных итераций методом.
        converged (boolean): флаг, указывающий, найден ли корень.
    """
    b = (a + c) / 2
    fa = f(a)
    fb = f(b)
    fc = f(c)
    x = b
    fx = fb
    delta_x = c - a

    converged = False
    for i in range(0, iter_max + 1):
        h1 = c - b
        h2 = b - a
        r = h1 / h2
        t = x

        aa = (fc - (r + 1) * fb + r * fa) / (h1 * (h1 + h2))
        bb = (fc - fb) / h1 - aa * h1
        cc = fb

        signal_bb = int(math.copysign(1, bb))

        z = (-bb + signal_bb * math.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)
        x = b + z

        delta_x = x - t
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        if x > b:
            a = b
            fa = fb
        else:
            c = b
            fc = fb

        b = x
        fb = fx

    root = x
    return root, i, converged

def newton(f, df, x0, toler, iter_max):
    """Вычисление корня уравнения методом Ньютона.

    Аргументы:
        f (function): уравнение f(x).
        df (function): производная уравнения f(x).
        x0 (float): начальное приближение.
        toler (float): допустимая погрешность (критерий остановки).
        iter_max (int): максимальное количество итераций (критерий остановки).

    Возвращает:
        root (float): значение корня.
        iter (int): количество использованных итераций методом.
        converged (boolean): флаг, указывающий, найден ли корень.
    """
    fx = f(x0)
    dfx = df(x0)
    x = x0

    print(f"i = 000,\tx = {x:+.4f},\tfx = {fx:+.4f}")

    converged = False
    for i in range(1, iter_max + 1):
        delta_x = -fx / dfx
        x += delta_x
        fx = f(x)
        dfx = df(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler or dfx == 0:
            converged = True
            break

    root = x
    return root, i, converged
