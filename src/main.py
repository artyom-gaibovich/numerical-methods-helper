"""
Реализация численных методов на Python.

Автор: Cristiano Fraga G. Nunes <cfgnunes@gmail.com>

Минимальная требуемая версия Python - 3.6.
"""

import math

import numpy as np

import differentiation
import integration
import interpolation
import limits
import linear_systems
import linear_systems_iterative
import ode
import polynomials
import solutions


def print_docstring(func):
    """Вывести строку документации функции (декоратор)."""
    def wrapper(*args, **kwargs):
        print(func.__doc__)
        result = func(*args, **kwargs)
        print("\n")
        return result
    return wrapper


@print_docstring
def example_limit_epsilon_delta():
    """Запустить пример 'Пределы: определение эпсилон-дельта'."""
    def f(x):
        return math.sin(x) / x

    x = 0
    toler = 10 ** -5
    iter_max = 100

    print("Входные данные:")
    print(f"x = {x}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Выполнение:")
    limit, i, converged = limits.limit_epsilon_delta(f, x, toler, iter_max)

    print("Выходные данные:")
    print(f"limit = {limit:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_bisection():
    """Запустить пример 'Решения: Метод бисекции'."""
    # Метод бисекции (поиск корней уравнения)
    #   Плюсы:
    #       Это надежный метод с гарантированной сходимостью;
    #       Это простой метод, который ищет корень, используя бинарный поиск;
    #       Нет необходимости вычислять производную функции.
    #   Минусы:
    #       Медленная сходимость;
    #       Необходимо задать интервал поиска [a, b];
    #       Указанный интервал должен иметь смену знака, f(a) * f(b) < 0.

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Выполнение:")
    root, i, converged = solutions.bisection(f, a, b, toler, iter_max)

    print("Выходные данные:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_secant():
    """Запустить пример 'Решения: Метод секущих'."""
    # Метод секущих (поиск корней уравнения)
    #   Плюсы:
    #       Это быстрый метод (медленнее метода Ньютона);
    #       Он основан на методе Ньютона, но не требует вычисления
    #           производной функции.
    #   Минусы:
    #       Может расходиться, если функция не является приблизительно линейной
    #           в интервале, содержащем корень;
    #       Необходимо задать две точки, 'a' и 'b', где
    #           f(a)-f(b) должно быть ненулевым.

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Выполнение:")
    root, i, converged = solutions.secant(f, a, b, toler, iter_max)

    print("Выходные данные:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_regula_falsi():
    """Запустить пример 'Решения: Метод ложного положения'."""

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Выполнение:")
    root, i, converged = solutions.regula_falsi(f, a, b, toler, iter_max)

    print("Выходные данные:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_pegasus():
    """Запустить пример 'Решения: Метод Пегас'."""

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Выполнение:")
    root, i, converged = solutions.pegasus(f, a, b, toler, iter_max)

    print("Выходные данные:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_muller():
    """Запустить пример 'Решения: Метод Мюллера'."""

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Выполнение:")
    root, i, converged = solutions.muller(f, a, b, toler, iter_max)

    print("Выходные данные:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_newton():
    """Запустить пример 'Решения: Метод Ньютона'."""
    # Метод Ньютона (поиск корней уравнения)
    #   Плюсы:
    #       Это быстрый метод.
    #    Минусы:
    #       Может расходиться;
    #       Необходимо вычислять производную функции;
    #       Необходимо задать начальное значение x0, где
    #           f'(x0) должно быть ненулевым.

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    def df(x):
        return 12 * x ** 2 + 1 - math.sin(x)

    x0 = 1.0
    toler = 0.01
    iter_max = 100

    print("Входные данные:")
    print(f"x0 = {x0}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Выполнение:")
    root, i, converged = solutions.newton(f, df, x0, toler, iter_max)

    print("Выходные данные:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_interpolation_lagrange():
    """Запустить пример 'Интерполяция: Лагранж'."""
    x = np.array([2, 11 / 4, 4])
    y = np.array([1 / 2, 4 / 11, 1 / 4])
    x_int = 3

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    y_int = interpolation.lagrange(x, y, x_int)

    print("Выходные данные:")
    print(f"y_int = {y_int:.5f}")


@print_docstring
def example_interpolation_newton():
    """Запустить пример 'Интерполяция: Ньютон'."""
    x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
    y = np.array([0.3162, 0.5477, 0.6325, 0.7746, 0.8367])
    x_int = 0.2

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    y_int = interpolation.newton(x, y, x_int)

    print("Выходные данные:")
    print(f"y_int = {y_int:.5f}")


@print_docstring
def example_interpolation_gregory_newton():
    """Запустить пример 'Интерполяция: Грегори-Ньютон'."""
    x = np.array([110, 120, 130])
    y = np.array([2.0410, 2.0790, 2.1140])
    x_int = 115

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    y_int = interpolation.gregory_newton(x, y, x_int)

    print("Выходные данные:")
    print(f"y_int = {y_int:.5f}")


@print_docstring
def example_interpolation_neville():
    """Запустить пример 'Интерполяция: Невилль'."""
    x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
    x_int = 1.5

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    y_int, q = interpolation.neville(x, y, x_int)

    print("Выходные данные:")
    print(f"y_int = {y_int:.5f}")
    print(f"q =\n{q}")


@print_docstring
def example_polynomial_root_limits():
    """Запустить пример 'Полиномы: Границы корней'."""
    c = np.array([1, 2, -13, -14, 24])

    print("Входные данные:")
    print(f"c = {c}")

    limits = polynomials.root_limits(c)

    print("Выходные данные:")
    print(f"limits = {limits}")


@print_docstring
def example_polynomial_briot_ruffini():
    """Запустить пример 'Полиномы: Метод Бриота-Руффини'."""
    a = np.array([2, 0, -3, 3, -4])
    root = -2

    print("Входные данные:")
    print(f"a = {a}")
    print(f"root = {root:.5f}")

    b, rest = polynomials.briot_ruffini(a, root)

    print("Выходные данные:")
    print(f"b = {b}")
    print(f"rest = {rest}")


@print_docstring
def example_polynomial_newton_divided_difference():
    """Запустить пример 'Полиномы: Разделенные разности Ньютона'."""
    x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")

    f = polynomials.newton_divided_difference(x, y)

    print("Выходные данные:")
    print(f"f = {f}")


@print_docstring
def example_differentiation_backward_difference():
    """Запустить пример 'Дифференцирование: Обратная разность'."""
    x = np.array([0.0, 0.2, 0.4])
    y = np.array([0.00000, 0.74140, 1.3718])

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")

    dy = differentiation.backward_difference(x, y)

    print("Выходные данные:")
    print(f"dy = {dy}")



@print_docstring
def example_differentiation_three_point():
    """Запуск примера 'Дифференцирование: трехточечная схема'."""
    x = np.array([1.1, 1.2, 1.3, 1.4])
    y = np.array([9.025013, 11.02318, 13.46374, 16.44465])

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")

    dy = differentiation.three_point(x, y)

    print("Результат:")
    print(f"dy = {dy}")


@print_docstring
def example_differentiation_five_point():
    """Запуск примера 'Дифференцирование: пятиточечная схема'."""
    x = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
    y = np.array([-1.709847, -1.373823, -1.119214,
                  -0.9160143, -0.7470223, -0.6015966])

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")

    dy = differentiation.five_point(x, y)

    print("Результат:")
    print(f"dy = {dy}")


@print_docstring
def example_trapezoidal_array():
    """Запуск примера 'Интегрирование: правило трапеций' (массивы)."""
    x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    y = np.array([124, 134, 148, 156, 147, 133,
                  121, 109, 99, 85, 78, 89, 104, 116, 123])

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")

    xi = integration.trapezoidal_array(x, y)

    print("Результат:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_trapezoidal():
    """Запуск примера 'Интегрирование: правило трапеций'."""
    def f(x):
        return x ** 2 * math.log(x ** 2 + 1)

    a = 0.0
    b = 2.0
    h = 0.25
    n = int((b - a) / h)

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")

    xi = integration.trapezoidal(f, a, b, n)

    print("Результат:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_simpson_array():
    """Запуск примера 'Интегрирование: составное правило 1/3 Симпсона' (массивы)."""
    x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    y = np.array([124, 134, 148, 156, 147, 133,
                  121, 109, 99, 85, 78, 89, 104, 116, 123])

    print("Входные данные:")
    print(f"x = {x}")
    print(f"y = {y}")

    xi = integration.simpson_array(x, y)

    print("Результат:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_simpson():
    """Запуск примера 'Интегрирование: составное правило 1/3 Симпсона'."""
    def f(x):
        return x ** 2 * math.log(x ** 2 + 1)

    a = 0.0
    b = 2.0
    h = 0.25
    n = int((b - a) / h)

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")

    xi = integration.simpson(f, a, b, n)

    print("Результат:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_romberg():
    """Запуск примера 'Интегрирование: метод Ромберга'."""
    def f(x):
        return x ** 2 * math.log(x ** 2 + 1)

    a = 0.0
    b = 2.0
    h = 0.25
    n = int((b - a) / h)

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")

    xi = integration.romberg(f, a, b, n)

    print("Результат:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_ode_euler():
    """Запуск примера 'ОДУ: метод Эйлера'."""
    def f(x, y):
        return y - x ** 2 + 1

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Выполнение:")
    vx, vy = ode.euler(f, a, b, n, ya)

    print("Результат:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_taylor2():
    """Запуск примера 'ОДУ: метод Тейлора (порядок 2)'."""
    def f(x, y):
        return y - x ** 2 + 1

    def df1(x, y):
        return y - x ** 2 + 1 - 2 * x

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Выполнение:")
    vx, vy = ode.taylor2(f, df1, a, b, n, ya)

    print("Результат:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_taylor4():
    """Запуск примера 'ОДУ: метод Тейлора (порядок 4)'."""
    def f(x, y):
        return y - x ** 2 + 1

    def df1(x, y):
        return y - x ** 2 + 1 - 2 * x

    def df2(x, y):
        return y - x ** 2 + 1 - 2 * x - 2

    def df3(x, y):
        return y - x ** 2 + 1 - 2 * x - 2

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Выполнение:")
    vx, vy = ode.taylor4(f, df1, df2, df3, a, b, n, ya)

    print("Результат:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_rk4():
    """Запуск примера 'ОДУ: метод Рунге-Кутты (порядок 4)'."""
    def f(x, y):
        return y - x ** 2 + 1

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    vx, vy = ode.rk4(f, a, b, n, ya)

    print("Результат:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_rk4_system():
    """Запуск примера 'ОДУ: метод Рунге-Кутты (порядок 4) для систем дифференциальных уравнений'."""
    f = []
    f.append(lambda x, y: - 4 * y[0] + 3 * y[1] + 6)
    f.append(lambda x, y: - 2.4 * y[0] + 1.6 * y[1] + 3.6)
    a = 0.0
    b = 0.5
    h = 0.1
    n = int((b - a) / h)
    ya = np.zeros(len(f))
    ya[0] = 0.0
    ya[1] = 0.0

    print("Входные данные:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Выполнение:")
    vx, vy = ode.rk4_system(f, a, b, n, ya)

    print("Результат:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_gauss_elimination_pp():
    """Запуск примера 'Линейные системы: метод Гаусса'."""
    a = np.array([[1, -1, 2, -1], [2, -2, 3, -3], [1, 1, 1, 0], [1, -1, 4, 3]])
    b = np.array([-8, -20, -2, 4])

    print("Входные данные:")
    print(f"a =\n{a}")
    print(f"b = {b}")

    a = linear_systems.gauss_elimination_pp(a, b)

    print("Результат:")
    print(f"a =\n{a}")

    return a


@print_docstring
def example_backward_substitution(a):
    """Запуск примера 'Линейные системы: метод обратной подстановки'."""
    upper = a[:, 0:-1]
    d = a[:, -1]

    print("Входные данные:")
    print(f"upper =\n{upper}")
    print(f"d = {d}")

    x = linear_systems.backward_substitution(upper, d)

    print("Результат:")
    print(f"x = {x}")


@print_docstring
def example_forward_substitution():
    """Запуск примера 'Линейные системы: метод прямой подстановки'."""
    lower = np.array([[3, 0, 0, 0], [-1, 1, 0, 0],
                      [3, -2, -1, 0], [1, -2, 6, 2]])
    c = np.array([5, 6, 4, 2])

    print("Входные данные:")
    print(f"lower =\n{lower}")
    print(f"c = {c}")

    x = linear_systems.forward_substitution(lower, c)

    print("Результат:")
    print(f"x = {x}")


@print_docstring
def example_jacobi():
    """Запуск примера 'Итеративные линейные системы: метод Якоби'."""
    a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
                  [2, -1, 10, -1], [0, 3, -1, 8]])
    b = np.array([6, 25, -11, 15])
    x0 = np.array([0, 0, 0, 0])
    toler = 10 ** -3
    iter_max = 10

    print("Входные данные:")
    print(f"a =\n{a}")
    print(f"b = {b}")
    print(f"x0 = {x0}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    x, i = linear_systems_iterative.jacobi(a, b, x0, toler, iter_max)

    print("Результат:")
    print(f"x = {x}")
    print(f"i = {i}")



@print_docstring
def example_gauss_seidel():
    """Запуск примера 'Итерационные линейные системы: метод Гаусса-Зейделя'."""
    a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
                  [2, -1, 10, -1], [0, 3, -1, 8]])
    b = np.array([6, 25, -11, 15])
    x0 = np.array([0, 0, 0, 0])
    toler = 10 ** -3
    iter_max = 10

    print("Входные данные:")
    print(f"a =\n{a}")
    print(f"b = {b}")
    print(f"x0 = {x0}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    x, i = linear_systems_iterative.gauss_seidel(a, b, x0, toler, iter_max)

    print("Выходные данные:")
    print(f"x = {x}")
    print(f"i = {i}")


def main():
    """Запуск главной функции."""
    # Выполнить все примеры

    # Пределы
    example_limit_epsilon_delta()

    # Решение уравнений
    example_solution_bisection()
    example_solution_secant()
    example_solution_regula_falsi()
    example_solution_pegasus()
    example_solution_muller()
    example_solution_newton()

    # Интерполяция
    example_interpolation_lagrange()
    example_interpolation_newton()
    example_interpolation_gregory_newton()
    example_interpolation_neville()

    # Алгоритмы для многочленов
    example_polynomial_root_limits()
    example_polynomial_briot_ruffini()
    example_polynomial_newton_divided_difference()

    # Численное дифференцирование
    example_differentiation_backward_difference()
    example_differentiation_three_point()
    example_differentiation_five_point()

    # Численное интегрирование
    example_trapezoidal_array()
    example_trapezoidal()
    example_simpson_array()
    example_simpson()
    example_romberg()

    # Задачи Коши для обыкновенных дифференциальных уравнений
    example_ode_euler()
    example_ode_taylor2()
    example_ode_taylor4()
    example_ode_rk4()

    # Системы дифференциальных уравнений
    example_ode_rk4_system()

    # Методы для линейных систем
    a = example_gauss_elimination_pp()
    example_backward_substitution(a)
    example_forward_substitution()

    # Итерационные методы для линейных систем
    example_jacobi()
    example_gauss_seidel()


if __name__ == '__main__':
    main()
