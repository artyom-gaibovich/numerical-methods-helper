"""Методы для вычисления пределов."""

import math


def limit_epsilon_delta(f, x, toler, iter_max):
    """Вычисляет предел, используя определение эпсилон-дельта.

    Args:
        f (function): функция f(x).
        x (float): значение, к которому приближается независимая переменная.
        toler (float): допустимая погрешность (критерий останова).
        iter_max (int): максимальное количество итераций (критерий останова).

    Returns:
        limit (float): значение предела.
    """
    delta = 0.1
    limit_low_prev = -math.inf
    limit_up_prev = math.inf

    converged = False
    for i in range(0, iter_max + 1):
        delta /= (i + 1)
        limit_low = f(x - delta)
        limit_up = f(x + delta)

        if math.fabs(limit_low - limit_low_prev) <= toler \
           and math.fabs(limit_up - limit_up_prev) <= toler \
           and math.fabs(limit_up - limit_low) <= toler:
            converged = True
            break

        limit_up_prev = limit_up
        limit_low_prev = limit_low

    if math.fabs(limit_up - limit_low) > 10 * toler:
        raise ValueError("Двусторонний предел не существует.")

    return limit_low, i, converged
