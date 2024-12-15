from typing import Callable

import numpy as np
from scipy import integrate
from tabulate import tabulate


def f(x):
    return np.sin(x)


def Q(x, N):
    return 0.25 * x ** (N - 1) + 0.153 * x ** (N - 2) - 0.241 * x ** 2


def p(x):
    return abs(x - 0.5)


def phi(x, p_func: Callable = p):
    return p_func(x) * f(x)


def p_integral(k, a, b) -> float:
    def primitive_left(x, k) -> float:
        return 0.5 * x ** (k + 1) / (k + 1) - x ** (k + 2) / (k + 2)

    def primitive_right(x, k) -> float:
        return x ** (k + 2) / (k + 2) - 0.5 * x ** (k + 1) / (k + 1)

    if a <= 0.5 <= b:
        integral_left = primitive_left(0.5, k) - primitive_left(a, k)
        integral_right = primitive_right(b, k) - primitive_right(0.5, k)
        return integral_left + integral_right
    elif b <= 0.5:
        return primitive_left(b, k) - primitive_left(a, k)
    else:
        return primitive_right(b, k) - primitive_right(a, k)


def print_table(x_arr, y_arr, lab1="x", lab2="y"):
    x = [lab1] + list(map(float, x_arr))
    y = [lab2] + list(map(float, y_arr))
    table = (x, y)
    print(tabulate(table, tablefmt="fancy_grid"))


def m_list(a: float, b: float, N: int, p_int: Callable[[int, float, float], float] = p_integral) -> list[float]:
    return [p_int(i, a, b) for i in range(N)]


def build_matrex(N: int, x_arr: list[float]) -> list[list[float]]:
    matrex = [[pow(x_arr[i], j) for i in range(N)] for j in range(N)]
    return matrex


def A_list(a: float, b: float, N: int, x_arr: list[float], p_int: Callable[[int, float, float], float] = p_integral):
    np_mat = np.array(build_matrex(N, x_arr))
    np_ans = np.array(m_list(a, b, N, p_int))
    return np.linalg.solve(np_mat, np_ans)


def approx_integral(N: int, A: list[float], y_arr: list[float]) -> float:
    s = 0
    for i in range(N):
        s += A[i] * y_arr[i]
    return s


def precise_integral(a: float, b: float) -> float:
    return integrate.quad(phi, a, b)[0]


def main():
    print("Приближенное вычисление интегралов по различным КФ. Вариант 7")
    a, b = map(float, input("Введите промежутки интегрирования (по умолчанию [0, 1]): ").split() or (0, 1))
    prec = precise_integral(a, b)
    print(f"Точное значение интеграла (интеграл от {a} до {b} от |x-0.5|*sin(x)): {prec}")
    N = int(input("Введите количество узлов КФ (по умолчанию, 4): N = ") or 4)
    while len(x_arr := np.array(list(
            map(float, input(f"Введите {N} узлов квадратурной формулы: ").split() or np.linspace(a, b, N))))) != N:
        print(f"Требуется ввести ровно {N} узлов!")
    y_arr = f(x_arr)
    print_table(x_arr, y_arr)
    m = m_list(a, b, N)
    print_table(range(0, N), m, "i", "m_i")
    A = A_list(a, b, N, x_arr)
    print_table(x_arr, A, "x", "A")
    appr = approx_integral(N, A, y_arr)
    Q_arr = Q(x_arr, N)
    print(Q_arr)
    Q_A = A_list(a, b, N, Q_arr)
    q_appr = approx_integral(N, Q_A, Q_arr)
    print(f"Значение интеграла для многочлена степени {N - 1}: {q_appr}.\n"
          f"Погрешность: {q_appr - integrate.quad(lambda x: Q(x, N), a, b)[0]}\n")

    print(f"Приближенное значение интеграла функции phi(x) = |x-0.5|*sin(x) от {a} до {b}: {appr}\n"
          f"Абсолютная погрешность между точным и приближенным значением: {abs(appr - prec)}")


if __name__ == '__main__':
    main()
