from collections.abc import Callable

import numpy as np
from task41 import p, p_integral, approx_integral, m_list, A_list, print_table
from scipy import integrate
import math


def f(x: float | list[float]) -> float | list[float]:
    return np.sin(x)


def p1(x: float) -> float:
    return 1


def p2(x: float) -> float:
    return 1 / math.sqrt(1 - x ** 2)


def p1_integral(k: int, a: float, b: float) -> float:
    return (b ** (k + 1) - a ** (k + 1)) / (k + 1)


def p2_integral(k: int, a: float, b: float) -> float:
    return integrate.quad(lambda x: x ** k * p2(x), a, b)[0]


def polynom_n(x: float, N: int) -> float:
    return 0.175 * x ** (N - 1) - 2.55 * x + 1.125


def polynom_2n(x: float, N: int) -> float:
    return 0.175 * x ** (2 * N - 1) - 2.55 * x + 1.125


def simple_solution(weight: Callable[[float], float], weight_integral: Callable[[int, float, float], float], a: float,
                    b: float,
                    N: int):
    phi = lambda x: f(x) * weight(x)
    precise = integrate.quad(phi, a, b)[0]
    print(f"Точное значение интеграла p(x) * f(x) от {a} до {b}: {precise}")
    h = (b - a) / (N - 1)
    x_arr = [a + i * h for i in range(N)]
    y_arr = f(x_arr)
    print("Узлы КФ:")
    print_table(x_arr, y_arr)
    m = m_list(a, b, N, weight_integral)
    print("Моменты весовой функции ИКФ:")
    print_table(range(0, N), m, "i", "m_i")
    A = A_list(a, b, N, x_arr, weight_integral)
    print("Коэффициенты КФ: ")
    print_table(x_arr, A, "x", "A")
    approx = approx_integral(N, A, y_arr)
    poly_arr = [polynom_n(x, N) for x in x_arr]
    poly_approx = approx_integral(N, A, poly_arr)
    print(f"Погрешность значения интеграла для многочлена степени N - 1 = {N - 1}: "
          f"{abs(poly_approx - integrate.quad(lambda x: polynom_n(x, N) * weight(x), a, b)[0])}")

    print(f"Приближенное значение интеграла f(x) * p(x) от {a} до {b} по ИКФ: {approx}\n"
          f"Погрешность: {abs(approx - precise)}")


def p_integral_qf(k: int, a: float, b: float, weight: Callable[[float], float]) -> float:
    m = 100000
    h = (b - a) / m
    z_arr = [a + i * h for i in range(m + 1)]
    func = lambda x: weight(x) * x ** k
    q = sum(func(z_arr[i] + h / 2) for i in range(m))
    return q * h


def m_list_new(a: float, b: float, N: int, weight: Callable[[float], float]) -> list[float]:
    return [p_integral_qf(i, a, b, weight) for i in range(2 * N)]  # неточное значение
    # return [integrate.quad(lambda x: x**i * weight(x), a, b)[0] for i in range(2 * N)]


def build_m_matrex(new_m_list: list[float], N: int) -> list[list[float]]:
    mat = [[new_m_list[i] for i in range(j, N + j)] for j in range(N)]
    return mat


def find_a(new_m_list: list[float], N: int) -> list[float]:
    ans_col = [-new_m_list[i] for i in range(N, 2 * N)]
    return list(map(float, np.linalg.solve(build_m_matrex(new_m_list, N), ans_col)))


def find_roots(a_list: list[float]) -> list[float]:
    coefs = [1.0] + a_list
    return list(map(float, np.roots(coefs)))


def build_x_matrex(N: int, x_arr: list[float]) -> list[list[float]]:
    mat = [[pow(x_arr[i], j) for i in range(N)] for j in range(N)]
    return mat


def new_A_list(x_mat: list[list[float]], new_m_list: list[float], N: int):
    return list(map(float, np.linalg.solve(x_mat, new_m_list[:N])))


def new_solution(weight: Callable[[float], float], a: float, b: float,
                 N: int):
    phi = lambda x: f(x) * weight(x)
    precise = integrate.quad(phi, a, b)[0]

    m = m_list_new(a, b, N, weight)
    print("Моменты весовой функции:")
    print_table(range(2 * N), m, "i", "m_i")

    a_list = find_a(m, N)[::-1]
    print("Коэффициенты ортогонального многочлена:")
    print_table(range(N), a_list, "i", "a_i")

    x_arr = sorted(find_roots(a_list))
    x_mat = build_x_matrex(N, x_arr)
    A = new_A_list(x_mat, m, N)
    print("Коэффициенты КФ:")
    print_table(range(1, N + 1), A, "i", "A_i")
    y_arr = [f(x) for x in x_arr]
    print("Узлы КФ:")
    print_table(x_arr, y_arr)
    approx = approx_integral(N, A, y_arr)
    poly_arr = [polynom_2n(x, N) for x in x_arr]
    poly_approx = approx_integral(N, A, poly_arr)
    print(f"Погрешность значения интеграла для многочлена степени 2N - 1 = {2 * N - 1}: "
          f"{abs(poly_approx - integrate.quad(lambda x: polynom_2n(x, N) * weight(x), a, b)[0])}")

    print(f"Приближенное значение интеграла f(x) * p(x) от {a} до {b} по КФ НАСТ: {approx}\n"
          f"Погрешность: {abs(approx - precise)}")


def main():
    print("Приближенное вычисление интегралов с помощью КФ НАСТ. Вариант 7")
    print("f = sin(x). Весовые функции: \n"
          "1. p = |x - 0.5|\n"
          "2. p = 1\n"
          "3. p = 1 / sqrt(1 - x^2)\n")
    weights = ((p, p_integral), (p1, p1_integral), (p2, p2_integral))
    weight_strs = ("|x - 0.5|", "1", "1 / sqrt(1 - x^2)")
    while (N := int(input("Введите количество узлов ИКФ и КФ НАСТ (по умолчанию, 6): N = ") or 6)) < 2:
        print("Введите значение N >= 2!")
    choice = int(input(
        "Выберите весовую функцию, для которой нужно найти интеграл f(x) * p(x) (по умолчанию, 0 - для каждой): ") or 0) - 1
    if choice == -1:
        a, b = map(float, input(
            "Введите промежутки интегрирования для веса p = |x - 0.5| (по умолчанию [0, 1]): ").split() or (0, 1))
        for i, (weight, weight_integral) in enumerate(weights):
            print("-------------------------------------------------------")
            print(f"Решение для веса p = {weight_strs[i]} с помощью ИКФ:")
            simple_solution(weight, weight_integral, a, b, N)
            print("\nРешение с помощью КФ НАСТ:")
            new_solution(weight, a, b, N)
            a, b = (-1, 1)
    else:
        a, b = (-1, 1)
        if choice == 0:
            a, b = map(float, input(
                "Введите промежутки интегрирования для веса p = |x - 0.5| (по умолчанию [0, 1]): ").split() or (0, 1))
        weight, weight_integral = weights[choice]
        print(f"\nРешение для веса p = {weight_strs[choice]} с помощью ИКФ:")
        simple_solution(weight, weight_integral, a, b, N)
        print("Решение с помощью КФ НАСТ:")
        new_solution(weight, a, b, N)


if __name__ == "__main__":
    main()
