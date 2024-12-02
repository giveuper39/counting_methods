#   y'(x) = -y(x) + cos(x), y(0) = 1
from typing import Callable

import numpy as np
from tabulate import tabulate


def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)


def f(x, y):
    return -y + np.cos(x)


def solution(x: float) -> float:
    return np.sin(x) / 2 + np.cos(x) / 2 + 1 / (2 * np.exp(x))


def taylor_solution(x: float, x0: float, y0: float, n: int = 5) -> float:
    derivs = [y0, -y0 + np.cos(x0), y0 - np.cos(x0) - np.sin(x0), -y0 + np.sin(x0)]  # производные меняются циклически
    y = 0
    for i in range(n + 1):
        y += derivs[i % 4] * (x - x0) ** i / factorial(i)
    return y


def taylor_method(x_arr: list[float], x0: float, y0: float, n: int = 5) -> list[float]:
    y_arr = [taylor_solution(x_arr[i], x0, y0, n) for i in range(len(x_arr))]
    return y_arr


def print_taylor_method(x_arr: list[float], y_arr: list[float]):
    table = [(i, x, y_prec := solution(x), y_t, abs(y_t - y_prec)) for i, (x, y_t) in enumerate(zip(x_arr, y_arr), -2)]
    print(tabulate(table, tablefmt="fancy_grid",
                   headers=("k", "x_k", "Точное решение", "Решение Тейлором", "Погрешность"),
                   floatfmt=("", "", ".15f", ".15f", "")))


def runge_kutta(x_arr: list[float], y0: float, N: int, h) -> list[float]:
    y_arr = [y0]
    for m in range(N + 1):
        k1 = h * f(x_arr[m], y_arr[m])
        k2 = h * f(x_arr[m] + h / 2, y_arr[m] + k1 / 2)
        k3 = h * f(x_arr[m] + h / 2, y_arr[m] + k2 / 2)
        k4 = h * f(x_arr[m] + h, y_arr[m] + k3)
        y_arr.append(y_arr[m] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return y_arr


def euler_method(x_arr: list[float], y0: float, N: int, h: float) -> list[float]:
    y_arr = [y0]
    for k in range(N + 1):
        y_arr.append(y_arr[k] + h * f(x_arr[k], y_arr[k]))
    return y_arr


def euler1_method(x_arr: list[float], y0: float, N: int, h: float) -> list[float]:
    y_arr = [y0]
    y2_arr = []
    for k in range(N + 1):
        y2_arr.append(y_arr[k] + h / 2 * f(x_arr[k], y_arr[k]))
        y_arr.append(y_arr[k] + h * f(x_arr[k] + h / 2, y2_arr[k]))
    return y_arr


def euler2_method(x_arr: list[float], y0: float, N: int, h: float) -> list[float]:
    y_arr = [y0]
    Y_arr = [None]
    for k in range(N + 1):
        Y_arr.append(y_arr[k] + h * f(x_arr[k], y_arr[k]))
        y_arr.append(y_arr[k] + h / 2 * (f(x_arr[k], y_arr[k]) + f(x_arr[k + 1], Y_arr[k + 1])))
    return y_arr


def adams_method(x_arr: list[float], y0: float, N: int, h: float, n: int) -> list[float]:
    x0 = x_arr[0]
    y_arr = taylor_method(x_arr, x0, y0, n)[:5]
    for m in range(4, N + 2):
        y_arr.append(y_arr[m] + 1 / 720 * h * (1901 * f(x_arr[m], y_arr[m])
                                               - 2774 * f(x_arr[m - 1], y_arr[m - 1])
                                               + 2616 * f(x_arr[m - 2], y_arr[m - 2])
                                               - 1274 * f(x_arr[m - 3], y_arr[m - 3])
                                               + 251 * f(x_arr[m - 4], y_arr[m - 4])))
    return y_arr


def print_method_table(x_arr: list[float], y_arr: list[float], start: int = 0):
    table = [(i, x, y) for i, (x, y) in enumerate(zip(x_arr, y_arr), start)]
    print(tabulate(table, tablefmt="fancy_grid", floatfmt=("", ".2f", ".15f"), headers=("i", "x_i", "y_i")))


def print_yn_errors(yn_arr: list[float], precise: float):
    err_arr = [abs(precise - yn_arr[i]) for i in range(len(yn_arr))]
    label_arr = ["Тейлора", "Адамса", "Рунге-Кутта", "Эйлера", "Эйлера I", "Эйлера II"]
    table = [(label, yn, err) for label, yn, err in zip(label_arr, yn_arr, err_arr)]
    print(tabulate(table, tablefmt="fancy_grid", headers=("Метод", "Значение y_N", "Погрешность"),
                   floatfmt=("", ".15f", "")))


def main():
    print("Численное решение задачи Коши для обыкновенного ДУ 1-го порядка. Вариант 7")
    print("ДУ: y'(x) = -y(x) + cos(x)\n"
          "Задача Коши: y(0) = 1")
    print("Точное решение задачи Коши: y(x) = sin(x) / 2 + cos(x) / 2 + 1 / 2e^x")
    x0 = 0
    y0 = 1
    ans = "y"
    n = int(input("Введите число слагаемых для метода Тейлора (по умолчанию 6): n = ") or 6) - 1
    while ans in ("y", "Y"):
        N = int(input("Введите количество точек для таблицы значений (по умолчанию N = 10): N = ") or 10)
        h = float(input("Введите шаг (по умолчанию h = 0.1): h = ") or 0.1)

        taylor_x_arr = [x0 + k * h for k in range(-2, N + 1)]
        taylor_arr = taylor_method(taylor_x_arr, x0, y0, n)
        print_taylor_method(taylor_x_arr, taylor_arr)

        adams_arr = adams_method(taylor_x_arr, y0, N, h, n)
        print(len(taylor_x_arr), len(adams_arr))
        print("Таблица для метода Адамса:")
        print_method_table(taylor_x_arr, adams_arr, -2)

        x_arr = [x0 + k * h for k in range(N + 1)]
        runge_arr = runge_kutta(x_arr, y0, N, h)
        print("Таблица для метода Рунге-Кутты:")
        print_method_table(x_arr, runge_arr)

        euler_arr = euler_method(x_arr, y0, N, h)
        print("Таблица для метода Эйлера:")
        print_method_table(x_arr, euler_arr)

        euler1_arr = euler1_method(x_arr, y0, N, h)
        print("Таблица для метода Эйлера I:")
        print_method_table(x_arr, euler1_arr)

        euler2_arr = euler1_method(x_arr, y0, N, h)
        print("Таблица для метода Эйлера II:")
        print_method_table(x_arr, euler2_arr)

        yn_arr = [taylor_solution(x_arr[N], x0, y0, n), adams_arr[-1], runge_arr[N], euler_arr[N], euler1_arr[N], euler2_arr[N]]
        precise = solution(x_arr[N])
        print_yn_errors(yn_arr, precise)

        ans = input("Хотите поменять значения N и h? (y/N): ") or "N"


if __name__ == '__main__':
    main()
