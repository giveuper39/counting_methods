# f(x) = 10*cos(x) - 0.1*x^2
# [A, B] = [-8, 2]
from typing import Any
import numpy as np


class Function:
    @staticmethod
    def f(x):
        return 10 * np.cos(x) - 0.1 * (x ** 2)

    @staticmethod
    def df(x):
        return -10 * np.sin(x) - 0.2 * x


def main():
    print("Нахождение корней нечетной степени трансцендетного уравнения 10*cos(x) - 0.1x^2 = 0 на отрезке [A, B]")
    a, b = map(float, input("Введите через пробел левую и правую границу отрезка, "
                            "на котором производится поиск корней (по умолчанию A=-8, B=2): ").split() or (-8, 2))
    root_segments = []
    conf_segment = "y"
    while conf_segment in ("y", "Y"):
        num_segments = int(input("Введите количество разбиений отрезка [A, B] (по умолчанию - 10000): ") or 10000)
        precision = (b - a) / num_segments
        x_arr = np.arange(a, b, precision)
        y_arr = Function.f(x_arr)
        draw_plot(x_arr, y_arr)
        root_segments_tuple = root_separation(x_arr, y_arr, num_segments)
        roots = []
        root_segments = separate_found_roots(root_segments_tuple, roots)
        if roots:
            print("После отделения корней были найдены следующие корни уравнения:")
            for i in roots:
                print("x = ", i)
        print("После отделения корней были найдены следующие отрезки перемены знака: ")
        for i, seg in enumerate(root_segments):
            print(f"{i + 1}. [{seg[0]:.4f}, {seg[1]:.4f}]")
        conf_segment = input("Хотите ли вы попробовать другое количество разбиений? (y/N): ") or "n"
    num = int(input(f"Введите номер отрезка для анализа или 0, если требуется найти все корни на [{a}, {b}]: "))
    while num > len(root_segments):
        num = int(input(f"Введите число от 0 до {len(root_segments)}: "))
    conf_epsilon = "y"
    while conf_epsilon in ("y", "Y"):
        e = int(input("Введите точность приближения (по умолчанию - 10^-6): 10^-") or 6)
        if num == 0:
            for seg in root_segments:
                find_roots_on_segment(seg, e)
        else:
            find_roots_on_segment(root_segments[num - 1], e)
        conf_epsilon = input("Хотите попробовать другую точность приближения? (y/N): ") or "n"


def find_roots_on_segment(segment: tuple, e: int):
    a, b = segment
    print(f"------------------------------------------------------\n"
          f"Отрезок: [{segment[0]:.4f}, {segment[1]:.4f}]\n")
    bisection_method(a, b, e)
    newtons_method(a, b, e)
    mod_newtons_method(a, b, e)
    secant_method(a, b, e)
    print(f"------------------------------------------------------\n")


def bisection_method(a: float, b: float, e: int):
    xs = float((a + b) / 2)
    steps = 0
    while b - a > 2 * 10 ** (-e):
        c = (a + b) / 2
        if Function.f(a) * Function.f(c) < 0:
            b = c
        else:
            a = c
        steps += 1
    x = (a + b) / 2
    print_method_info("Методом бисекции", x, steps, b - a, e, xs)


def newtons_method(a: float, b: float, e: int):
    x0 = (a + b) / 2
    xs = float(x0)
    steps = 1
    x1 = x0 - Function.f(x0) / Function.df(x0)
    while abs(x1 - x0) > 10 ** (-e):
        x0 = x1
        x1 = x0 - Function.f(x0) / Function.df(x0)
        steps += 1
    print_method_info("Методом Ньютона", x1, steps, abs(x1 - x0), e, xs)


def mod_newtons_method(a: float, b: float, e: int):
    x0 = (a + b) / 2
    xs = float(x0)
    c = Function.df(x0)
    steps = 1
    x1 = x0 - Function.f(x0) / c
    while abs(x1 - x0) > 10 ** (-e):
        x0 = x1
        x1 = x0 - Function.f(x0) / c
        steps += 1
    print_method_info("Модифицированным методом Ньютона", x1, steps, abs(x1 - x0), e, xs)


def secant_method(a: float, b: float, e: int):
    x0 = a
    x1 = b
    xs1 = float(a)
    xs2 = float(b)
    x2 = x1 - Function.f(x1) / (Function.f(x1) - Function.f(x0)) * (x1 - x0)
    steps = 1
    while abs(x2 - x1) > 10 ** (-e):
        x0 = x1
        x1 = x2
        x2 = x1 - Function.f(x1) / (Function.f(x1) - Function.f(x0)) * (x1 - x0)
        steps += 1
    print_method_info("Методом секущих", x2, steps, abs(x2 - x1), e, xs1, xs2)


def print_method_info(method: str, x: float, steps: int, root_dist: float, e: int, x0: float, x1: float = 0):
    start = f"x0 = {x0:.5f}" if x1 == 0 else f"x0 = {x0:.5f}, x1 = {x1:.5f}"
    print(
        f"{method} на данном отрезке с точностью e = 10^-{e} был найден корень x = {x}. \n"
        f"Начальное приближение: {start}. Количество шагов для достижения заданной точности {steps}. Длина последнего отрезка {root_dist}.\n"
        f"Абсолютная величина невязки {abs(Function.f(x))}.\n")


def separate_found_roots(root_segments_tuple: tuple, roots: list) -> list[tuple]:
    root_segments = []
    for i in range(len(root_segments_tuple)):
        if root_segments_tuple[i][0] == root_segments_tuple[i][1]:
            roots.append(root_segments_tuple[i][0])
        else:
            root_segments.append(root_segments_tuple[i])
    return root_segments


def draw_plot(x_arr: np.ndarray, y_arr: np.ndarray):
    import matplotlib.pyplot as plt
    plt.axhline(0, color='red')
    plt.axvline(0, color='red')
    plt.plot(x_arr, y_arr)
    plt.grid()
    plt.show(block=False)
    plt.pause(0.001)


def root_separation(x_arr: np.ndarray, y_arr: np.ndarray, num_segments: int) -> tuple[tuple[Any, Any], ...]:
    root_segments = []
    for i in range(num_segments - 1):
        x1, x2 = x_arr[i], x_arr[i + 1]
        y1, y2 = y_arr[i], y_arr[i + 1]
        if y1 * y2 < 0:
            root_segments.append((x1, x2))
        else:
            if y1 == 0:
                root_segments.append((x1, x1))
            if y2 == 0:
                root_segments.append((x2, x2))

    return tuple(root_segments)


if __name__ == "__main__":
    main()
