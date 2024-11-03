import numpy as np
from scipy import integrate


def f(x):
    return np.exp(x) * np.sin(x)


def precise_integral(a, b):
    def primitive(x):
        return 1 / 2 * np.exp(x) * (np.sin(x) - np.cos(x))

    return primitive(b) - primitive(a)


def P(n, x):
    funcs = (0.424,
             3.34 + 2.513 * x,
             -3.84 + 0.43 * x - 0.927 * (x ** 2),
             2 + 5.12 * x - 3.2156 * (x ** 2) + 0.45 * (x ** 3))
    return funcs[n]


def calculate_integrals(a: float, b: float, f) -> tuple:
    h = (b - a) / 3
    res = (["Левого прямоугольника", (b - a) * f(a)],
           ["Правого прямоугольника", (b - a) * f(b)],
           ["Среднего прямоугольника", (b - a) * f((a + b) / 2)],
           ["Трапеции", (b - a) / 2 * (f(a) + f(b))],
           ["Симпсона (параболы)", (b - a) / 6 * (f(a) + f(b) + 4 * f((a + b) / 2))],
           ["Трех восьмых", (b - a) / 8 * (f(a) + 3 * f(a + h) + 3 * f(a + 2 * h) + f(b))]
           )
    return res


def add_errors(res: tuple, precise: float):
    for arr in res:
        arr[1] = float(arr[1])
        arr.append(abs(precise - arr[1]))


def print_table(table: tuple):
    from tabulate import tabulate
    headers = ("Квадратурная формула", "Значение интеграла", "Погрешность")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".16f"))


def main():
    print("Приближенное вычисление интегралов по различным КФ. Вариант 7")
    a, b = map(float, input("Введите промежутки интегрирования (по умолчанию [0, 1]): ").split() or (0, 1))
    precise = precise_integral(a, b)
    print(f"Точное значение интеграла функции f(x) = sin(x)*e^x от {a} до {b}: {precise}")
    f_table = calculate_integrals(a, b, f)
    add_errors(f_table, precise)
    print_table(f_table)
    for i in range(4):
        print(f"Таблица для многочлена степени {i}:")
        p_func = lambda x: P(i, x)
        p_precise = integrate.quad(p_func, a, b)[0]
        p_table = calculate_integrals(a, b, p_func)
        add_errors(p_table, p_precise)
        print_table(p_table)


if __name__ == "__main__":
    main()
