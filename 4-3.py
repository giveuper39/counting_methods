import numpy as np
from tabulate import tabulate
from scipy import integrate


class F:
    @staticmethod
    def f(x: float) -> float:
        return np.exp(x) * np.sin(x)

    @staticmethod
    def f_int(a, b):
        def primitive(x: float) -> float:
            return 1 / 2 * np.exp(x) * (np.sin(x) - np.cos(x))

        return primitive(b) - primitive(a)


def P(x: float, n: int) -> float:
    poly = [2.345, 3.21 * x - 1.65, 13 * x**2 - 1.65 * x + 5.4, 1.75 * x ** 3 + 2 * x - 0.412]
    return poly[n]


def calculate_integral_errors(f, a: float, b: float, m: int, j: float, include: str = "01234") -> list[list[str | float]]:
    h = (b - a) / m
    z_arr = [a + i * h for i in range(m + 1)]
    p = f(a) + f(b)
    w = sum(f(z_arr[i]) for i in range(1, m))
    q = sum(f(z_arr[i] + h / 2) for i in range(m))
    left_rect = h * (f(a) + w)
    right_rect = h * (w + f(b))
    mid_rect = h * q
    trap = h / 2 * (p + 2 * w)
    simpson = h / 6 * (p + 2 * w + 4 * q)
    formulae = [left_rect, right_rect, mid_rect, trap, simpson]
    f_names = ["ЛП", "ПП", "СП", "Трапеций", "Симпсона"]
    include_inds = map(int, list(include))
    res = [[f_names[i], formulae[i], err := abs(formulae[i] - j), err / abs(j)] for i in include_inds]
    return res


def runge_correction(res1: list[list[str | float]], res2: list[list[float]], l: int, j: float):
    d_arr = [0, 0, 1, 1, 3]
    f_names = ["ЛП", "ПП", "СП", "Трапеций", "Симпсона"]
    res = []
    for i in range(5):
        r = d_arr[i] + 1
        J_h = res1[i][1]
        J_hl = res2[i][1]
        J = (l ** r * J_hl - J_h) / (l ** r - 1)
        res.append([f_names[i], J_h, J_hl, J, err := abs(J - j), err / abs(J)])
    return res


def print_integral_table(table: list[list[str | float]]):
    headers = ("СКФ", "Значение интеграла", "Абсолютная погрешность", "Относительная погрешность")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=(".15f", ".15f")))


def print_runge_table(table: list[list[str | float]]):
    headers = ("СФК", "J(h)", "J(h / l)", "J", "Абсолютная погрешность", "Относительная погрешность")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=(".15f", ".15f", ".15f", ".15f")))


def precision_check(n: int, a: float, b: float, m: int):
    f = lambda x: P(x, n)
    j = integrate.quad(f, a, b)[0]
    if n == 0:
        include = "01234"
    elif n == 1:
        include = "234"
    else:
        include = "4"
    print_integral_table(calculate_integral_errors(f, a, b, m, j, include))



def main():
    f = F.f
    print("Приближенное вычисление интеграла по СКФ")
    a, b = map(float, input("Введите промежутки интегрирования a и b (по умолчанию, [0, 1]): ").split() or (0, 1))
    m = int(input(f"Введите число разбиений отрезка [a, b] = [{a}, {b}] (по умолчанию, 10): m = ") or 10)
    j = F.f_int(a, b)
    print(f"Точное значение интеграла f(x) = sin(x) * e^x: {j}")
    table1 = calculate_integral_errors(f, a, b, m, j)
    print(f"Значения интеграла по СКФ при {m = }:")
    print_integral_table(table1)
    while(l := int(input(f"Сейчас мы увеличим значение {m = } в l раз. Введите l (по умолчанию, 10): l = ") or 10)) == 1:
        print("Минимальное значение l = 2!")
    table2 = calculate_integral_errors(f, a, b, m * l, j)
    print(f"Значения интеграла по СКФ при m = {m * l}:")
    print_integral_table(table2)
    runge = runge_correction(table1, table2, l, j)
    print("Уточнение по Рунге-Ромбергу:")
    print_runge_table(runge)
    for n in range(4):
        print(f"Проверка СКФ для многочлена степени {n}:")
        precision_check(n, a, b, m)


if __name__ == '__main__':
    main()
