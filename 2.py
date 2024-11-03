import numpy as np
from tabulate import tabulate


def f(x):
    # return np.exp(-x) - x ** 2 / 2
    return 4 * (x ** 8) + 2 * (x ** 7) - 4 * (x ** 5) - 3 * (x ** 4) + 8 * (x ** 2) - 3 * x + 2


def get_nodes(a: float, b: float, m: int) -> np.ndarray:
    nodes = np.linspace(a, b, m + 1)
    return nodes


def print_table(nodes: np.ndarray, node_values: np.ndarray):
    x = list(map(float, nodes))
    y = list(map(float, node_values))
    table = (['x'] + x, ['y'] + y)
    print(tabulate(table, tablefmt="fancy_grid", floatfmt=".4f"))


def draw_graph(a: float, b: float, func1, func2):
    import matplotlib.pyplot as plt
    plt.axhline(0, color='red')
    plt.axvline(0, color='red')
    x = np.linspace(a, b, 10000)
    y1 = func1(x)
    y2 = func2(x)
    plt.plot(x, np.c_[y1, y2], label=["f(x)", "L(x)"])
    plt.legend()
    plt.grid()
    plt.show(block=False)
    plt.pause(0.001)


def lagrange_value(n: int, x: float | np.ndarray, nodes: np.ndarray) -> float | np.ndarray:
    def multi(k: int, x: float | np.ndarray) -> float:
        res = 1
        for j in range(n):
            if j == k:
                continue
            res *= (x - nodes[j])
        return res

    l = lambda k, x: multi(k, x) / multi(k, nodes[k])
    L = 0
    for i in range(n):
        L += l(i, x) * f(nodes[i])
    return L


def find_ind(arr: np.ndarray, a: float) -> int:
    delta = np.inf
    i = 0
    while i < len(arr) and abs(a - arr[i]) < delta:
        delta = abs(a - arr[i])
        i += 1
    return i - 1


def slice_arr(arr: np.ndarray, n: int, ind: int) -> np.ndarray:
    m1 = int(np.floor(n / 2))
    m2 = int(np.ceil(n / 2)) + 1
    # print(m1, m2)
    if ind - m1 < 0:
        ind1 = 0
        ind2 = n + 1
    elif m2 + ind > len(arr):
        ind2 = len(arr)
        ind1 = len(arr) - n - 1
    else:
        ind1 = ind - m1
        ind2 = ind + m1
    # print(ind1, ind2)
    # print(arr[ind1: ind2 + 1])
    try:
        return arr[ind1: ind2 + 1]
    except IndexError:
        return arr


def main():
    print("Задача алгебраического интерполирования функции, вариант 7.")
    x_num = int(input("Введите число значений в таблице (по умолчанию: 16): x_num = ") or 16)
    m = x_num - 1
    print(f"Максимальная степень интерполяционного многочлена m = {m}")
    a, b = map(float, input(
        "Введите начало и конец отрезка [a, b], из которого выбираются узлы интерполяции (по умолчанию - [0, 1]): ").split()
               or (0, 1))
    nodes_arr = get_nodes(a, b, m)
    node_values = f(nodes_arr)
    print_table(nodes_arr, node_values)
    ok = "y"
    n_prev = -1
    nodes = nodes_arr
    while ok in ("y", "Y"):
        n = int(input(f"Введите степень интерполяционного многочлена <= {m}: n = ") or m)
        while n > m:
            print(f"Введенное n = {n} > максимальной степени многочлена m = {m}.")
            n = int(input(f"Введите степень интерполяционного многочлена <= {m}: n = ") or m)
        x = float(input("Введите точку интерполирования: x = "))
        if x in nodes_arr:
            print(f"Выбранный x совпадает с одним из узлов интерполяции x = {x}")
        else:
            if n != n_prev:
                if n < m:
                    nodes = slice_arr(nodes_arr, n, find_ind(nodes_arr, x))
                else:
                    nodes = nodes_arr
            # print(nodes)
            l = lagrange_value(len(nodes), x, nodes)
            print(f"Значение многочлена Лагранжа в точке x = {x}: L(x) = {l}. \n"
                  f"Значение функции в точке x: f(x) = {f(x)}\n"
                  f"Абсолютная фактическая погрешность составляет {abs(f(x) - l)}.")
            draw_graph(a, b, f, lambda z: lagrange_value(n, z, nodes))
            n_prev = n
        ok = input("Хотите ввести новые параметры n и x? (y/N): ") or "n"


if __name__ == "__main__":
    main()
