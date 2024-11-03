from typing import Callable, Tuple, Any
from tabulate import tabulate
import numpy as np

V = 7
k = (V % 5) + 1


class F1:
    @staticmethod
    def f(x):
        return np.log(1 + x)

    @staticmethod
    def df(x):
        return 1 / (1 + x)

    @staticmethod
    def d2f(x):
        return - 1 / ((1 + x) **2 )


class F2:
    @staticmethod
    def f(x):
        return np.exp(1.5 * k * x)

    @staticmethod
    def df(x):
        return 1.5 * k * np.exp(1.5 * k * x)

    @staticmethod
    def d2f(x):
        return 2.25 * k ** 2 * np.exp(1.5 * k * x)


def create_table(num_values: int, x0: float, h: float, func: Callable) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x0, x0 + h * num_values, num=num_values + 1)
    y = func(x)
    return x, y


def print_table(x_arr: np.ndarray, y_arr: np.ndarray):
    x = ["x"] + list(map(float, x_arr))
    y = ["y"] + list(map(float, y_arr))
    table = (x, y)
    print(tabulate(table, tablefmt="fancy_grid", floatfmt=".4f"))


def first_derivative(ind: int, x_arr: np.ndarray, y_arr: np.ndarray, h: float, df: Callable) -> tuple:
    if ind == 0:
        d = (-3 * y_arr[0] + 4 * y_arr[1] - y_arr[2]) / (2 * h)
    elif ind == len(x_arr) - 1:
        d = (3 * y_arr[-1] - 4 * y_arr[-2] + y_arr[-3]) / (2 * h)
    else:
        d = (y_arr[ind + 1] - y_arr[ind - 1]) / (2 * h)
    return float(d), abs(df(x_arr[ind]) - d)


def new_first_derivative(ind: int, x_arr: np.ndarray, y_arr: np.ndarray, h: float, df: Callable) -> tuple:
    if ind == 0:
        d = (-25 * y_arr[0] + 48 * y_arr[1] - 36 * y_arr[2] + 16 * y_arr[3] - 3 * y_arr[4]) / (12 * h)
    elif ind == 1:
        d = (-3 * y_arr[0] - 10 * y_arr[1] + 18 * y_arr[2] - 6 * y_arr[3] + y_arr[4]) / (12 * h)
    elif ind == len(x_arr) - 2:
        d = (3 * y_arr[-1] + 10 * y_arr[-2] - 18 * y_arr[-3] + 6 * y_arr[-4] - y_arr[-5]) / (12 * h)
    elif ind == len(x_arr) - 1:
        d = (25 * y_arr[-1] - 48 * y_arr[-2] + 36 * y_arr[-3] - 16 * y_arr[-4] + 3 * y_arr[-5]) / (12 * h)
    else:
        d = (y_arr[ind - 2] - 8 * y_arr[ind - 1] + 8 * y_arr[ind + 1] - y_arr[ind + 2]) / (12 * h)
    return float(d), abs(df(x_arr[ind]) - d)


def second_derivative(ind: int, x_arr: np.ndarray, y_arr: np.ndarray, h: float, d2f: Callable) -> tuple | None:
    if ind == 0:
        d = (2 * y_arr[0] - 5 * y_arr[1] + 4 * y_arr[2] - y_arr[3]) / (h ** 2)
    elif ind == len(x_arr) - 1:
        d = (2 * y_arr[-1] - 5 * y_arr[-2] + 4 * y_arr[-3] - y_arr[-4]) / (h ** 2)
    else:
        d = (y_arr[ind + 1] - 2 * y_arr[ind] + y_arr[ind - 1]) / (h ** 2)
    return float(d), abs(d2f(x_arr[ind]) - d)


def print_answer_table(x_arr: np.ndarray, y_arr: np.ndarray, h: float, df: Callable, d2f: Callable):
    def line(ind: int) -> tuple:
        return (x_arr[ind], y_arr[ind],
                *first_derivative(ind, x_arr, y_arr, h, df),
                *new_first_derivative(ind, x_arr, y_arr, h, df),
                *second_derivative(ind, x_arr, y_arr, h, d2f))

    table_labels = ("x", "f(x)", "f'(x) с погрешностью O(h^2)", "Погрешность",
                    "f'(x) с погрешностью O(h^4)", "Погрешность",
                    "f\"(x)", "Погрешность")
    table = (line(i) for i in range(len(x_arr)))
    print(tabulate(table, headers=table_labels, tablefmt="fancy_grid"))


def print_runge_romberg(ind: int, x_arr: np.ndarray, y_arr: np.ndarray, h: float, x_arr2: np.ndarray,
                        y_arr2: np.ndarray,
                        df: Callable, d2f: Callable):
    J1_h, err11 = first_derivative(ind, x_arr, y_arr, h, df)
    J1_h2, err12 = first_derivative(ind * 2, x_arr2, y_arr2, h / 2, df)
    J1 = (4 * J1_h2 - J1_h) / 3
    err13 = abs(df(x_arr[ind]) - J1)
    res1 = (x_arr[ind], y_arr[ind], J1_h, err11, J1_h2, err12, J1, err13),
    table_labels = ("x", "f(x)", "J(h)", "Погрешность",
                    "J(h/2)", "Погрешность",
                    "𝐽", "Погрешность")
    print(f"Уточненные значения первой производной в точке x = {x_arr[ind]}:")
    print(tabulate(res1, headers=table_labels, tablefmt="fancy_grid"))

    J2_h, err21 = second_derivative(ind, x_arr, y_arr, h, d2f)
    J2_h2, err22 = second_derivative(ind * 2, x_arr2, y_arr2, h / 2, d2f)
    J2 = (4 * J2_h2 - J2_h) / 3
    err23 = abs(d2f(x_arr[ind]) - J2)
    res2 = (x_arr[ind], y_arr[ind], J2_h, err21, J2_h2, err22, J2, err23),
    print(f"Уточненные значения второй производной в точке x = {x_arr[ind]}:")
    print(tabulate(res2, headers=table_labels, tablefmt="fancy_grid"))


def main():
    print("Нахождение производных таблично-заданной функции по формулам численного дифференцирования. Вариант 7")
    funcs = (F1, F2)
    ans = 1
    while ans != 4:
        if ans in (1, 3):
            print("На выбор предоставлены две функции:\n"
                  "1. f(x) = e^(-x) - x^2 / 2\n"
                  "2. f(x) = e^(4.5*x)")
            while (func_num := int(
                    input(
                        "Введите номер функции, для которой будет решаться задача (по умолчанию, 1): ") or 1)) not in (
                    1, 2):
                print("Неверный номер, введите 1 или 2!")
            F = funcs[func_num - 1]
            func = F.f
            df = F.df
            d2f = F.d2f
        while (num_values := int(
                input("Введите количество значений в таблице (большее или равное 5, по умолчанию 10): ") or 10)) < 5:
            print("Количество значений должно быть не меньше 5!")
        m = num_values - 1
        x0 = float(input("Введите начальное значение x0 в таблице (по умолчанию, 0): x0 = ") or 0)
        h = float(input("Введите шаг h в таблице (по умолчанию, 0.1): h = ") or 0.1)
        x_arr, y_arr = create_table(m, x0, h, func)
        print_table(x_arr, y_arr)
        print_answer_table(x_arr, y_arr, h, df, d2f)
        if ans == 3:
            while (j := int(
                    input(
                        "Введите индекс узла таблицы, для которого нужно уточнить производную: j = ") or 0)) > m or j < 0:
                print("Введенное значение не является индексом узла!")
            x_arr2, y_arr2 = create_table(2 * m, x0, h / 2, func)
            print_table(x_arr2, y_arr2)
            print_runge_romberg(j, x_arr, y_arr, h, x_arr2, y_arr2, df, d2f)

        print("\nДоступны следующие варианты: \n"
              "1. Поменять функцию и параметры таблицы\n"
              "2. Поменять только параметры таблицы\n"
              "3. Уточнить по Рунге-Ромбергу\n"
              "4. Выйти\n")
        ans = int(input("Введите номер следующего действия (по умолчанию 4): ") or 4)


if __name__ == '__main__':
    main()
