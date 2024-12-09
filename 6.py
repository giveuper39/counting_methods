import numpy as np


def f(x):
    return np.sin(x)


def pv(x):
    return abs(x - 0.5)


def p1(x):
    return 1


def p2(x):
    return 1 / np.sqrt(1 - x ** 2)


def main():
    print("Приближенное вычисление интегралов с помощью КФ НАСТ. Вариант 7")



if __name__ == "__main__":
    main()