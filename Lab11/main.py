import math

from math_methods import *
from sympy import symbols

variant = 29

a0, a1, a2, a3, a4 = symbols('a0 a1 a2 a3 a4')

x_down, x_up = -1, 1
y_down, y_up = 0, 0


def main_task():
    f = symbols('f', cls=Function)
    diff_eq_2 = Eq(np.sin(variant) * f(x).diff(x, x) + (1 + np.cos(variant) * x ** 2) * f(x), -1)
    print(f"Basic function: {diff_eq_2}")

    f0, f1, f2 = get_basis_functions()
    print(f"fi0(x)={f0}\nfi1(x)={f1}\nfi2(x)={f2}")

    check_is_linear(f1, f2)

    f = a0 * f0 + a1 * f1 + a2 * f2
    print(f"y(x)={f}")

    psi_x = Eq(np.sin(variant) * f.diff(x, x) + (1 + np.cos(variant) * x ** 2) * f, -1)
    print(psi_x.expand())

    result = collocation_method(f, psi_x)
    print(f"Collocation_method: {result.expand()}")

    Galerkin_result = Galerkin_method(f, psi_x)
    print(f"Galerkin_method: {Galerkin_result.expand()}")

    least_square_result = least_square_method(f, psi_x)
    print(f"Least_square_method: {least_square_result.expand()}")

    discrete_least_squares_result = discrete_least_squares(f, psi_x)
    print(f"Discrete_least_square_result: {discrete_least_squares_result.expand()}")

    return result, Galerkin_result, least_square_result, discrete_least_squares_result


def show_plots():
    t = np.arange(-1, 1, 0.01)

    y = [collocation_result.subs(x, i) for i in t]
    y2 = [Galerkin_result.subs(x, i) for i in t]
    y1 = [least_square_result.subs(x, i) for i in t]
    y1_1 = [discrete_least_squares_res.subs(x, i) for i in t]

    plt.plot(t, y, label=r'f_1(x) - collocation method')
    plt.plot(t, y1, label=r'f_2(x) - Least square method')
    plt.plot(t, y1_1, label=r'f_3(x) - Discrete least square method')
    plt.plot(t, y2, label=r'f_4(x) - Galerkin method')

    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$f(x)$', fontsize=14)

    plt.grid(True)
    plt.legend(loc='best', fontsize=12)

    plt.show()

    return


def test_least_square_method():
    def get_basis_func():
        return 0, x * (1 - x), x ** 2 * (1 - x)

    f = symbols('f', cls=Function)
    diff_eq_2 = Eq(f(x).diff(x, x) + f(x), -x)
    print(f"Basic function: {diff_eq_2}")

    f0, f1, f2 = get_basis_func()
    print(f"fi0(x) = {f0}\nfi1(x) = {f1}\nfi2(x) = {f2}")

    check_is_linear(f1, f2)

    f = f0 + a1 * f1 + a2 * f2
    print(f"y(x)={f}")

    psi_x = Eq(f.diff(x, x) + f, -x)

    result = least_square_method(f, psi_x, 0, 1)
    print(result.expand())

    return result


def test_discrete_least_square_method():
    def get_basis_func():
        return 0, x * (1 - x), x ** 2 * (1 - x)

    def get_collocation_points():
        return 0.2, 0.5, 0.9

    f = symbols('f', cls=Function)
    diff_eq_2 = Eq(f(x).diff(x, x) + f(x), -x)
    print(f"Basic function: {diff_eq_2}")

    f0, f1, f2 = get_basis_func()
    print(f"fi0(x) = {f0}\nfi1(x) = {f1}\nfi2(x) = {f2}")

    check_is_linear(f1, f2)

    f = f0 + a1 * f1 + a2 * f2
    print(f"y(x)={f}")

    psi_x = Eq(f.diff(x, x) + f, -x)

    result = discrete_least_squares(f, psi_x, get_collocation_points)
    print(result.expand())

    return result


def test_collocation_method():
    def get_point():
        return -0.5, 0, 0.5

    def get_basis_func():
        return 0, 1 - x ** 2, x ** 2 * (1 - x ** 2)

    f = symbols('f', cls=Function)
    diff_eq_2 = Eq(f(x).diff(x, x) + (1 + x ** 2) * f(x), -1)
    print(f"Basic function: {diff_eq_2}")

    f0, f1, f2 = get_basis_func()
    print(f"fi0(x) = {f0}\nfi1(x) = {f1}\nfi2(x) = {f2}")

    check_is_linear(f1, f2)

    f = f0 + a1 * f1 + a2 * f2
    print(f"y(x)={f}")

    psi_x = Eq(f.diff(x, x) + (1 + x ** 2) * f, -1)

    result = collocation_method(f, psi_x, get_point)
    print(result)

    return


def test_galerkin_method():
    def get_basis_func():
        return 0, x * (1 - x), x ** 2 * (1 - x)

    f = symbols('f', cls=Function)
    diff_eq_2 = Eq(f(x).diff(x, x) + f(x), x)
    print(f"Basic function: {diff_eq_2}")

    f0, f1, f2 = get_basis_func()
    print(f"fi0(x) = {f0}\nfi1(x) = {f1}\nfi2(x) = {f2}")

    check_is_linear(f1, f2)

    f = f0 + a1 * f1 + a2 * f2
    print(f"y(x)={f}")

    psi_x = Eq(f.diff(x, x) + f, x)

    result = Galerkin_method(f, psi_x, get_basis_func, 0, 1)
    print(result.expand())
    print(result.subs(x, 0.25))
    print(result.subs(x, 0.5))
    print(result.subs(x, 0.75))

    return


def test2_galerkin_method():
    def get_basis_func():
        return 1 - x, x * (1 - x), x ** 2 * (1 - x), x**3 * (1 - x)

    f = symbols('f', cls=Function)
    diff_eq_2 = Eq(f(x).diff(x, x) + x * f(x).diff(x) + f(x), 2*x)
    print(f"Basic function: {diff_eq_2}")

    f0, f1, f2, f3 = get_basis_func()
    print(f"fi0(x) = {f0}\nfi1(x) = {f1}\nfi2(x) = {f2}")

    check_is_linear(f1, f2)

    f = f0 + a1 * f1 + a2 * f2 + a3 * f3
    print(f"y(x)={f}")

    psi_x = Eq(f.diff(x, x) + x * f.diff(x) + f, 2*x)
    print(psi_x.expand())

    result = Galerkin_method(f, psi_x, get_basis_func, 0, 1)
    print(result.expand())
    print(result.subs(x, 0.25))
    return


if __name__ == '__main__':
    collocation_result, Galerkin_result, least_square_result, discrete_least_squares_res = main_task()
    show_plots()

    # test_collocation_method()
    # test_least_square_method()
    # test_discrete_least_square_method()
    # test_galerkin_method()
    # test2_galerkin_method()


