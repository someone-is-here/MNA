from sympy import Symbol, diff, sin, cos, series, exp, symbols, Function, Eq, dsolve, linsolve, solve, integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

variant = 29
x = Symbol('x')
y = Symbol('y')
a0, a1, a2, a3 = symbols('a0 a1 a2 a3')


def basis_functions(i):
    if i == 0:
        return 0

    return (x**i) * (1-x**2)


def check_is_linear(fi_1, fi_2):
    if diff(fi_2/fi_1) == 0:
        raise Exception('Basic system error! Linearly dependent functions.')

    return


def get_collocation_points():
    return -0.5, 0, 0.5


def get_basis_functions():
    return 0, 1 - x ** 2, x ** 2 * (1 - x ** 2)


def get_equation(func, points):
    return func.subs(x, points[0]), func.subs(x, points[1]), func.subs(x, points[2])


def collocation_method(f, psi_x,
                       get_collocation_point_default=get_collocation_points):
    (x1, x2, x3) = get_collocation_point_default()
    print(x1, x2, x3)

    eq_1_with_x1 = psi_x.subs(x, x1)
    eq_2_with_x2 = psi_x.subs(x, x2)
    eq_3_with_x3 = psi_x.subs(x, x3)

    res = solve((eq_1_with_x1, eq_2_with_x2, eq_3_with_x3), (a0, a1, a2))
    result_function = f.subs(a1, res[a1]).subs(a2, res[a2])

    return result_function


def least_square_method(f, psi_x,
                        a=-1, b=1):
    integral_psi_square_1 = integrate((psi_x.lhs - psi_x.rhs) * (psi_x.lhs - psi_x.rhs).diff(a1), (x, a, b))
    integral_psi_square_2 = integrate((psi_x.lhs - psi_x.rhs) * (psi_x.lhs - psi_x.rhs).diff(a2), (x, a, b))
    print(integral_psi_square_1, integral_psi_square_2)

    res = solve((integral_psi_square_1, integral_psi_square_2), (a1, a2))
    print(f"res: {res}")

    result_function = f.subs(a1, res[a1]).subs(a2, res[a2])

    return result_function


def discrete_least_squares(f, psi_x, get_collocation_point_default=get_collocation_points):
    x1, x2, x3 = get_collocation_point_default()
    sum_psi_x = ((psi_x.lhs - psi_x.rhs) ** 2).subs(x, x1) + \
                ((psi_x.lhs - psi_x.rhs) ** 2).subs(x, x2) + \
                ((psi_x.lhs - psi_x.rhs) ** 2).subs(x, x3)

    eq_1 = sum_psi_x.diff(a1)
    eq_2 = sum_psi_x.diff(a2)

    res = solve((eq_1, eq_2), (a1, a2))
    print(f"res: {res}")

    result_function = f.subs(a1, res[a1]).subs(a2, res[a2])

    return result_function


def Galerkin_method(f, psi_x,
                    get_basis_functions_default=get_basis_functions,
                    a=-1, b=1):

    f0, f1, f2 = get_basis_functions_default()

    eq_1_with_x1 = integrate((psi_x.lhs - psi_x.rhs) * f1, (x, a, b))
    eq_2_with_x2 = integrate((psi_x.lhs - psi_x.rhs) * f2, (x, a, b))

    print(f"psi1({a0},{a1},{a2}) = {eq_1_with_x1}")
    print(f"psi2({a0},{a1},{a2}) = {eq_2_with_x2}")

    res = solve((eq_1_with_x1, eq_2_with_x2), (a1, a2))
    print(res)

    result_function = f.subs(a1, res[a1]).subs(a2, res[a2])

    return result_function
