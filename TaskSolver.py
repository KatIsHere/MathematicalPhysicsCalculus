#Dubska Kate, 2018
"""Build an approximation of a solution to differential equation, 
using collocation method and method of least squares.
Compare resoults to the real solution."""

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np
import math


class Differential:
    """Finds a solution to the differential equation of type:
                -(k(x)u'(x))' + p(x)u'(x) + q(x)u(x) = f(x),    x in [a, b]
            with terms:    
                -k(a)*u(a)' + alfa1*u(a) = nue1
                k(b)*u(b)' + alfa2*u(b) = nue2
        k, p and q functions:
                k(x) = k1*cos(k2*x) + k3
                p(x) = p1*sin(p2*x) + p3
                q(x) = q1*cos(q2*x) + q3
        and exact solution is:
                u(x) = m1*sin(m2*x) + m3

        parameters can be set up in the class constructor or set_coefs function
        IMPORTANT: a <= b, else they would be set to 0
    """
    def __init__(self, a, b, k1 = 1, k2 = 1, k3 = 1, p1 = 1, 
                      p2 = 1, p3 = 1, q1 = 1, q2 = 1, q3 = 1, 
                      m1 = 1, m2 = 1, m3 = 1, alfa1 = 1, alfa2 = 1):
        self.solution_ci = None
        self._exact_solution = None
        self.set_interval(a ,b)
        self.set_coefs(k1, k2, k3, p1, p2, p3, 
            q1, q2, q3, m1, m2, m3, alfa1, alfa2)

    def set_interval(self, a, b):
        if(a <= b):
            self.a = a
            self.b = b
        else:
            self.a = self.b = 0

    def set_coefs(self, k1, k2, k3, p1, p2, p3, 
                    q1, q2, q3, m1, m2, m3, alfa1, alfa2):
        """Setting up parameters and functions"""
        self._exact_solution = lambda x: m1*math.sin(x * m2) + m3
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.q1, self.q2, self.q3 = q1, q2, q3
        self._alfa1, self._alfa2 = alfa1, alfa2
        self.k = lambda x: k1*math.cos(k2*x) + k3  
        self.p = lambda x: p1*math.sin(p2*x) + p3
        self.q = lambda x: q1*math.cos(q2*x) + q3
        self.k_der = lambda x: -k2*k1*math.sin(k2*x)        # k'(x) - derivative for operator calculations 
        self._nue1 = -self.k(self.a)*m1*m2*math.cos(self.a*m2) + alfa1*self._exact_solution(self.a)
        self._nue2 = self.k(self.b)*m1*m2*math.cos(self.b*m2) + alfa2*self._exact_solution(self.b)
        self._f = lambda x: -self.k(x)*(-m1*m2*m2*math.sin(m2*x)) - self.k_der(x)*m1*m2*math.cos(x*m2) \
                            + self.p(x)*m1*m2*math.cos(m2*x) + self.q(x)*self._exact_solution(x)
        self.__substitute_nue()
        
    def init_base_func_number(self, n):
        assert n >= 2    # there should be at least 2 base functions
        self.n = n
        dist = self.b - self.a
        self.__A = self.b + self.k(self.b)*dist/(2*self.k(self.b) + self._alfa2*dist)
        self.__B = self.a - self.k(self.a)*dist/(2*self.k(self.a) + self._alfa1*dist)

    def __base_phi(self, x, i):
        """Base function phi_i(x)
        INPUT: x - value, i - phi function number, i in range(self.n)
        OUTPUT: value of phi_{i}(x)"""
        res = 0
        if i == 0:
            res = (x - self.a)**2 * (x - self.__A)
        elif i == 1:
            res = (self.b - x)**2 * (self.__B - x)
        else:
            res = (x - self.a)**2 * math.pow((self.b - x), i)
        return res 

    def __base_dphi(self, x, i):
        """Base functions' phi_i(x) first derivative -- for accurate calculations
        INPUT: x - value, i - phi function number, i in range(self.n)
        OUTPUT: value of phi_{i}'(x)"""
        res = 0
        if i == 0:
            res = 2*(x - self.a)*(x - self.__A) + (x - self.a)**2
        elif i == 1:
            res = -2*(self.b - x)*(self.__B - x) - (self.b - x)**2
        else:
            res = 2*(x - self.a)*((self.b - x)**i) - i*((x - self.a)**2)*((self.b - x)**(i - 1))
        return res 

    def __base_d2phi(self, x, i):
        """Base functions' phi_i(x) second derivative -- for accurate calculations
        INPUT: x - value, i - phi function number, i in range(self.n)
        OUTPUT: value of phi_{i}''(x)"""
        res = 0
        if i == 0:
            res = 6*x - 2*self.__A - 4*self.a 
        elif i == 1:
            res = 4*self.b + 2*self.__B - 6*x
        elif i >= 2:
            res = 2*(self.b - x)**i - 2*(x -self.a)*i*((self.b - x)**(i - 1))   \
                    - 2*i*((self.b - x)**(i - 1))*(x - self.a) \
                    + i*(i - 1)*((x - self.a)**2) * ((self.b - x)**(i - 2))
        return res

    def __substitute_nue(self):
        # psi = A_psi * x + B_psi
        # u(x) = v(x) + psi(x)
        self._A_psi = 0
        self._B_psi = 0
        if self._nue1 != 0 or self._nue2 != 0:
            self._A_psi = (self._nue2 * self._alfa1 - self._nue1*self._alfa2)/(self.k(self.b) * self._alfa1 +   \
                                self.k(self.a)*self._alfa2 + self._alfa1*self._alfa2*(self.b - self.a))
            self._B_psi = (self._nue1 + self._A_psi*(self.k(self.a) - self._alfa1*self.a))/self._alfa1
        
        self.f = lambda x: self._f(x) - self.q(x)*(self._A_psi*x + self._B_psi) - self._A_psi * self.p(x)   \
                            + self.k_der(x)*self._A_psi

    def __operator(self, x_j, func_numb):
        """Operator A(phi_{i}) from the task
            * INPUT:    x_j - value
                        func_numb - number of the base function"""
        phi = self.__base_phi(x_j, func_numb)
        phi_diff = self.__base_dphi(x_j, func_numb) 
        k_diff = self.k_der(x_j)
        sec_phi_diff = self.__base_d2phi(x_j, func_numb)
        return -(k_diff * phi_diff + self.k(x_j) * sec_phi_diff) + \
                    self.p(x_j)*phi_diff + self.q(x_j)*phi

    # collocation solve method
    def collocation_solve(self, cheb_dots):
        """Find approximate solution of the task with collocation method
        * INPUT: cheb_dots - Chebishow from segment [a, b] in which 
                 the approximate solution should be the nearrest"""
        assert len(cheb_dots) == self.n
        C = np.array([[1.0 for j in range(self.n)] for i in range(self.n)])
        self.solution_ci = None
        F = []
        for j in range(self.n):
            F.append(self.f(cheb_dots[j]))
            for i in range(self.n):
                C[j][i] = self.__operator(cheb_dots[j], i)
        self.solution_ci = np.linalg.solve(C, F)

    # least squares solve method
    def least_square_solve(self):
        """Find approximate solution of the task with least square method"""
        C = np.array([[1.0 for j in range(self.n)] for i in range(self.n)])
        F = np.array([1.0 for j in range(self.n)])
        self.solution_ci = None
        for j in range(self.n):
            for i in range(self.n):
                C[j][i] = integrate.quad((lambda x: self.__operator(x, i)*self.__operator(x, j)), self.a, self.b)[0]
            F[j] = integrate.quad((lambda x: self.f(x)*self.__operator(x, j)), self.a, self.b)[0]
        self.solution_ci = np.linalg.solve(C, F)

    def solution_difference(self):
        # (u, v) = int{a, b} u*v dx
        # ||u - v||^2 = (u - v, u - v) = int{a, b} (u(x) - v(x))^2 dx
        inner_sum = lambda x: (self._exact_solution(x) - self.solution(x))**2
        diff = integrate.quad(inner_sum, self.a, self.b)[0]
        return math.sqrt(diff)

    def solution(self, x):
        suma = self._A_psi*x + self._B_psi
        for i in range(self.n):
            suma += self.solution_ci[i] * self.__base_phi(x, i)
        return suma

    def __str__(self):
        task = "-((" + str(self.k1) + "cos(" + str(self.k2) + "x) + " \
                + str(self.k3) + ") u')' + (" + str(self.p1) + "sin(" + str(self.p2) + "x) + " \
                + str(self.p3) + ") u' + (" + str(self.q1) + "cos(" + str(self.q2) + "x) + " \
                + str(self.q3) + ") u = \n                " + str(self.k1 * self.k2* self.m1 * self.m2) \
                + "cos("+str(self.m2) + "x) sin(" + str(self.k2) + "x) + " + \
                str(self.m1 * self.m2 * self.m2) \
                + "(" + str(self.k3) + " + " + str(self.k1) + "cos(" + \
                str(self.k2) + "x))sin(" + str(self.m2)\
                + "x) + (" + str(self.q3) + " + " + str(self.q1) + "cos(" + str(self.q2) + "x))" \
                + "("  + str(self.m3) + " + " + str(self.m1) + "sin(" + str(self.m2) + "x))"    \
                + str(self.m1*self.m2) + "cos(" + str(self.m2) + "x)(" + str(self.p3) + " + " + \
                str(self.p1) + "sin(" + str(self.p2) + "x))"
        return task

                 
def solve_and_plot(N = 15, dots = 100):
    """Example for how to work with the class
    Other methods will be added"""
    task = Differential(0, 1, k1=1, k2=3, k3=2, p1=1, p2=4, p3=2, q1=2,
                            q2=3, q3=2, m1=2, m2=7, m3=5, alfa1=3, alfa2=5)
    task.init_base_func_number(N)
    # EXACT VALUES
    val = np.linspace(0, 1, dots)
    real_val = []
    for x in val:
        real_val.append(task._exact_solution(x))
    # COLLOCATION METHOD
    x_cheb = np.linspace(0, 1, N)
    task.collocation_solve(x_cheb)
    approx_val_coll = []
    for x in val:
        approx_val_coll.append(task.solution(x))
    # LEAST SQUARE METHOD
    task.least_square_solve()
    approx_val_least = []
    for x in val:
        approx_val_least.append(task.solution(x))
    # PLOTTING
    plt.figure(figsize=(12, 7))
    plt.grid()
    plt.plot(val, real_val, '-r', label=r'$\ u(x)$')
    plt.plot(val, approx_val_coll, '--g', label="collocation method")
    plt.plot(val, approx_val_least, '-.y', label="least square method")
    plt.legend(fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    solve_and_plot()
