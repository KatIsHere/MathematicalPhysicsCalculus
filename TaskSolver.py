#@by Dubska Kate
"""Build an approximation of a solution to differential equation, 
using collocation method and method of least squares.
Compare resoults to the real solution.

The equation looks like:
-(k(x)u'(x))' + p(x)u'(x) + q(x)u(x) = f(x)
with added terms:
    -ku' + A*u = nue1,  x = a
    ku' + B*u = nue2,   x = b"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.integrate as integrate
import numpy as np
import math

def gen_x(a, b, N= 250):
  return [a + (b-a)/N * i for i in range(N+1)]

def evalIntSimpson(func, a, b):
  return (b-a)/6 * (func(a) + 4* func((a+b)/2) + func(b))

def IntSimpson(func, arr):
  return sum([evalIntSimpson(func, arr[k], arr[k+1]) for k in range(len(arr) - 1)])


def integral(f, a, b):
  # return quad(f, a, b)[0]
  return IntSimpson(f, gen_x(a, b, 50*(b-a)))

def deriv(f, x):
  dx = 0.0001
  return (f(x+dx) - f(x-dx))/(2*dx)

def dderiv(f, x):
  df = lambda x: deriv(f, x)
  return deriv(df, x)

class DifferentialTask:
        
    def __init__(self, a, b, k1 = 1, k2 = 1, k3 = 1, p1 = 1, 
            p2 = 1, p3 = 1, q1 = 1, q2 = 1, q3 = 1, 
            m1 = 1, m2 = 1, m3 = 1, alfa1 = 1, alfa2 = 1):
        self.a = a
        self.b = b
        self._base_func = []
        self._base_func_der = []
        self._base_func_sec_der = []
        self.set_coefs(k1, k2, k3, p1, p2, p3, 
            q1, q2, q3, m1, m2, m3, alfa1, alfa2)
    
    def set_coefs(self, k1, k2, k3, p1, p2, p3, 
            q1, q2, q3, m1, m2, m3, alfa1, alfa2):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.k = lambda x: k1*math.cos(k2*x) + k3   # k' = -k1*k2*math.sin(k2*x)
        self.p = lambda x: p1*math.sin(p2*x) + p3
        self.q = lambda x: q1*math.cos(q2*x) + q3
        self.k_der = lambda x: -k2*k1*math.sin(k2*x)
        self._nue1 = m1*m2*(-k2 - k1*math.cos(k2*self.a))*math.cos(m2*self.a) + \
                    alfa1*(m3 + m1*math.sin(m2*self.a))
        self._nue2 = m1*m2*(k3 + k1*math.cos(k2*self.b))*math.cos(m2*self.b) + \
                    alfa2*(m3 + m1*math.sin(m2*self.b))
        self._f = lambda x: k1*k2*m1*m2*math.cos(k2*x) + \
                m1*m2*m2*(k3+k1*math.cos(k2*x))*math.sin(m2*x) + \
                (q3 + q1*math.cos(q2*x))*(m3 + m1*math.sin(m2*x)) + \
                m1*m2*math.cos(m2*x)*(p3 + p1*math.sin(p2*x))
        self._alfa1 = alfa1
        self._alfa2 = alfa2
        self._base_func.clear()
        self._base_func_der.clear()
        self._base_func_sec_der.clear()
        self.__substitute_nue()
        self.__init_base_2func()
        
    def __init_base_2func(self):
        dist = self.b - self.a
        A = self.b + self.k(self.b)*dist/(2*self.k(self.b) + self._alfa2*dist)
        B = self.a - self.k(self.a)*dist/(2*self.k(self.a) + self._alfa1*dist)
        self._base_func.append(lambda x: (x - self.a)**2 * (x - A))
        self._base_func.append(lambda x: (self.b - x)**2 * (B - x))
        self._base_func_der.append(lambda x: 2*(x - self.a)*(x - A) + (x - self.a)**2)
        self._base_func_der.append(lambda x: -2*(self.b - x)*(B - x) - (self.b - x)**2)
        self._base_func_sec_der.append(lambda x: 6*x - 2*A - 4*self.a)
        self._base_func_sec_der.append(lambda x: 4*self.b + 2*B - 6*x)

    def set_solution(self, u):
        self._exact_solution = u

    # base functions
    def init_base_func(self, n):
        assert n > 2
        self.n = n
  

    # WORKS RIGHT
    def __substitute_nue(self):
        # psi = A_psi * x + B_psi
        # u(x) = v(x) + psi(x)
        if self._nue1 != 0 or self._nue2 != 0:
            self._B_psi = (self._nue1*self.k(self.b) + self._nue2*self.k(self.a)-   \
                    self._alfa1*self.a*self._nue2 + self._nue1*self._alfa2*self.b)/ \
                    (self.k(self.a)*self._alfa2 - self._alfa1*self._alfa2*self.a +  \
                    self._alfa1*self.k(self.b) + self._alfa1*self._alfa2*self.b)
            self._A_psi = (self._nue1 - self._alfa1*self._B_psi)/(-self.k(self.a) + self._alfa1*self.a)
            self.f = lambda x: self._f(x) - self.q(x)*(self._A_psi*x + self._B_psi) -     \
                    self._A_psi * self.p(x) - self.k1*self.k2*math.sin(self.k2*x)*self._A_psi
            self.__substitute = True
        else:
            self.f = self._f
            self.__substitute = False
   

    def __operator(self, phi, phi_der, phi_sec_der, x_j):
        delta_x = 0.0001
        phi_diff = phi_der(x_j)     #np.diff(phi(x_j))/np.diff(x_j)
        #phi_diff = (phi(x_j + delta_x) - phi(x_j - delta_x))/(2*delta_x)
        k_diff = self.k_der(x_j)
        sec_phi_diff = phi_sec_der(x_j)
        #sec_phi_diff = (phi(x_j + delta_x) - 2*phi(x_j) + phi(x_j - delta_x))/(delta_x*delta_x)
        return (-(k_diff * phi_diff + self.k(x_j) * sec_phi_diff) + self.p(x_j)*phi_diff + self.q(x_j)*phi(x_j))

    def plot_oper(self):
        plt.figure()
        x = np.linspace(self.a, self.b)
        val = [self.__operator(self._exact_solution, 1, 1, xi) for xi in x]
        func = [self._f(xi) for xi in x]
        plt.plot(x, val,"b", label = "Au" )
        plt.plot(x, func, 'g', label = "f(x)")
        plt.legend() 

    # collocation solve method
    def collocation_solve(self, cheb_dots):
        assert len(cheb_dots) == self.n
        C = np.array([[1.0 for j in range(self.n)] for i in range(self.n)])
        F = np.array([self.f(dot) for dot in cheb_dots])
        for j in range(self.n):
            C[j][0] = self.__operator(self._base_func[0], self._base_func_der[0], self._base_func_sec_der[0], cheb_dots[j])
            C[j][1] = self.__operator(self._base_func[1], self._base_func_der[1], self._base_func_sec_der[1], cheb_dots[j])
            for i in range(2, self.n):
                funci = lambda x: ((x - self.a)**2) *  math.pow((self.b - x), i)
                funcder = lambda x: 2*(x - self.a)*((self.b - x)**i) - i*((x - self.a)**2)*((self.b - x)**(i - 1))
                funcsecder = lambda x: (i - 1)*(i - 2)*(x - self.a)**(i - 3)*(x - self.b)**2 + \
                    (i - 1)*(x - self.a)**(i - 2)*2*(x - self.b) + \
                    2*(i - 1)*(x - self.a)**(i - 2)*(x - self.b) +  2*(x - self.a)**(i - 1)
                C[j][i] = self.__operator(funci,funcder, funcsecder, cheb_dots[j])
        self.solution_ci = np.linalg.solve(C, F)


    # least squares solve method
    def least_square_solve(self):
        C = np.array([[1.0 for j in range(self.n)] for i in range(self.n)])
        F = np.array([1.0 for j in range(self.n)])
        for h in range(4):
            i = h // 2
            j = h % 2
            f = lambda x: self.__operator(self._base_func[i], self._base_func_der[i],self._base_func_sec_der[i], x)*self.__operator(self._base_func[j], 
                            self._base_func_der[j],self._base_func_sec_der[j], x)
            C[i][j] = integral( f, self.a, self.b)
        F[0] = integral(lambda x: self.f(x)*self.__operator(self._base_func[0], self._base_func_der[0],self._base_func_sec_der[0], x) , 
                    self.a, self.b)
        F[1] = integral(lambda x: self.f(x)*self.__operator(self._base_func[1], self._base_func_der[1],self._base_func_sec_der[1], x) , 
                    self.a, self.b)

        for j in range(3, self.n + 1):
            funcj = lambda x: ((x - self.a)**2) *  math.pow((self.b - x), j)
            funcderj = lambda x: 2*(x - self.a)*((self.b - x)**j) - j*((x - self.a)**2)*((self.b - x)**(j - 1))
            funcsecderj = lambda x: (j - 1)*(j - 2)*(x - self.a)**(j - 3)*(x - self.b)**2 + \
                    (j - 1)*(x - self.a)**(j - 2)*2*(x - self.b) + \
                    2*(j - 1)*(x - self.a)**(j - 2)*(x - self.b) +  2*(x - self.a)**(j - 1)
            for i in range(3, self.n + 1):
                funci = lambda x: ((x - self.a)**2) *  math.pow((self.b - x), i)
                funcderi = lambda x: 2*(x - self.a)*((self.b - x)**i) - i*((x - self.a)**2)*((self.b - x)**(i - 1))
                funcsecderi = lambda x: (i - 1)*(i - 2)*(x - self.a)**(i - 3)*(x - self.b)**2 + \
                    (i - 1)*(x - self.a)**(i - 2)*2*(x - self.b) + \
                    2*(i - 1)*(x - self.a)**(i - 2)*(x - self.b) +  2*(x - self.a)**(i - 1)
                f = lambda x: self.__operator(funci, funcderi, funcsecderi, x)*self.__operator(funcj, funcderj, funcsecderj, x)
                C[j - 1][i - 1] = integral( f, self.a, self.b)
            F[j - 1] = integral(lambda x: self.f(x)*self.__operator(funcj, funcderj,funcsecderj, x), self.a, self.b)
        self.solution_ci = np.linalg.solve(C, F)

    def solution_difference(self):
        # (u, v) = int{a, b} u*v dx
        # ||u - v||^2 = (u - v, u - v) = int{a, b} (u(x) - v(x))^2 dx
        inner_sum = lambda x: (self._exact_solution(x) - self.solution(x))**2
        diff = integral(inner_sum, self.a, self.b)
        return math.sqrt(diff)

    def solution(self, x):
        if self.__substitute:
            suma = self._A_psi*x + self._B_psi
        else :
            suma = 0
        suma += self.solution_ci[0]*self._base_func[0](x) \
                                                + self.solution_ci[1]*self._base_func[1](x)
        for i in range(2, self.n):
            suma += ((x - self.a)**2) *  math.pow((self.b - x), i)
        return suma


    def plotBaseFunc(self):
        plt.figure()
        X = np.linspace(self.a, self.b)
        F = []
        for func in self._base_func:
            F = np.array([func(x) for x in X])
            plt.plot(X, F)
        
    def plotBaseFuncDer(self):
        plt.figure()
        X = np.linspace(self.a, self.b)
        F = []
        for func in self._base_func_der:
            F = np.array([func(x) for x in X])
            plt.plot(X, F)

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


N = 10   # func number
m_1 = 2; m_2 = 7; m_3 = 5
task = DifferentialTask(0, 1, k1=1, k2=3, k3=2, p1=2, p2=2, p3=3, q1=2, q2=3, q3=2, m1=m_1, m2=m_2, m3=m_3, alfa1=5, alfa2=5)
# u = m_1 * sin⁡(m_2 * x)+ m_3
task.set_solution(lambda x: m_1*math.sin(x * m_2) + m_3)
task.init_base_func(N)
x_cheb = np.linspace(0, 1, N)

dots = 100
val = np.linspace(0, 1, dots)
real_val = [task._exact_solution(x) for x in val]

task.collocation_solve(x_cheb)
approx_val_coll = [task.solution(x) for x in val]
diff1 = task.solution_difference()

task.least_square_solve()
approx_val_least = [task.solution(x) for x in val]
diff2 = task.solution_difference()

print(diff1, diff2)
#task.plot_oper()
plt.figure(figsize=(12, 7))
t = str(task)
#plt.text(0.005, 1.97, str(task))
plt.plot(val, real_val, 'r', label = "exact solution")
plt.plot(val, approx_val_coll, 'g', label="collocation method")
plt.plot(val, approx_val_least, 'y', label = "least square method")
plt.legend()
plt.show()
