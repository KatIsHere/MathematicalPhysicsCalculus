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
import scipy.integrate as integrate
import numpy as np
import math

# def gen_x(a, b, N= 250):
#   return [a + (b-a)/N * i for i in range(N+1)]

# def evalIntSimpson(func, a, b):
#   return (b-a)/6 * (func(a) + 4* func((a+b)/2) + func(b))

# def IntSimpson(func, arr):
#   return sum([evalIntSimpson(func, arr[k], arr[k+1]) for k in range(len(arr) - 1)])


# def integral(f, a, b):
#   # return quad(f, a, b)[0]
#   return IntSimpson(f, gen_x(a, b, 50*(b-a)))

# def deriv(f, x):
#   dx = 0.0001
#   return (f(x+dx) - f(x-dx))/(2*dx)

# def dderiv(f, x):
#   df = lambda x: deriv(f, x)
#   return deriv(df, x)

class DifferentialTask:
        
    def __init__(self, a, b, k1 = 1, k2 = 1, k3 = 1, p1 = 1, 
                      p2 = 1, p3 = 1, q1 = 1, q2 = 1, q3 = 1, 
                      m1 = 1, m2 = 1, m3 = 1, alfa1 = 1, alfa2 = 1):
        self.a = a
        self.b = b
        self.set_coefs(k1, k2, k3, p1, p2, p3, 
            q1, q2, q3, m1, m2, m3, alfa1, alfa2)
    
    def set_coefs(self, k1, k2, k3, p1, p2, p3, 
                    q1, q2, q3, m1, m2, m3, alfa1, alfa2):
        """Setting up parameters and functions"""
        self.k1 = k1; self.k2 = k2; self.k3 = k3
        self.p1 = p1; self.p2 = p2; self.p3 = p3
        self.q1 = q1; self.q2 = q2; self.q3 = q3
        self.m1 = m1; self.m2 = m2; self.m3 = m3
        self.k = lambda x: k1*math.cos(k2*x) + k3  
        self.p = lambda x: p1*math.sin(p2*x) + p3
        self.q = lambda x: q1*math.cos(q2*x) + q3
        self.k_der = lambda x: -k2*k1*math.sin(k2*x)    # k'(x) - derivative for operator calculations 
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
        self.__substitute_nue()
        
    def init_base_func_number(self, n):
        assert n >= 2    # there should be at least 2 base functions
        self.n = n
        dist = self.b - self.a
        self.__A = self.b + self.k(self.b)*dist/(2*self.k(self.b) + self._alfa2*dist)
        self.__B = self.a - self.k(self.a)*dist/(2*self.k(self.a) + self._alfa1*dist)

    def base_phi(self, x, i):
        """Base function phi_i(x)"""
        res = 0
        if i == 0:
            res = (x - self.a)**2 * (x - self.__A)
        elif i == 1:
            res = (self.b - x)**2 * (self.__B - x)
        else:
            res = (x - self.a)**2 *  math.pow((self.b - x), i)
        return res 

    def base_phi_der(self, x, i):
        """Base functions' phi_i(x) first derivative -- for accurate calculations"""
        res = 0
        if i == 0:
            res = 2*(x - self.a)*(x - self.__A) + (x - self.a)**2
        elif i == 1:
            res = -2*(self.b - x)*(self.__B - x) - (self.b - x)**2
        else:
            res = 2*(x - self.a)*((self.b - x)**i) - i*((x - self.a)**2)*((self.b - x)**(i - 1))
        return res 

    def base_phi_sec_der(self, x, i):
        """Base functions' phi_i(x) second derivative -- for accurate calculations"""
        res = 0
        if i == 0:
            res = 6*x - 2*self.__A - 4*self.a
        elif i == 1:
            res = 4*self.b + 2*self.__B - 6*x
        else:
            res = (i - 1)*(i - 2)*(x - self.a)**(i - 3)*(x - self.b)**2 + \
                    (i - 1)*(x - self.a)**(i - 2)*2*(x - self.b) + \
                    2*(i - 1)*(x - self.a)**(i - 2)*(x - self.b) +  2*(x - self.a)**(i - 1)
        return res 
            

    def set_solution(self, u):
        """Set up exact solution of the task - if one is available"""
        self._exact_solution = u


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
   

    def __operator(self, x_j, func_numb):
        """Operator from the task
            * INPUT:    x_j - value
                        func_numb - number of the base function"""
        delta_x = 0.0001
        phi_diff = self.base_phi_der(x_j, func_numb) 
        k_diff = self.k_der(x_j)
        sec_phi_diff = self.base_phi_sec_der(x_j, func_numb)
        return (-(k_diff * phi_diff + self.k(x_j) * sec_phi_diff) + self.p(x_j)*phi_diff + self.q(x_j)*self.base_phi(x_j))


    # def plot_oper(self):
    #     """Plot opa"""
    #     plt.figure()
    #     x = np.linspace(self.a, self.b)
    #     val = [self.__operator(self._exact_solution, 1, 1, xi) for xi in x]
    #     func = [self._f(xi) for xi in x]
    #     plt.plot(x, val,"b", label = "Au" )
    #     plt.plot(x, func, 'g', label = "f(x)")
    #     plt.legend() 

    # collocation solve method
    def collocation_solve(self, cheb_dots):
        """Find approximate solution of the task with collocation method
        * INPUT: cheb_dots - Chebishow from segment [a, b] in which 
                 the approximate solution should be the nearrest"""
        assert len(cheb_dots) == self.n
        C = np.array([[1.0 for j in range(self.n)] for i in range(self.n)])
        F = np.array([self.f(dot) for dot in cheb_dots])
        for j in range(self.n):
            C[j][0] = self.__operator(cheb_dots[j], 0)
            C[j][1] = self.__operator(cheb_dots[j], 1)
            for i in range(2, self.n):
                C[j][i] = self.__operator(cheb_dots[j], i)
        self.solution_ci = np.linalg.solve(C, F)


    # least squares solve method
    def least_square_solve(self):
        """Find approximate solution of the task with least square method"""
        C = np.array([[1.0 for j in range(self.n)] for i in range(self.n)])
        F = np.array([1.0 for j in range(self.n)])
        for j in range(self.n + 1):
            for i in range(self.n + 1):
                C[j - 1][i - 1] = integrate.quad(lambda x: self.__operator(x, i)*self.__operator(x, j), self.a, self.b)           #integral( f, self.a, self.b)
            F[j - 1] = integrade.quad(lambda x: self.f(x)*self.__operator(x, j), self.a, self.b)
        self.solution_ci = np.linalg.solve(C, F)


    def solution_difference(self):
        # (u, v) = int{a, b} u*v dx
        # ||u - v||^2 = (u - v, u - v) = int{a, b} (u(x) - v(x))^2 dx
        inner_sum = lambda x: (self._exact_solution(x) - self.solution(x))**2
        diff = integrate.quad(inner_sum, self.a, self.b)
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


    #def plotBaseFunc(self):
     #   plt.figure()
      #  X = np.linspace(self.a, self.b)
       # F = []
        #for func in self._base_func:
         #   F = np.array([func(x) for x in X])
          #  plt.plot(X, F)
        
    #def plotBaseFuncDer(self):
     #   plt.figure()
      #  X = np.linspace(self.a, self.b)
       # F = []
        #for func in self._base_func_der:
         #   F = np.array([func(x) for x in X])
          #  plt.plot(X, F)

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
task.init_base_func_number(N)
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
