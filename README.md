# MathematicalPhysicsCalculus
Some functions for solving tasks of mathematical physics

Builds an approximation of a solution to differential equation, 
using implemented methods. Contains example of how to use the class and results of its' calculations

The task looks like:


         -(k(x)u'(x))' + p(x)u'(x) + q(x)u(x) = f(x), x in [a, b]

where:   

         -k(a)u'(a) + alfa1*u(a) = nue1

         k(b)u'(b) + alfa2*u(b) = nue2      nue1, nue2 - real numbers
      
      
TaskSolve.py contains a class wich find an approximate solution of the task above with collocation method, least square method and integral approximation method.
