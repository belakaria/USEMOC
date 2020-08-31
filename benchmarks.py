import math
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy


################################################

def Currin(x, d):
    return float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))

def branin(x1,d):
    x=deepcopy(x1)
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)
################################################
def BlackBox_const1(x1,d):
    x=deepcopy(x1)
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)
def BlackBox_const2(x,d):
    return float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))

# whiteBox_constraints are constraints that does not require modeling. 
#In this example sum of inputs >=0
def whiteBox_const1(x,d):
    sum_=0
    for i in range(d):
        sum_=x[i]
    return -1*sum_
## Conbination constraint should be defined in the following way always
def combination_const1(functions,BlackBox_constraints,x):# the definition should not change
#    contraint of the for f0+f1-BB_c0-x[0]<=0
    contraint=functions[0]+functions[1]-BlackBox_constraints[0]+x[0]
    return contraint