# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
import os
import numpy as np
from model import GaussianProcess
import scipy
from platypus import NSGAII, Problem, Real
from acquisitions import UCB, LCB, TS, ei, pi,compute_beta

######################Algorithm input##############################
paths='.'

from benchmarks import branin,Currin,whiteBox_const1,combination_const1,BlackBox_const1,BlackBox_const2
functions=[branin,Currin]
#this is just a toy example of how to define constraints based on their type,
# this means we want branin and currin<=0 and all other constraints <=0
BlackBox_constraints=[BlackBox_const1,BlackBox_const2]   # if none leave brackets empty [] 
whiteBox_constraints=[whiteBox_const1]# if none leave brackets empty [] 
Combination_constraints=[combination_const1]# if none leave brackets empty [] 
d=2
seed=1
np.random.seed(seed)
total_iterations=100
intial_number=1
############################Set aquisation function 
acquisation=ei
batch_size=1 #In case you need batch version, you can set the batch size here 

######################################################################
def evaluation(xx,d):
    global functions,BlackBox_constraints,whiteBox_constraints,Combination_constraints
    x=[item for item in xx]
    y=[functions[i](x,d) for i in range(len(functions))]
    B_c=[BlackBox_constraints[i](x,d) for i in range(len(BlackBox_constraints))]
    W_c=[whiteBox_constraints[i](x,d) for i in range(len(whiteBox_constraints))]
    C_c=[Combination_constraints[i](y,B_c,x) for i in range(len(Combination_constraints)) ]
    all_=y+B_c+W_c+C_c
    print(all_)
    return all_
functions_and_contraints=evaluation
M=len(functions)
BB_C=len(BlackBox_constraints)
WB_C=len(whiteBox_constraints)
C_C=len(Combination_constraints)
total_C=BB_C+WB_C+C_C
bound=[0,1]
Fun_bounds=[bound]*d


###################GP Initialisation##########################

GPs=[]
GPs_C=[]

for i in range(M):
    GPs.append(GaussianProcess(d))
for i in range(BB_C):
    GPs_C.append(GaussianProcess(d))

for k in range(intial_number):
    exist=True
    while exist:
        x_rand=list(np.random.uniform(low=bound[0], high=bound[1], size=(d,)))
        if (any((x_rand == x).all() for x in GPs[0].xValues))==False:
            exist=False
    functions_contraints_values=functions_and_contraints(x_rand,d)
    for i in range(M):
        GPs[i].addSample(np.asarray(x_rand),functions_contraints_values[i])
    for i in range(BB_C):
        GPs_C[i].addSample(np.asarray(x_rand),functions_contraints_values[i+M])
    with  open(os.path.join(paths,'Inputs.txt'), "a") as filehandle:  
        for item in x_rand:
            filehandle.write('%f ' % item)
        filehandle.write('\n')
    filehandle.close()
    with  open(os.path.join(paths,'Outputs.txt'), "a") as filehandle:  
        for listitem in functions_contraints_values:
            filehandle.write('%f ' % listitem)
        filehandle.write('\n')
    filehandle.close()

for i in range(M):   
    GPs[i].fitModel()
for i in range(BB_C):   
    GPs_C[i].fitModel()


for l in range(total_iterations):

    beta=compute_beta(l+1,d)
    cheap_pareto_set=[]
    def CMO(x):
        global beta        
        x=np.asarray(x)
        BB_constraints_mean=[GPs_C[i].getPrediction(x)[0] for i in range(BB_C)]
        WB_constraints=[whiteBox_constraints[i](x,d) for i in range(WB_C)]
        C_constraints=[Combination_constraints[i]([GPs[i].getPrediction(x)[0] for i in range(M)],BB_constraints_mean,x) for i in range(C_C)]
        constraints=BB_constraints_mean+WB_constraints+C_constraints
        return [acquisation(x,beta,GPs[i])[0] for i in range(M)],constraints
        
    
    problem = Problem(d, M,total_C)
    problem.types[:] = Real(bound[0], bound[1])
    problem.constraints[:] = ["<=0" for i in range(total_C)]
    problem.function = CMO
    algorithm = NSGAII(problem)
    algorithm.run(2500)
    cheap_pareto_set=[solution.variables for solution in algorithm.result]
    cheap_pareto_set_unique=[]
    for i in range(len(cheap_pareto_set)):
        if (any((cheap_pareto_set[i] == x).all() for x in GPs[0].xValues))==False:
            cheap_pareto_set_unique.append(cheap_pareto_set[i])

    UBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]+beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(M)] for x in cheap_pareto_set_unique]
    LBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]-beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(M)] for x in cheap_pareto_set_unique]
    uncertaities= [scipy.spatial.Rectangle(UBs[i], LBs[i]).volume() for i in range(len(cheap_pareto_set_unique))]
    
    batch_indecies=np.argsort(uncertaities)[::-1][:batch_size]
    batch=[cheap_pareto_set_unique[i] for i in batch_indecies]

#---------------Updating and fitting the GPs-----------------   
    for x_best in batch:
        functions_contraints_values=functions_and_contraints(x_best,d)
        for i in range(M): 
            GPs[i].addSample(np.asarray(x_best),functions_contraints_values[i])
            GPs[i].fitModel()
        for i in range(BB_C): 
            GPs_C[i].addSample(x_best,functions_contraints_values[M+i])
            GPs_C[i].fitModel()
        with  open(os.path.join(paths,'Inputs.txt'), "a") as filehandle:  
            for item in x_best:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()
        with  open(os.path.join(paths,'Outputs.txt'), "a") as filehandle:  
            for listitem in functions_contraints_values:
                filehandle.write('%f ' % listitem)
            filehandle.write('\n')
        filehandle.close()

     

