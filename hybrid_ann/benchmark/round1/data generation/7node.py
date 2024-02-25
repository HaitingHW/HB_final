import numpy as np
import random
import scipy.integrate as scp
import numpy.random as rnd
import time
import matplotlib.pyplot as plt
import numpy.random as rnd
import copy
from matplotlib.pyplot import figure
from pyomo.environ import *
from pyomo.dae import *
import pandas as pd
import pickle
import scipy.stats
from math import comb
from numpy.polynomial import polynomial as P
from scipy.integrate import odeint

eps  = np.finfo(float).eps

''' Data treatment'''
def save_pkl(item, fname):
    sn = 'tmp7/' + fname
    with open(sn, 'wb') as handle:
        pickle.dump(item, handle) #, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'File saved at: {sn}')
    return None
# return None

def load_pkl(fname):
    with open(fname, 'rb') as handle:
        ans = pickle.load(handle)
    print(f'Loaded from: {fname}')
    return ans

def to_dict(x,dt):
    obs = list(x)
    # tp = list(time)
    dicx = {}
    for i in range(len(x)):
        dicx[dt*i] = obs[i]            # change thye value of 38.4
    return dicx

def get_grad(x, t):
    dxdt = [[],[],[],[]]
    for n in range(x.shape[0]):
        for i in range(len(x[0]) - 1):
            dxdt[n].append((x[n][i + 1] - x[n][i])/(t[i + 1] - t[i]))
        dxdt[n].append(dxdt[n][-1])
    return dxdt


import sys

eff_neuron = 7# actually applied number of neurons
no_euron = 15 # total number of neurons


x = load_pkl(sys.argv[1])

idx = x['idx']
file_idx = x['inputs']

xobs1 = load_pkl('data/xobs1.pkl')
xobs2 = load_pkl('data/xobs2.pkl')
xobs3 = load_pkl('data/xobs3.pkl')
xobs4 = load_pkl('data/xobs4.pkl')
xobs5 = load_pkl('data/xobs5.pkl')

# N_range = load_pkl('data/N_range.pkl')
# I0_range = load_pkl('data/I0_range.pkl')

operation_con = load_pkl('data/operation_con.pkl')

tt1   = load_pkl('data/tt1.pkl')
std_value1 = load_pkl('data/std_value1.pkl')
std_value2 = load_pkl('data/std_value2.pkl')
std_value3 = load_pkl('data/std_value3.pkl')
std_value4 = load_pkl('data/std_value4.pkl')
std_value5 = load_pkl('data/std_value5.pkl')


data_init = load_pkl('data/data_init.pkl')

tf     = 250
steps_ = 25
dt     = tf/steps_


#Nrmalization
#Nrmalization
std1 = (np.ones((4,26)).T*std_value1).T
std2 = (np.ones((4,26)).T*std_value2).T
std3 = (np.ones((4,26)).T*std_value3).T
std4 = (np.ones((4,26)).T*std_value4).T
std5 = (np.ones((4,26)).T*std_value5).T

# Maximum value of x and q for normalization for test problem
x_max = np.max(np.array([xobs1[0],xobs2[0],xobs3[0]]))
# x_max_t
q_max = np.max(np.array([xobs1[2],xobs2[2],xobs3[2]]))
# q_max_t


Xt1   = [to_dict(xobs1[0],dt), to_dict(xobs1[1],dt), to_dict(xobs1[2],dt), to_dict(xobs1[3],dt)] 
dXdt1 = [to_dict(get_grad(xobs1, tt1)[0],dt),to_dict(get_grad(xobs1, tt1)[1],dt),to_dict(get_grad(xobs1, tt1)[2],dt),to_dict(get_grad(xobs1, tt1)[3],dt)]
Xt2   = [to_dict(xobs2[0],dt), to_dict(xobs2[1],dt), to_dict(xobs2[2],dt), to_dict(xobs2[3],dt)] 
dXdt2 = [to_dict(get_grad(xobs2, tt1)[0],dt),to_dict(get_grad(xobs2, tt1)[1],dt),to_dict(get_grad(xobs2, tt1)[2],dt),to_dict(get_grad(xobs2, tt1)[3],dt)]
Xt3   = [to_dict(xobs3[0],dt), to_dict(xobs3[1],dt), to_dict(xobs3[2],dt), to_dict(xobs3[3],dt)] 
dXdt3 = [to_dict(get_grad(xobs3, tt1)[0],dt),to_dict(get_grad(xobs3, tt1)[1],dt),to_dict(get_grad(xobs3, tt1)[2],dt),to_dict(get_grad(xobs3, tt1)[3],dt)]
Xt4   = [to_dict(xobs4[0],dt), to_dict(xobs4[1],dt), to_dict(xobs4[2],dt), to_dict(xobs4[3],dt)] 
dXdt4 = [to_dict(get_grad(xobs4, tt1)[0],dt),to_dict(get_grad(xobs4, tt1)[1],dt),to_dict(get_grad(xobs4, tt1)[2],dt),to_dict(get_grad(xobs4, tt1)[3],dt)]
Xt5   = [to_dict(xobs5[0],dt), to_dict(xobs5[1],dt), to_dict(xobs5[2],dt), to_dict(xobs5[3],dt)] 
dXdt5 = [to_dict(get_grad(xobs5, tt1)[0],dt),to_dict(get_grad(xobs5, tt1)[1],dt),to_dict(get_grad(xobs5, tt1)[2],dt),to_dict(get_grad(xobs5, tt1)[3],dt)]


I1 = operation_con[0][1]
I2 = operation_con[1][1]
I3 = operation_con[2][1]
I4 = operation_con[3][1]
I5 = operation_con[4][1]

N1 = operation_con[0][0]
N2 = operation_con[1][0]
N3 = operation_con[2][0]
N4 = operation_con[3][0]
N5 = operation_con[4][0]

number_datapoints1 = xobs1.shape[1]
number_datapoints2 = xobs2.shape[1]
number_datapoints3 = xobs3.shape[1]


I_max = np.max([I1,I2,I3])

I_all = np.array([I1,I2,I3])



def serial_model_generation(termnumber):

    serialstructure = np.zeros((termnumber+1,termnumber))

    for i in range(serialstructure.shape[1]):

        for j in range (i+1):

            serialstructure[i][j] = 1

    serialstructure = np.roll(serialstructure, 1, axis=0).tolist()

    serialstructure

    int_serialstructure = []

    for i in serialstructure:

        int_serialstructure.append([int(k) for k in i])

    return int_serialstructure




serialstructures = serial_model_generation(no_euron)

sp = serialstructures[eff_neuron]

model         = AbstractModel()

# -- variable definition -- #

# defining time as continous variable
model.t       = ContinuousSet(bounds=[0, tt1[-1]])


# defining measurement times
model.tm      = Set(within=model.t)



# defining measured values as parameters
model.x1_noise = Param(model.tm)
model.n1_noise = Param(model.tm)
model.q1_noise = Param(model.tm)
model.f1_noise = Param(model.tm)

model.x2_noise = Param(model.tm)
model.n2_noise = Param(model.tm)
model.q2_noise = Param(model.tm)
model.f2_noise = Param(model.tm)

model.x3_noise = Param(model.tm)
model.n3_noise = Param(model.tm)
model.q3_noise = Param(model.tm)
model.f3_noise = Param(model.tm)


model.x4_noise = Param(model.tm)
model.n4_noise = Param(model.tm)
model.q4_noise = Param(model.tm)
model.f4_noise = Param(model.tm)

model.x5_noise = Param(model.tm)
model.n5_noise = Param(model.tm)
model.q5_noise = Param(model.tm)
model.f5_noise = Param(model.tm)
# defining state variables
model.x1 = Var(model.t, within=PositiveReals,initialize=Xt1[0]) 
model.n1 = Var(model.t, within=PositiveReals,initialize=Xt1[1])
model.q1 = Var(model.t, within=PositiveReals,initialize=Xt1[2])
model.f1 = Var(model.t, within=PositiveReals,initialize=Xt1[3])  

model.x2 = Var(model.t, within=PositiveReals,initialize=Xt2[0]) 
model.n2 = Var(model.t, within=PositiveReals,initialize=Xt2[1])
model.q2 = Var(model.t, within=PositiveReals,initialize=Xt2[2])
model.f2 = Var(model.t, within=PositiveReals,initialize=Xt2[3]) 

model.x3 = Var(model.t, within=PositiveReals,initialize=Xt3[0]) 
model.n3 = Var(model.t, within=PositiveReals,initialize=Xt3[1])
model.q3 = Var(model.t, within=PositiveReals,initialize=Xt3[2])
model.f3 = Var(model.t, within=PositiveReals,initialize=Xt3[3])

model.x4 = Var(model.t, within=PositiveReals,initialize=Xt4[0]) 
model.n4 = Var(model.t, within=PositiveReals,initialize=Xt4[1])
model.q4 = Var(model.t, within=PositiveReals,initialize=Xt4[2])
model.f4 = Var(model.t, within=PositiveReals,initialize=Xt4[3])

model.x5 = Var(model.t, within=PositiveReals,initialize=Xt5[0]) 
model.n5 = Var(model.t, within=PositiveReals,initialize=Xt5[1])
model.q5 = Var(model.t, within=PositiveReals,initialize=Xt5[2])
model.f5 = Var(model.t, within=PositiveReals,initialize=Xt5[3])

# fix initial value

# defining parameters to be determined
model.I = RangeSet(no_euron) # define number of neurons
model.J = RangeSet(3) # number of inputs
model.k = RangeSet(1)
init_dict_w1 = {(1,1):np.random.normal(0, 1)}
init_dict_b1 = {(1,1):np.random.normal(0, 1)}
init_dict_w2 = {(1,1):np.random.normal(0, 1)}
# init_dict_b2 = {(1,1):0}



for i in range(1,no_euron+1): # range of the number of neuron
    for j in range(1,4):# range of the number of inputs
        # print (i,j)
        init_dict_ij = {(i,j):np.random.normal(0, 0.1)}
        init_dict_w1.update(init_dict_ij)
model.w1 = Var(model.I,model.J, initialize=init_dict_w1,bounds = (-1,1))

for i in range(1,2): # for 1 layer this is fixed
    for j in range(1,no_euron+1): # range of the number of neuron
        # print (i,j)
        init_dict_ij = {(i,j):np.random.normal(0, 0.1)}
        init_dict_b1.update(init_dict_ij)
model.b1 = Var(model.k,model.I,initialize=init_dict_b1,bounds = (-1,1))

for i in range(1,2):# for 1 layer this is fixed
    for j in range(1,no_euron+1):# range of the number of neuron
        # print (i,j)
        init_dict_ij = {(i,j):np.random.normal(0, 0.1)}
        init_dict_w2.update(init_dict_ij)
model.w2           = Var(model.k,model.I,initialize=init_dict_w2,bounds = (-1,1))
model.b2           = Var(initialize=0,bounds = (-1,1))

model.node1 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))
model.node2 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))
model.node3 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))
model.node4 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))
model.node5 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))


model.ud               = Var(domain = Reals, bounds=(0,1),  initialize=0)
model.un               = Var(domain = Reals, bounds=(0,5),  initialize=2.8)
model.kn               = Var(domain = Reals, bounds=(0,5),  initialize=1.4)
model.theta            = Var(domain = Reals, bounds=(0,10),  initialize=7.5)
model.gamma            = Var(domain = Reals, bounds=(0,10),  initialize=8.4)
model.epsilon          = Var(domain = Reals, bounds=(0,1),  initialize=0.1)


std_x1        = std_value1[0]  
std_n1        = std_value1[1]
std_q1        = std_value1[2]    
std_f1        = std_value1[3]

std_x2        = std_value2[0]  
std_n2        = std_value2[1]
std_q2        = std_value2[2]    
std_f2        = std_value2[3]

std_x3        = std_value3[0]  
std_n3        = std_value3[1]
std_q3        = std_value3[2]    
std_f3        = std_value3[3]



# Define u
model.u1     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1))
model.u2     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1))
model.u3     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1))
model.u4     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1))
model.u5     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1))


# defining derivatives
model.x1dot = DerivativeVar(model.x1, wrt=model.t,domain = Reals,initialize = dXdt1[0])
model.n1dot = DerivativeVar(model.n1, wrt=model.t,domain = Reals,initialize = dXdt1[1])
model.q1dot = DerivativeVar(model.q1, wrt=model.t,domain = Reals,initialize = dXdt1[2])
model.f1dot = DerivativeVar(model.f1, wrt=model.t,domain = Reals,initialize = dXdt1[3])

model.x2dot = DerivativeVar(model.x2, wrt=model.t,domain = Reals,initialize = dXdt2[0])
model.n2dot = DerivativeVar(model.n2, wrt=model.t,domain = Reals,initialize = dXdt2[1])
model.q2dot = DerivativeVar(model.q2, wrt=model.t,domain = Reals,initialize = dXdt2[2])
model.f2dot = DerivativeVar(model.f2, wrt=model.t,domain = Reals,initialize = dXdt2[3])

model.x3dot = DerivativeVar(model.x3, wrt=model.t,domain = Reals,initialize = dXdt3[0])
model.n3dot = DerivativeVar(model.n3, wrt=model.t,domain = Reals,initialize = dXdt3[1])
model.q3dot = DerivativeVar(model.q3, wrt=model.t,domain = Reals,initialize = dXdt3[2])
model.f3dot = DerivativeVar(model.f3, wrt=model.t,domain = Reals,initialize = dXdt3[3])

model.x4dot = DerivativeVar(model.x4, wrt=model.t,domain = Reals,initialize = dXdt4[0])
model.n4dot = DerivativeVar(model.n4, wrt=model.t,domain = Reals,initialize = dXdt4[1])
model.q4dot = DerivativeVar(model.q4, wrt=model.t,domain = Reals,initialize = dXdt4[2])
model.f4dot = DerivativeVar(model.f4, wrt=model.t,domain = Reals,initialize = dXdt4[3])

model.x5dot = DerivativeVar(model.x5, wrt=model.t,domain = Reals,initialize = dXdt5[0])
model.n5dot = DerivativeVar(model.n5, wrt=model.t,domain = Reals,initialize = dXdt5[1])
model.q5dot = DerivativeVar(model.q5, wrt=model.t,domain = Reals,initialize = dXdt5[2])
model.f5dot = DerivativeVar(model.f5, wrt=model.t,domain = Reals,initialize = dXdt5[3])

# -- differential equations -- #

# differential equation for u, x, n#

# EXP1 ------------------------------------
def NN_node11_exp1(model,t):

    return model.node1[1,1,t] == tanh(model.x1[t]/x_max*model.w1[1,1]+model.q1[t]/q_max*model.w1[1,2]+ I1/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node11_exp1_cons = Constraint(model.t, rule = NN_node11_exp1)

def NN_node12_exp1(model,t):

    return model.node1[1,2,t] == tanh(model.x1[t]/x_max*model.w1[2,1]+model.q1[t]/q_max*model.w1[2,2]+ I1/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node12_exp1_cons = Constraint(model.t, rule = NN_node12_exp1)

def NN_node13_exp1(model,t):
    return model.node1[1,3,t] == tanh(model.x1[t]/x_max*model.w1[3,1]+model.q1[t]/q_max*model.w1[3,2]+ I1/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node13_exp1_cons = Constraint(model.t, rule = NN_node13_exp1)

def NN_node14_exp1(model,t):
    return model.node1[1,4,t] == tanh(model.x1[t]/x_max*model.w1[4,1]+model.q1[t]/q_max*model.w1[4,2]+ I1/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node14_exp1_cons = Constraint(model.t, rule = NN_node14_exp1)

def NN_node15_exp1(model,t):
    return model.node1[1,5,t] == tanh(model.x1[t]/x_max*model.w1[5,1]+model.q1[t]/q_max*model.w1[5,2]+ I1/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node15_exp1_cons = Constraint(model.t, rule = NN_node15_exp1)

def NN_node16_exp1(model,t):
    return model.node1[1,6,t] == tanh(model.x1[t]/x_max*model.w1[6,1]+model.q1[t]/q_max*model.w1[6,2]+ I1/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node16_exp1_cons = Constraint(model.t, rule = NN_node16_exp1)

def NN_node17_exp1(model,t):
    return model.node1[1,7,t] == tanh(model.x1[t]/x_max*model.w1[7,1]+model.q1[t]/q_max*model.w1[7,2]+ I1/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node17_exp1_cons = Constraint(model.t, rule = NN_node17_exp1)

def NN_node18_exp1(model,t):
    return model.node1[1,8,t] == tanh(model.x1[t]/x_max*model.w1[8,1]+model.q1[t]/q_max*model.w1[8,2]+ I1/I_max *model.w1[8,3] + model.b1[1,8])

model.NN_node18_exp1_cons = Constraint(model.t, rule = NN_node18_exp1)

def NN_node19_exp1(model,t):
    return model.node1[1,9,t] == tanh(model.x1[t]/x_max*model.w1[9,1]+model.q1[t]/q_max*model.w1[9,2]+ I1/I_max *model.w1[9,3] + model.b1[1,9])

model.NN_node19_exp1_cons = Constraint(model.t, rule = NN_node19_exp1)

def NN_node110_exp1(model,t):
    return model.node1[1,10,t] == tanh(model.x1[t]/x_max*model.w1[10,1]+model.q1[t]/q_max*model.w1[10,2]+ I1/I_max *model.w1[10,3] + model.b1[1,10])

model.NN_node110_exp1_cons = Constraint(model.t, rule = NN_node110_exp1)

def NN_node111_exp1(model,t):
    return model.node1[1,11,t] == tanh(model.x1[t]/x_max*model.w1[11,1]+model.q1[t]/q_max*model.w1[11,2]+ I1/I_max *model.w1[11,3] + model.b1[1,11])

model.NN_node111_exp1_cons = Constraint(model.t, rule = NN_node111_exp1)

def NN_node112_exp1(model,t):
    return model.node1[1,12,t] == tanh(model.x1[t]/x_max*model.w1[12,1]+model.q1[t]/q_max*model.w1[12,2]+ I1/I_max *model.w1[12,3] + model.b1[1,12])

model.NN_node112_exp1_cons = Constraint(model.t, rule = NN_node112_exp1)

def NN_node113_exp1(model,t):
    return model.node1[1,13,t] == tanh(model.x1[t]/x_max*model.w1[13,1]+model.q1[t]/q_max*model.w1[13,2]+ I1/I_max *model.w1[13,3] + model.b1[1,13])

model.NN_node113_exp1_cons = Constraint(model.t, rule = NN_node113_exp1)

def NN_node114_exp1(model,t):
    return model.node1[1,14,t] == tanh(model.x1[t]/x_max*model.w1[14,1]+model.q1[t]/q_max*model.w1[14,2]+ I1/I_max *model.w1[14,3] + model.b1[1,14])

model.NN_node114_exp1_cons = Constraint(model.t, rule = NN_node114_exp1)

def NN_node115_exp1(model,t):
    return model.node1[1,15,t] == tanh(model.x1[t]/x_max*model.w1[15,1]+model.q1[t]/q_max*model.w1[15,2]+ I1/I_max *model.w1[15,3] + model.b1[1,15])

model.NN_node115_exp1_cons = Constraint(model.t, rule = NN_node115_exp1)



def u1_con_exp1(model,t):
    return model.u1[t] ==  sp[0]*model.node1[1,1,t]*model.w2[1,1] + sp[1]*model.node1[1,2,t]*model.w2[1,2] + sp[2]*model.node1[1,3,t]*model.w2[1,3] + sp[3]*model.node1[1,4,t]*model.w2[1,4] + sp[4]*model.node1[1,5,t]*model.w2[1,5] + sp[5]*model.node1[1,6,t]*model.w2[1,6] + sp[6]*model.node1[1,7,t]*model.w2[1,7] + sp[7]*model.node1[1,8,t]*model.w2[1,8] + sp[8]*model.node1[1,9,t]*model.w2[1,9] + sp[9]*model.node1[1,10,t]*model.w2[1,10] + sp[10]*model.node1[1,11,t]*model.w2[1,11] + sp[11]*model.node1[1,12,t]*model.w2[1,12] + sp[12]*model.node1[1,13,t]*model.w2[1,13] + sp[13]*model.node1[1,14,t]*model.w2[1,14] + sp[14]*model.node1[1,15,t]*model.w2[1,15] + model.b2
model.u1_con_exp1  =Constraint(model.t,rule = u1_con_exp1)

def u1_e1_const(model, t):
  return model.u1[t] >= 0
model.u1_e1_const = Constraint(model.t, rule = u1_e1_const)


def dxdt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x1dot[t] == model.u1[t] * model.x1[t] - model.ud * model.x1[t]
model.dxdtcon1 = Constraint(model.t, rule = dxdt1)

def dndt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n1dot[t] == -model.un*(model.n1[t]/(model.n1[t]+model.kn))*model.x1[t]
model.dndtcon1 = Constraint(model.t, rule = dndt1)

def dqdt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q1dot[t] == model.un*(model.n1[t]/(model.n1[t]+model.kn)) - (model.u1[t]-model.ud)*model.q1[t]
model.dqdtcon = Constraint(model.t, rule = dqdt1)

def dfdt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f1dot[t] == model.u1[t]*(model.theta*model.q1[t]-model.epsilon*model.f1[t]) - model.gamma*model.un*(model.n1[t]/(model.n1[t]+model.kn)) + model.ud*model.epsilon*model.f1[t]
model.dfdtcon1 = Constraint(model.t, rule = dfdt1)


# EXP2 ------------------------------------
def NN_node21_exp2(model,t):

    return model.node2[1,1,t] == tanh(model.x2[t]/x_max*model.w1[1,1]+model.q2[t]/q_max*model.w1[1,2]+ I2/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node21_exp2_cons = Constraint(model.t, rule = NN_node21_exp2)

def NN_node22_exp2(model,t):

    return model.node2[1,2,t] == tanh(model.x2[t]/x_max*model.w1[2,1]+model.q2[t]/q_max*model.w1[2,2]+ I2/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node22_exp2_cons = Constraint(model.t, rule = NN_node22_exp2)

def NN_node23_exp2(model,t):
    return model.node2[1,3,t] == tanh(model.x2[t]/x_max*model.w1[3,1]+model.q2[t]/q_max*model.w1[3,2]+ I2/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node23_exp2_cons = Constraint(model.t, rule = NN_node23_exp2)

def NN_node24_exp2(model,t):
    return model.node2[1,4,t] == tanh(model.x2[t]/x_max*model.w1[4,1]+model.q2[t]/q_max*model.w1[4,2]+ I2/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node24_exp2_cons = Constraint(model.t, rule = NN_node24_exp2)

def NN_node25_exp2(model,t):
    return model.node2[1,5,t] == tanh(model.x2[t]/x_max*model.w1[5,1]+model.q2[t]/q_max*model.w1[5,2]+ I2/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node25_exp2_cons = Constraint(model.t, rule = NN_node25_exp2)

def NN_node26_exp2(model,t):
    return model.node2[1,6,t] == tanh(model.x2[t]/x_max*model.w1[6,1]+model.q2[t]/q_max*model.w1[6,2]+ I2/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node26_exp2_cons = Constraint(model.t, rule = NN_node26_exp2)

def NN_node27_exp2(model,t):
    return model.node2[1,7,t] == tanh(model.x2[t]/x_max*model.w1[7,1]+model.q2[t]/q_max*model.w1[7,2]+ I2/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node27_exp2_cons = Constraint(model.t, rule = NN_node27_exp2)

def NN_node28_exp2(model,t):
    return model.node2[1,8,t] == tanh(model.x2[t]/x_max*model.w1[8,1]+model.q2[t]/q_max*model.w1[8,2]+ I2/I_max *model.w1[8,3] + model.b1[1,8])

model.NN_node28_exp2_cons = Constraint(model.t, rule = NN_node28_exp2)

def NN_node29_exp2(model,t):
    return model.node2[1,9,t] == tanh(model.x2[t]/x_max*model.w1[9,1]+model.q2[t]/q_max*model.w1[9,2]+ I2/I_max *model.w1[9,3] + model.b1[1,9])

model.NN_node29_exp2_cons = Constraint(model.t, rule = NN_node29_exp2)

def NN_node210_exp2(model,t):
    return model.node2[1,10,t] == tanh(model.x2[t]/x_max*model.w1[10,1]+model.q2[t]/q_max*model.w1[10,2]+ I2/I_max *model.w1[10,3] + model.b1[1,10])

model.NN_node210_exp2_cons = Constraint(model.t, rule = NN_node210_exp2)

def NN_node211_exp2(model,t):
    return model.node2[1,11,t] == tanh(model.x2[t]/x_max*model.w1[11,1]+model.q2[t]/q_max*model.w1[11,2]+ I2/I_max *model.w1[11,3] + model.b1[1,11])

model.NN_node211_exp2_cons = Constraint(model.t, rule = NN_node211_exp2)

def NN_node212_exp2(model,t):
    return model.node2[1,12,t] == tanh(model.x2[t]/x_max*model.w1[12,1]+model.q2[t]/q_max*model.w1[12,2]+ I2/I_max *model.w1[12,3] + model.b1[1,12])

model.NN_node212_exp2_cons = Constraint(model.t, rule = NN_node212_exp2)

def NN_node213_exp2(model,t):
    return model.node2[1,13,t] == tanh(model.x2[t]/x_max*model.w1[13,1]+model.q2[t]/q_max*model.w1[13,2]+ I2/I_max *model.w1[13,3] + model.b1[1,13])

model.NN_node213_exp2_cons = Constraint(model.t, rule = NN_node213_exp2)

def NN_node214_exp2(model,t):
    return model.node2[1,14,t] == tanh(model.x2[t]/x_max*model.w1[14,1]+model.q2[t]/q_max*model.w1[14,2]+ I2/I_max *model.w1[14,3] + model.b1[1,14])

model.NN_node214_exp2_cons = Constraint(model.t, rule = NN_node214_exp2)

def NN_node215_exp2(model,t):
    return model.node2[1,15,t] == tanh(model.x2[t]/x_max*model.w1[15,1]+model.q2[t]/q_max*model.w1[15,2]+ I2/I_max *model.w1[15,3] + model.b1[1,15])

model.NN_node215_exp2_cons = Constraint(model.t, rule = NN_node215_exp2)


def u2_con_exp2(model,t):
    return model.u2[t] ==  sp[0]*model.node2[1,1,t]*model.w2[1,1] + sp[1]*model.node2[1,2,t]*model.w2[1,2] + sp[2]*model.node2[1,3,t]*model.w2[1,3] + sp[3]*model.node2[1,4,t]*model.w2[1,4] + sp[4]*model.node2[1,5,t]*model.w2[1,5] + sp[5]*model.node2[1,6,t]*model.w2[1,6] + sp[6]*model.node2[1,7,t]*model.w2[1,7] + sp[7]*model.node2[1,8,t]*model.w2[1,8] + sp[8]*model.node2[1,9,t]*model.w2[1,9] + sp[9]*model.node2[1,10,t]*model.w2[1,10] + sp[10]*model.node2[1,11,t]*model.w2[1,11] + sp[11]*model.node2[1,12,t]*model.w2[1,12] + sp[12]*model.node2[1,13,t]*model.w2[1,13] + sp[13]*model.node2[1,14,t]*model.w2[1,14] + sp[14]*model.node2[1,15,t]*model.w2[1,15] + model.b2
model.u2_con_exp2  =Constraint(model.t,rule = u2_con_exp2)

def u2_e2_const(model, t):
  return model.u2[t] >= 0
model.u2_e2_const = Constraint(model.t, rule = u2_e2_const)


def dxdt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x2dot[t] == model.u2[t] * model.x2[t] - model.ud * model.x2[t]
model.dxdtcon2 = Constraint(model.t, rule = dxdt2)

def dndt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n2dot[t] == -model.un*(model.n2[t]/(model.n2[t]+model.kn))*model.x2[t]
model.dndtcon2 = Constraint(model.t, rule = dndt2)

def dqdt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q2dot[t] == model.un*(model.n2[t]/(model.n2[t]+model.kn)) - (model.u2[t]-model.ud)*model.q2[t]
model.dqdtcon2 = Constraint(model.t, rule = dqdt2)

def dfdt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f2dot[t] == model.u2[t]*(model.theta*model.q2[t]-model.epsilon*model.f2[t]) - model.gamma*model.un*(model.n2[t]/(model.n2[t]+model.kn)) + model.ud*model.epsilon*model.f2[t]
model.dfdtcon2 = Constraint(model.t, rule = dfdt2)


# EXP3 ------------------------------------
def NN_node31_exp3(model,t):

    return model.node3[1,1,t] == tanh(model.x3[t]/x_max*model.w1[1,1]+model.q3[t]/q_max*model.w1[1,2]+ I3/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node31_exp3_cons = Constraint(model.t, rule = NN_node31_exp3)

def NN_node32_exp3(model,t):

    return model.node3[1,2,t] == tanh(model.x3[t]/x_max*model.w1[2,1]+model.q3[t]/q_max*model.w1[2,2]+ I3/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node32_exp3_cons = Constraint(model.t, rule = NN_node32_exp3)

def NN_node33_exp3(model,t):
    return model.node3[1,3,t] == tanh(model.x3[t]/x_max*model.w1[3,1]+model.q3[t]/q_max*model.w1[3,2]+ I3/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node33_exp3_cons = Constraint(model.t, rule = NN_node33_exp3)

def NN_node34_exp3(model,t):
    return model.node3[1,4,t] == tanh(model.x3[t]/x_max*model.w1[4,1]+model.q3[t]/q_max*model.w1[4,2]+ I3/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node34_exp3_cons = Constraint(model.t, rule = NN_node34_exp3)

def NN_node35_exp3(model,t):
    return model.node3[1,5,t] == tanh(model.x3[t]/x_max*model.w1[5,1]+model.q3[t]/q_max*model.w1[5,2]+ I3/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node35_exp3_cons = Constraint(model.t, rule = NN_node35_exp3)

def NN_node36_exp3(model,t):
    return model.node3[1,6,t] == tanh(model.x3[t]/x_max*model.w1[6,1]+model.q3[t]/q_max*model.w1[6,2]+ I3/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node36_exp3_cons = Constraint(model.t, rule = NN_node36_exp3)

def NN_node37_exp3(model,t):
    return model.node3[1,7,t] == tanh(model.x3[t]/x_max*model.w1[7,1]+model.q3[t]/q_max*model.w1[7,2]+ I3/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node37_exp3_cons = Constraint(model.t, rule = NN_node37_exp3)

def NN_node38_exp3(model,t):
    return model.node3[1,8,t] == tanh(model.x3[t]/x_max*model.w1[8,1]+model.q3[t]/q_max*model.w1[8,2]+ I3/I_max *model.w1[8,3] + model.b1[1,8])

model.NN_node38_exp3_cons = Constraint(model.t, rule = NN_node38_exp3)

def NN_node39_exp3(model,t):
    return model.node3[1,9,t] == tanh(model.x3[t]/x_max*model.w1[9,1]+model.q3[t]/q_max*model.w1[9,2]+ I3/I_max *model.w1[9,3] + model.b1[1,9])

model.NN_node39_exp3_cons = Constraint(model.t, rule = NN_node39_exp3)

def NN_node310_exp3(model,t):
    return model.node3[1,10,t] == tanh(model.x3[t]/x_max*model.w1[10,1]+model.q3[t]/q_max*model.w1[10,2]+ I3/I_max *model.w1[10,3] + model.b1[1,10])

model.NN_node310_exp3_cons = Constraint(model.t, rule = NN_node310_exp3)

def NN_node311_exp3(model,t):
    return model.node3[1,11,t] == tanh(model.x3[t]/x_max*model.w1[11,1]+model.q3[t]/q_max*model.w1[11,2]+ I3/I_max *model.w1[11,3] + model.b1[1,11])

model.NN_node311_exp3_cons = Constraint(model.t, rule = NN_node311_exp3)

def NN_node312_exp3(model,t):
    return model.node3[1,12,t] == tanh(model.x3[t]/x_max*model.w1[12,1]+model.q3[t]/q_max*model.w1[12,2]+ I3/I_max *model.w1[12,3] + model.b1[1,12])

model.NN_node312_exp3_cons = Constraint(model.t, rule = NN_node312_exp3)

def NN_node313_exp3(model,t):
    return model.node3[1,13,t] == tanh(model.x3[t]/x_max*model.w1[13,1]+model.q3[t]/q_max*model.w1[13,2]+ I3/I_max *model.w1[13,3] + model.b1[1,13])

model.NN_node313_exp3_cons = Constraint(model.t, rule = NN_node313_exp3)

def NN_node314_exp3(model,t):
    return model.node3[1,14,t] == tanh(model.x3[t]/x_max*model.w1[14,1]+model.q3[t]/q_max*model.w1[14,2]+ I3/I_max *model.w1[14,3] + model.b1[1,14])

model.NN_node314_exp3_cons = Constraint(model.t, rule = NN_node314_exp3)

def NN_node315_exp3(model,t):
    return model.node3[1,15,t] == tanh(model.x3[t]/x_max*model.w1[15,1]+model.q3[t]/q_max*model.w1[15,2]+ I3/I_max *model.w1[15,3] + model.b1[1,15])

model.NN_node315_exp3_cons = Constraint(model.t, rule = NN_node315_exp3)


def u3_con_exp3(model,t):
    return model.u3[t] ==  sp[0]*model.node3[1,1,t]*model.w2[1,1] + sp[1]*model.node3[1,2,t]*model.w2[1,2] + sp[2]*model.node3[1,3,t]*model.w2[1,3] + sp[3]*model.node3[1,4,t]*model.w2[1,4] + sp[4]*model.node3[1,5,t]*model.w2[1,5] + sp[5]*model.node3[1,6,t]*model.w2[1,6] + sp[6]*model.node3[1,7,t]*model.w2[1,7] + sp[7]*model.node3[1,8,t]*model.w2[1,8] + sp[8]*model.node3[1,9,t]*model.w2[1,9] + sp[9]*model.node3[1,10,t]*model.w2[1,10] + sp[10]*model.node3[1,11,t]*model.w2[1,11] + sp[11]*model.node3[1,12,t]*model.w2[1,12] + sp[12]*model.node3[1,13,t]*model.w2[1,13] + sp[13]*model.node3[1,14,t]*model.w2[1,14] + sp[14]*model.node3[1,15,t]*model.w2[1,15] + model.b2
model.u3_con_exp3  =Constraint(model.t,rule = u3_con_exp3)

def u3_e3_const(model, t):
  return model.u3[t] >= 0
model.u3_e3_const = Constraint(model.t, rule = u3_e3_const)


def dxdt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x3dot[t] == model.u3[t] * model.x3[t] - model.ud * model.x3[t]
model.dxdtcon3 = Constraint(model.t, rule = dxdt3)

def dndt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n3dot[t] == -model.un*(model.n3[t]/(model.n3[t]+model.kn))*model.x3[t]
model.dndtcon3 = Constraint(model.t, rule = dndt3)

def dqdt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q3dot[t] == model.un*(model.n3[t]/(model.n3[t]+model.kn)) - (model.u3[t]-model.ud)*model.q3[t]
model.dqdtcon3 = Constraint(model.t, rule = dqdt3)

def dfdt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f3dot[t] == model.u3[t]*(model.theta*model.q3[t]-model.epsilon*model.f3[t]) - model.gamma*model.un*(model.n3[t]/(model.n3[t]+model.kn)) + model.ud*model.epsilon*model.f3[t]
model.dfdtcon3 = Constraint(model.t, rule = dfdt3)

# EXP4 ------------------------------------
def NN_node41_exp4(model,t):

    return model.node4[1,1,t] == tanh(model.x4[t]/x_max*model.w1[1,1]+model.q4[t]/q_max*model.w1[1,2]+ I4/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node41_exp4_cons = Constraint(model.t, rule = NN_node41_exp4)

def NN_node42_exp4(model,t):

    return model.node4[1,2,t] == tanh(model.x4[t]/x_max*model.w1[2,1]+model.q4[t]/q_max*model.w1[2,2]+ I4/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node42_exp4_cons = Constraint(model.t, rule = NN_node42_exp4)

def NN_node43_exp4(model,t):
    return model.node4[1,3,t] == tanh(model.x4[t]/x_max*model.w1[3,1]+model.q4[t]/q_max*model.w1[3,2]+ I4/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node43_exp4_cons = Constraint(model.t, rule = NN_node43_exp4)

def NN_node44_exp4(model,t):
    return model.node4[1,4,t] == tanh(model.x4[t]/x_max*model.w1[4,1]+model.q4[t]/q_max*model.w1[4,2]+ I4/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node44_exp4_cons = Constraint(model.t, rule = NN_node44_exp4)

def NN_node45_exp4(model,t):
    return model.node4[1,5,t] == tanh(model.x4[t]/x_max*model.w1[5,1]+model.q4[t]/q_max*model.w1[5,2]+ I4/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node45_exp4_cons = Constraint(model.t, rule = NN_node45_exp4)

def NN_node46_exp4(model,t):
    return model.node4[1,6,t] == tanh(model.x4[t]/x_max*model.w1[6,1]+model.q4[t]/q_max*model.w1[6,2]+ I4/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node46_exp4_cons = Constraint(model.t, rule = NN_node46_exp4)

def NN_node47_exp4(model,t):
    return model.node4[1,7,t] == tanh(model.x4[t]/x_max*model.w1[7,1]+model.q4[t]/q_max*model.w1[7,2]+ I4/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node47_exp4_cons = Constraint(model.t, rule = NN_node47_exp4)

def NN_node48_exp4(model,t):
    return model.node4[1,8,t] == tanh(model.x4[t]/x_max*model.w1[8,1]+model.q4[t]/q_max*model.w1[8,2]+ I4/I_max *model.w1[8,3] + model.b1[1,8])

model.NN_node48_exp4_cons = Constraint(model.t, rule = NN_node48_exp4)

def NN_node49_exp4(model,t):
    return model.node4[1,9,t] == tanh(model.x4[t]/x_max*model.w1[9,1]+model.q4[t]/q_max*model.w1[9,2]+ I4/I_max *model.w1[9,3] + model.b1[1,9])

model.NN_node49_exp4_cons = Constraint(model.t, rule = NN_node49_exp4)

def NN_node410_exp4(model,t):
    return model.node4[1,10,t] == tanh(model.x4[t]/x_max*model.w1[10,1]+model.q4[t]/q_max*model.w1[10,2]+ I4/I_max *model.w1[10,3] + model.b1[1,10])

model.NN_node410_exp4_cons = Constraint(model.t, rule = NN_node410_exp4)

def NN_node411_exp4(model,t):
    return model.node4[1,11,t] == tanh(model.x4[t]/x_max*model.w1[11,1]+model.q4[t]/q_max*model.w1[11,2]+ I4/I_max *model.w1[11,3] + model.b1[1,11])

model.NN_node411_exp4_cons = Constraint(model.t, rule = NN_node411_exp4)

def NN_node412_exp4(model,t):
    return model.node4[1,12,t] == tanh(model.x4[t]/x_max*model.w1[12,1]+model.q4[t]/q_max*model.w1[12,2]+ I4/I_max *model.w1[12,3] + model.b1[1,12])

model.NN_node412_exp4_cons = Constraint(model.t, rule = NN_node412_exp4)

def NN_node413_exp4(model,t):
    return model.node4[1,13,t] == tanh(model.x4[t]/x_max*model.w1[13,1]+model.q4[t]/q_max*model.w1[13,2]+ I4/I_max *model.w1[13,3] + model.b1[1,13])

model.NN_node413_exp4_cons = Constraint(model.t, rule = NN_node413_exp4)

def NN_node414_exp4(model,t):
    return model.node4[1,14,t] == tanh(model.x4[t]/x_max*model.w1[14,1]+model.q4[t]/q_max*model.w1[14,2]+ I4/I_max *model.w1[14,3] + model.b1[1,14])

model.NN_node414_exp4_cons = Constraint(model.t, rule = NN_node414_exp4)

def NN_node415_exp4(model,t):
    return model.node4[1,15,t] == tanh(model.x4[t]/x_max*model.w1[15,1]+model.q4[t]/q_max*model.w1[15,2]+ I4/I_max *model.w1[15,3] + model.b1[1,15])

model.NN_node415_exp4_cons = Constraint(model.t, rule = NN_node415_exp4)
def u4_con_exp4(model,t):
    return model.u4[t] ==  sp[0]*model.node4[1,1,t]*model.w2[1,1] + sp[1]*model.node4[1,2,t]*model.w2[1,2] + sp[2]*model.node4[1,3,t]*model.w2[1,3] + sp[3]*model.node4[1,4,t]*model.w2[1,4] + sp[4]*model.node4[1,5,t]*model.w2[1,5] + sp[5]*model.node4[1,6,t]*model.w2[1,6] + sp[6]*model.node4[1,7,t]*model.w2[1,7] + sp[7]*model.node4[1,8,t]*model.w2[1,8] + sp[8]*model.node4[1,9,t]*model.w2[1,9] + sp[9]*model.node4[1,10,t]*model.w2[1,10] + sp[10]*model.node4[1,11,t]*model.w2[1,11] + sp[11]*model.node4[1,12,t]*model.w2[1,12] + sp[12]*model.node4[1,13,t]*model.w2[1,13] + sp[13]*model.node4[1,14,t]*model.w2[1,14] + sp[14]*model.node4[1,15,t]*model.w2[1,15] + model.b2
model.u4_con_exp4  =Constraint(model.t,rule = u4_con_exp4)

def u4_e4_const(model, t):
  return model.u4[t] >= 0
model.u4_e4_const = Constraint(model.t, rule = u4_e4_const)


def dxdt4(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x4dot[t] == model.u4[t] * model.x4[t] - model.ud * model.x4[t]
model.dxdtcon4 = Constraint(model.t, rule = dxdt4)

def dndt4(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n4dot[t] == -model.un*(model.n4[t]/(model.n4[t]+model.kn))*model.x4[t]
model.dndtcon4 = Constraint(model.t, rule = dndt4)

def dqdt4(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q4dot[t] == model.un*(model.n4[t]/(model.n4[t]+model.kn)) - (model.u4[t]-model.ud)*model.q4[t]
model.dqdtcon4 = Constraint(model.t, rule = dqdt4)

def dfdt4(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f4dot[t] == model.u4[t]*(model.theta*model.q4[t]-model.epsilon*model.f4[t]) - model.gamma*model.un*(model.n4[t]/(model.n4[t]+model.kn)) + model.ud*model.epsilon*model.f4[t]
model.dfdtcon4 = Constraint(model.t, rule = dfdt4)

# # EXP5 ------------------------------------

def NN_node51_exp5(model,t):

    return model.node5[1,1,t] == tanh(model.x5[t]/x_max*model.w1[1,1]+model.q5[t]/q_max*model.w1[1,2]+ I5/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node51_exp5_cons = Constraint(model.t, rule = NN_node51_exp5)

def NN_node52_exp5(model,t):

    return model.node5[1,2,t] == tanh(model.x5[t]/x_max*model.w1[2,1]+model.q5[t]/q_max*model.w1[2,2]+ I5/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node52_exp5_cons = Constraint(model.t, rule = NN_node52_exp5)

def NN_node53_exp5(model,t):
    return model.node5[1,3,t] == tanh(model.x5[t]/x_max*model.w1[3,1]+model.q5[t]/q_max*model.w1[3,2]+ I5/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node53_exp5_cons = Constraint(model.t, rule = NN_node53_exp5)

def NN_node54_exp5(model,t):
    return model.node5[1,4,t] == tanh(model.x5[t]/x_max*model.w1[4,1]+model.q5[t]/q_max*model.w1[4,2]+ I5/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node54_exp5_cons = Constraint(model.t, rule = NN_node54_exp5)

def NN_node55_exp5(model,t):
    return model.node5[1,5,t] == tanh(model.x5[t]/x_max*model.w1[5,1]+model.q5[t]/q_max*model.w1[5,2]+ I5/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node55_exp5_cons = Constraint(model.t, rule = NN_node55_exp5)

def NN_node56_exp5(model,t):
    return model.node5[1,6,t] == tanh(model.x5[t]/x_max*model.w1[6,1]+model.q5[t]/q_max*model.w1[6,2]+ I5/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node56_exp5_cons = Constraint(model.t, rule = NN_node56_exp5)

def NN_node57_exp5(model,t):
    return model.node5[1,7,t] == tanh(model.x5[t]/x_max*model.w1[7,1]+model.q5[t]/q_max*model.w1[7,2]+ I5/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node57_exp5_cons = Constraint(model.t, rule = NN_node57_exp5)

def NN_node58_exp5(model,t):
    return model.node5[1,8,t] == tanh(model.x5[t]/x_max*model.w1[8,1]+model.q5[t]/q_max*model.w1[8,2]+ I5/I_max *model.w1[8,3] + model.b1[1,8])

model.NN_node58_exp5_cons = Constraint(model.t, rule = NN_node58_exp5)

def NN_node59_exp5(model,t):
    return model.node5[1,9,t] == tanh(model.x5[t]/x_max*model.w1[9,1]+model.q5[t]/q_max*model.w1[9,2]+ I5/I_max *model.w1[9,3] + model.b1[1,9])

model.NN_node59_exp5_cons = Constraint(model.t, rule = NN_node59_exp5)

def NN_node510_exp5(model,t):
    return model.node5[1,10,t] == tanh(model.x5[t]/x_max*model.w1[10,1]+model.q5[t]/q_max*model.w1[10,2]+ I5/I_max *model.w1[10,3] + model.b1[1,10])

model.NN_node510_exp5_cons = Constraint(model.t, rule = NN_node510_exp5)

def NN_node511_exp5(model,t):
    return model.node5[1,11,t] == tanh(model.x5[t]/x_max*model.w1[11,1]+model.q5[t]/q_max*model.w1[11,2]+ I5/I_max *model.w1[11,3] + model.b1[1,11])

model.NN_node511_exp5_cons = Constraint(model.t, rule = NN_node511_exp5)

def NN_node512_exp5(model,t):
    return model.node5[1,12,t] == tanh(model.x5[t]/x_max*model.w1[12,1]+model.q5[t]/q_max*model.w1[12,2]+ I5/I_max *model.w1[12,3] + model.b1[1,12])

model.NN_node512_exp5_cons = Constraint(model.t, rule = NN_node512_exp5)

def NN_node513_exp5(model,t):
    return model.node5[1,13,t] == tanh(model.x5[t]/x_max*model.w1[13,1]+model.q5[t]/q_max*model.w1[13,2]+ I5/I_max *model.w1[13,3] + model.b1[1,13])

model.NN_node513_exp5_cons = Constraint(model.t, rule = NN_node513_exp5)

def NN_node514_exp5(model,t):
    return model.node5[1,14,t] == tanh(model.x5[t]/x_max*model.w1[14,1]+model.q5[t]/q_max*model.w1[14,2]+ I5/I_max *model.w1[14,3] + model.b1[1,14])

model.NN_node514_exp5_cons = Constraint(model.t, rule = NN_node514_exp5)

def NN_node515_exp5(model,t):
    return model.node5[1,15,t] == tanh(model.x5[t]/x_max*model.w1[15,1]+model.q5[t]/q_max*model.w1[15,2]+ I5/I_max *model.w1[15,3] + model.b1[1,15])

model.NN_node515_exp5_cons = Constraint(model.t, rule = NN_node515_exp5)


def u5_con_exp5(model,t):
    return model.u5[t] ==  sp[0]*model.node5[1,1,t]*model.w2[1,1] + sp[1]*model.node5[1,2,t]*model.w2[1,2] + sp[2]*model.node5[1,3,t]*model.w2[1,3] + sp[3]*model.node5[1,4,t]*model.w2[1,4] + sp[4]*model.node5[1,5,t]*model.w2[1,5] + sp[5]*model.node5[1,6,t]*model.w2[1,6] + sp[6]*model.node5[1,7,t]*model.w2[1,7] + sp[7]*model.node5[1,8,t]*model.w2[1,8] + sp[8]*model.node5[1,9,t]*model.w2[1,9] + sp[9]*model.node5[1,10,t]*model.w2[1,10] + sp[10]*model.node5[1,11,t]*model.w2[1,11] + sp[11]*model.node5[1,12,t]*model.w2[1,12] + sp[12]*model.node5[1,13,t]*model.w2[1,13] + sp[13]*model.node5[1,14,t]*model.w2[1,14] + sp[14]*model.node5[1,15,t]*model.w2[1,15] + model.b2
model.u5_con_exp5  =Constraint(model.t,rule = u5_con_exp5)

def u5_e5_const(model, t):
  return model.u5[t] >= 0
model.u5_e5_const = Constraint(model.t, rule = u5_e5_const)


def dxdt5(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x5dot[t] == model.u5[t] * model.x5[t] - model.ud * model.x5[t]
model.dxdtcon5 = Constraint(model.t, rule = dxdt5)

def dndt5(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n5dot[t] == -model.un*(model.n5[t]/(model.n5[t]+model.kn))*model.x5[t]
model.dndtcon5 = Constraint(model.t, rule = dndt5)

def dqdt5(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q5dot[t] == model.un*(model.n5[t]/(model.n5[t]+model.kn)) - (model.u5[t]-model.ud)*model.q5[t]
model.dqdtcon5 = Constraint(model.t, rule = dqdt5)

def dfdt5(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f5dot[t] == model.u5[t]*(model.theta*model.q5[t]-model.epsilon*model.f5[t]) - model.gamma*model.un*(model.n5[t]/(model.n5[t]+model.kn)) + model.ud*model.epsilon*model.f5[t]
model.dfdtcon5 = Constraint(model.t, rule = dfdt5)




def dx1dt_const(model,t):
  return model.x1dot[t] >= 0
model.dx1dt_const = Constraint(model.t, rule = dx1dt_const)
def dx2dt_const(model,t):
  return model.x2dot[t] >= 0
model.dx2dt_const = Constraint(model.t, rule = dx2dt_const)
def dx3dt_const(model,t):
  return model.x3dot[t] >= 0
model.dx3dt_const = Constraint(model.t, rule = dx3dt_const)
def dx4dt_const(model,t):
  return model.x4dot[t] >= 0
model.dx4dt_const = Constraint(model.t, rule = dx4dt_const)
def dx5dt_const(model,t):
  return model.x5dot[t] >= 0
model.dx5dt_const = Constraint(model.t, rule = dx5dt_const)



def max_x4_con(model,t):
    return model.x4[t]<= np.max(xobs4[0])
model.max_x4_con =  Constraint(model.t,rule = max_x4_con)

def max_n4_con(model,t):
    return model.n4[t]<= np.max(xobs4[1])
model.max_n4_con =  Constraint(model.t,rule = max_n4_con)

def max_q4_con(model,t):
    return model.q4[t]<= np.max(xobs4[2])
model.max_q4_con =  Constraint(model.t,rule = max_q4_con)

def max_f4_con(model,t):
    return model.f4[t]<= np.max(xobs4[3])
model.max_f4_con =  Constraint(model.t,rule = max_f4_con)


# bound maximum value of each state variable for exp 5
def max_x5_con(model,t):
    return model.x5[t]<= np.max(xobs5[0])
model.max_x5_con =  Constraint(model.t,rule = max_x5_con)

def max_n5_con(model,t):
    return model.n5[t]<= np.max(xobs5[1])
model.max_n5_con =  Constraint(model.t,rule = max_n5_con)

def max_q5_con(model,t):
    return model.q5[t]<= np.max(xobs5[2])
model.max_q5_con =  Constraint(model.t,rule = max_q5_con)

def max_f5_con(model,t):
    return model.f5[t]<= np.max(xobs5[3])
model.max_f5_con =  Constraint(model.t,rule = max_f5_con)



number_spc1 = xobs1.shape[0]
number_spc2 = xobs2.shape[0]
number_spc3 = xobs3.shape[0]

 

def obj(model):

 

    variance1    = (sum((model.x1[t]-model.x1_noise[t])**2 for t in model.tm)+sum((model.n1[t]-model.n1_noise[t])**2 for t in model.tm)+sum((model.q1[t]-model.q1_noise[t])**2 for t in model.tm)+sum((model.f1[t]-model.f1_noise[t])**2 for t in model.tm))/(number_datapoints1 * number_spc1)

 

    variance2    = (sum((model.x2[t]-model.x2_noise[t])**2 for t in model.tm)+sum((model.n2[t]-model.n2_noise[t])**2 for t in model.tm)+sum((model.q2[t]-model.q2_noise[t])**2 for t in model.tm)+sum((model.f2[t]-model.f2_noise[t])**2 for t in model.tm))/(number_datapoints2 * number_spc2)





    variance3    = (sum((model.x3[t]-model.x3_noise[t])**2 for t in model.tm)+sum((model.n3[t]-model.n3_noise[t])**2 for t in model.tm)+sum((model.q3[t]-model.q3_noise[t])**2 for t in model.tm)+sum((model.f3[t]-model.f3_noise[t])**2 for t in model.tm))/(number_datapoints3 * number_spc3)

    # Variance already has the square

 

    obj1 = (sum((model.x1[t]-model.x1_noise[t])**2 for t in model.tm)+sum((model.n1[t]-model.n1_noise[t])**2 for t in model.tm)+sum((model.q1[t]-model.q1_noise[t])**2 for t in model.tm)+sum((model.f1[t]-model.f1_noise[t])**2 for t in model.tm))/2/variance1 - (number_datapoints1 * number_spc1)*log(1/(sqrt(2*3.14159*variance1)))

 

    obj2 = (sum((model.x2[t]-model.x2_noise[t])**2 for t in model.tm)+sum((model.n2[t]-model.n2_noise[t])**2 for t in model.tm)+sum((model.q2[t]-model.q2_noise[t])**2 for t in model.tm)+sum((model.f2[t]-model.f2_noise[t])**2 for t in model.tm))/2/variance2 - (number_datapoints2 * number_spc2)*log(1/(sqrt(2*3.14159*variance2)))

 

    obj3 = (sum((model.x3[t]-model.x3_noise[t])**2 for t in model.tm)+sum((model.n3[t]-model.n3_noise[t])**2 for t in model.tm)+sum((model.q3[t]-model.q3_noise[t])**2 for t in model.tm)+sum((model.f3[t]-model.f3_noise[t])**2 for t in model.tm))/2/variance3 - (number_datapoints3 * number_spc3)*log(1/(sqrt(2*3.14159*variance3)))

    return obj1+obj2+obj3

    # return variance1+variance2+variance3

model.obj = Objective(rule=obj)


# def obj(model):
#     return 1/2*(sum(((model.x1[t]-model.x1_noise[t])**2/std_x1**2) +((model.n1[t]-model.n1_noise[t])**2/std_n1**2) for t in model.tm)+sum(((model.x2[t]-model.x2_noise[t])**2/std_x2**2) +((model.n2[t]-model.n2_noise[t])**2/std_n2**2) for t in model.tm)+ sum(((model.x3[t]-model.x3_noise[t])**2/std_x3**2) +((model.n3[t]-model.n3_noise[t])**2/std_n3**2) for t in model.tm))
# model.obj = Objective(rule=obj)


# -- model display -- #

# -- creating optimization problem -- #
instance = model.create_instance(data_init)
instance.x1[0].fix(0.18)
instance.n1[0].fix(N1)
instance.q1[0].fix(80)
instance.f1[0].fix(120)

instance.x2[0].fix(0.18)
instance.n2[0].fix(N2)
instance.q2[0].fix(80)
instance.f2[0].fix(120)

instance.x3[0].fix(0.18)
instance.n3[0].fix(N3)
instance.q3[0].fix(80)
instance.f3[0].fix(120)

instance.x4[0].fix(0.18)
instance.n4[0].fix(N4)
instance.q4[0].fix(80)
instance.f4[0].fix(120)

instance.x5[0].fix(0.18)
instance.n5[0].fix(N5)
instance.q5[0].fix(80)
instance.f5[0].fix(120)

discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(instance,nfe=60,ncp=3,wrt=instance.t,scheme='LAGRANGE-RADAU')


    # fix initial value

solver=SolverFactory('ipopt')
solver.options['max_iter'] = 100000
solver.options['tol'] = 1e-9

results = solver.solve(instance, tee=True)


structure_infor = [1,eff_neuron]

w1_index = [str(i)+','+str(j) for i in range(1,structure_infor[1]+1) for j in range(1,4)]
w1_data = [instance.w1[i,j]() for i in range(1,structure_infor[1]+1) for j in range(1,4)]
w2_index = [str(i)+','+str(j) for i in range(1,2) for j in range(1,structure_infor[1]+1)]
w2_data = [instance.w2[i,j]() for i in range(1,2) for j in range(1,structure_infor[1]+1)]
df_w2 = pd.DataFrame(w2_data).T
df_w2.columns = w2_index

b1_index = [str(i)+','+str(j) for i in range(1,2) for j in range(1,structure_infor[1]+1)]
b1_data = [instance.b1[i,j]() for i in range(1,2) for j in range(1,structure_infor[1]+1)]

df_b1 = pd.DataFrame(b1_data).T
df_b1.columns = b1_index

NN_pe = w1_data+b1_data+w2_data

NN_pe.append(instance.b2())
NN_pe.append(instance.ud())
NN_pe.append(instance.un())
NN_pe.append(instance.kn())
NN_pe.append(instance.theta())
NN_pe.append(instance.gamma())
NN_pe.append(instance.epsilon())

save_pkl(NN_pe,f'peoutput.{idx}')

save_pkl(instance.obj(),f'objoutput.{idx}')