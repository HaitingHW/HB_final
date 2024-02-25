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
from pyomo.environ import ConstraintList

eps  = np.finfo(float).eps


''' Data treatment'''
def save_pkl(item, fname):
    sn = 'tmp11/' + fname
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



x = load_pkl(sys.argv[1])

idx = x['idx']
file_idx = x['inputs']

xobs1 = load_pkl('data11/xobs1.pkl')
xobs2 = load_pkl('data11/xobs2.pkl')
xobs3 = load_pkl('data11/xobs3.pkl')
xobs4 = load_pkl('data11/xobs4.pkl')
mu_con   = load_pkl('data11/u_con.pkl')
tf_N  = load_pkl('data11/tf_N.pkl')
N_in  = load_pkl('data11/N_in.pkl')
operation_con = load_pkl('data11/operation_con.pkl')


tt1 = load_pkl('data11/tt1.pkl')
std_value1 = load_pkl('data11/std_value1.pkl')
std_value2 = load_pkl('data11/std_value2.pkl')
std_value3 = load_pkl('data11/std_value3.pkl')
std_value4 = load_pkl('data11/std_value4.pkl')
data_init = load_pkl('data11/data_init.pkl')

operation_con1 = operation_con[0]
operation_con2 = operation_con[1]
operation_con3 = operation_con[2]
operation_con4 = operation_con[3]

mu1 = mu_con[0] #u contains FNin and I0
mu2 = mu_con[1] 
mu3 = mu_con[2] 
mu4 = mu_con[3] 

tf    = 240
# tf    = 16*24
steps_= 10
dt    = tf/steps_ #dt is fixed


x_max = np.max(np.array([xobs1[0],xobs2[0],xobs3[0],xobs4[0]]))
q_max = np.max(np.array([xobs1[2],xobs2[2],xobs3[2],xobs4[2]]))
I_max = np.max(np.concatenate(mu_con).T[1])


num_N = 4
tf_N  = tf/(num_N)
dstep_N = int(tf_N/dt)

def get_grad(x, t):
    dxdt = [[],[],[],[],[]]
    for n in range(x.shape[0]):
        for i in range(len(x[0]) - 1):
            dxdt[n].append((x[n][i + 1] - x[n][i])/(t[i + 1] - t[i]))
        dxdt[n].append(dxdt[n][-1])
    return dxdt

Xt1   = [to_dict(xobs1[0],dt), to_dict(xobs1[1],dt), to_dict(xobs1[2],dt), to_dict(xobs1[3],dt), to_dict(xobs1[4],dt)] 
dXdt1 = [to_dict(get_grad(xobs1, tt1)[0],dt),to_dict(get_grad(xobs1, tt1)[1],dt),to_dict(get_grad(xobs1, tt1)[2],dt),to_dict(get_grad(xobs1, tt1)[3],dt),to_dict(get_grad(xobs1, tt1)[4],dt)]
Xt2   = [to_dict(xobs2[0],dt), to_dict(xobs2[1],dt), to_dict(xobs2[2],dt), to_dict(xobs2[3],dt), to_dict(xobs2[4],dt)] 
dXdt2 = [to_dict(get_grad(xobs2, tt1)[0],dt),to_dict(get_grad(xobs2, tt1)[1],dt),to_dict(get_grad(xobs2, tt1)[2],dt),to_dict(get_grad(xobs2, tt1)[3],dt),to_dict(get_grad(xobs2, tt1)[4],dt)]
Xt3   = [to_dict(xobs3[0],dt), to_dict(xobs3[1],dt), to_dict(xobs3[2],dt), to_dict(xobs3[3],dt), to_dict(xobs3[4],dt)] 
dXdt3 = [to_dict(get_grad(xobs3, tt1)[0],dt),to_dict(get_grad(xobs3, tt1)[1],dt),to_dict(get_grad(xobs3, tt1)[2],dt),to_dict(get_grad(xobs3, tt1)[3],dt),to_dict(get_grad(xobs3, tt1)[4],dt)]
Xt4   = [to_dict(xobs4[0],dt), to_dict(xobs4[1],dt), to_dict(xobs4[2],dt), to_dict(xobs4[3],dt), to_dict(xobs4[4],dt)] 
dXdt4 = [to_dict(get_grad(xobs4, tt1)[0],dt),to_dict(get_grad(xobs4, tt1)[1],dt),to_dict(get_grad(xobs4, tt1)[2],dt),to_dict(get_grad(xobs4, tt1)[3],dt),to_dict(get_grad(xobs4, tt1)[4],dt)]
# Xt5   = [to_dict(xobs5[0],dt), to_dict(xobs5[1],dt), to_dict(xobs5[2],dt), to_dict(xobs5[3],dt)] 
# dXdt5 = [to_dict(get_grad(xobs5, tt1)[0],dt),to_dict(get_grad(xobs5, tt1)[1],dt),to_dict(get_grad(xobs5, tt1)[2],dt),to_dict(get_grad(xobs5, tt1)[3],dt)]

N1 = operation_con[0][0]
N2 = operation_con[1][0]
N3 = operation_con[2][0]
N4 = operation_con[3][0]

# eff_neuron = 7# actually applied number of neurons
no_euron = 7 # total number of neurons


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


# defining state variables
model.x1 = Var(model.t, within=PositiveReals,initialize=Xt1[0]) 
model.n1 = Var(model.t, within=PositiveReals,initialize=Xt1[1])
model.q1 = Var(model.t, within=PositiveReals,initialize=Xt1[2])
model.f1 = Var(model.t, within=PositiveReals,initialize=Xt1[3])
model.V1 = Var(model.t, within=PositiveReals,initialize=Xt1[4])  



model.x2 = Var(model.t, within=PositiveReals,initialize=Xt2[0]) 
model.n2 = Var(model.t, within=PositiveReals,initialize=Xt2[1])
model.q2 = Var(model.t, within=PositiveReals,initialize=Xt2[2])
model.f2 = Var(model.t, within=PositiveReals,initialize=Xt2[3])
model.V2 = Var(model.t, within=PositiveReals,initialize=Xt2[4])


model.x3 = Var(model.t, within=PositiveReals,initialize=Xt3[0]) 
model.n3 = Var(model.t, within=PositiveReals,initialize=Xt3[1])
model.q3 = Var(model.t, within=PositiveReals,initialize=Xt3[2])
model.f3 = Var(model.t, within=PositiveReals,initialize=Xt3[3])
model.V3 = Var(model.t, within=PositiveReals,initialize=Xt3[4])

# model.x4 = Var(model.t, within=PositiveReals,initialize=Xt4[0]) 
# model.n4 = Var(model.t, within=PositiveReals,initialize=Xt4[1])
# model.q4 = Var(model.t, within=PositiveReals,initialize=Xt4[2])
# model.f4 = Var(model.t, within=PositiveReals,initialize=Xt4[3])
# model.V4 = Var(model.t, within=PositiveReals,initialize=Xt4[4])    




model.F_in1 = Var(model.t, within=NonNegativeReals,initialize=float(mu1[0][0]))
model.I1 = Var(model.t, within=NonNegativeReals,initialize=float(mu1[0][1]))


model.F_in2 = Var(model.t, within=NonNegativeReals,initialize=float(mu2[0][0]))
model.I2 = Var(model.t, within=NonNegativeReals,initialize=float(mu2[0][1]))


model.F_in3 = Var(model.t, within=NonNegativeReals,initialize=float(mu3[0][0]))
model.I3 = Var(model.t, within=NonNegativeReals,initialize=float(mu3[0][1]))


# model.F_in4 = Var(model.t, within=NonNegativeReals,initialize=float(mu4[0][0]))
# model.I4 = Var(model.t, within=NonNegativeReals,initialize=float(mu4[0][1]))



# model.N_in = Param(initialize=100)



def F_in1_def(model, t):
    if t <= tf_N*1:
        return model.F_in1[t] == float(mu1[0][0])
    elif tf_N*1 < t <= tf_N*2:
        return model.F_in1[t] == float(mu1[1][0])
    
    elif tf_N*2 < t <= tf_N*3:
        return model.F_in1[t] == float(mu1[2][0])
        
    # elif tf_N*3 < t <= tf_N*4:
    #     return m.Fn[t] == Fcn[3]
    else:
        return model.F_in1[t] == float(mu1[3][0])
model.F_in1_constr = Constraint(model.t, rule=F_in1_def)

def I1_def(model, t):
    if t <= tf_N*1:
        return model.I1[t] == float(mu1[0][1])
    elif tf_N*1 < t <= tf_N*2:
        return model.I1[t] == float(mu1[1][1])
    
    elif tf_N*2 < t <= tf_N*3:
        return model.I1[t] == float(mu1[2][1])
        
    # elif tf_N*3 < t <= tf_N*4:
    #     return m.Fn[t] == Fcn[3]
    else:
        return model.I1[t] == float(mu1[3][1])
model.I1_constr = Constraint(model.t, rule=I1_def)



def F_in2_def(model, t):
    if t <= tf_N*1:
        return model.F_in2[t] == float(mu2[0][0])
    elif tf_N*1 < t <= tf_N*2:
        return model.F_in2[t] == float(mu2[1][0])    
    elif tf_N*2 < t <= tf_N*3:
        return model.F_in2[t] == float(mu2[2][0])
        
    # elif tf_N*3 < t <= tf_N*4:
    #     return m.Fn[t] == Fcn[3]
    else:
        return model.F_in2[t] == float(mu2[3][0])
model.F_in2_constr = Constraint(model.t, rule=F_in2_def)

def I2_def(model, t):
    if t <= tf_N*1:
        return model.I2[t] == float(mu2[0][1])
    elif tf_N*1 < t <= tf_N*2:
        return model.I2[t] == float(mu2[1][1])
    
    elif tf_N*2 < t <= tf_N*3:
        return model.I2[t] == float(mu2[2][1])
        
    # elif tf_N*3 < t <= tf_N*4:
    #     return m.Fn[t] == Fcn[3]
    else:
        return model.I2[t] == float(mu2[3][1])
model.I2_constr = Constraint(model.t, rule=I2_def)


def F_in3_def(model, t):
    if t <= tf_N*1:
        return model.F_in3[t] == float(mu3[0][0])
    elif tf_N*1 < t <= tf_N*2:
        return model.F_in3[t] == float(mu3[1][0])    
    elif tf_N*2 < t <= tf_N*3:
        return model.F_in3[t] == float(mu3[2][0])
        
    # elif tf_N*3 < t <= tf_N*4:
    #     return m.Fn[t] == Fcn[3]
    else:
        return model.F_in3[t] == float(mu3[3][0])
model.F_in3_constr = Constraint(model.t, rule=F_in3_def)

def I3_def(model, t):
    if t <= tf_N*1:
        return model.I3[t] == float(mu3[0][1])
    elif tf_N*1 < t <= tf_N*2:
        return model.I3[t] == float(mu3[1][1])
    
    elif tf_N*2 < t <= tf_N*3:
        return model.I3[t] == float(mu3[2][1])
        
    # elif tf_N*3 < t <= tf_N*4:
    #     return m.Fn[t] == Fcn[3]
    else:
        return model.I3[t] == float(mu3[3][1])
model.I3_constr = Constraint(model.t, rule=I3_def)



# def F_in4_def(model, t):
#     if t <= tf_N*1:
#         return model.F_in4[t] == float(mu4[0][0])
#     elif tf_N*1 < t <= tf_N*2:
#         return model.F_in4[t] == float(mu4[1][0])    
#     elif tf_N*2 < t <= tf_N*3:
#         return model.F_in4[t] == float(mu4[2][0])
        
#     # elif tf_N*3 < t <= tf_N*4:
#     #     return m.Fn[t] == Fcn[3]
#     else:
#         return model.F_in4[t] == float(mu4[3][0])
# model.F_in4_constr = Constraint(model.t, rule=F_in4_def)

# def I4_def(model, t):
#     if t <= tf_N*1:
#         return model.I4[t] == float(mu4[0][1])
#     elif tf_N*1 < t <= tf_N*2:
#         return model.I4[t] == float(mu4[1][1])
    
#     elif tf_N*2 < t <= tf_N*3:
#         return model.I4[t] == float(mu4[2][1])
        
#     # elif tf_N*3 < t <= tf_N*4:
#     #     return m.Fn[t] == Fcn[3]
#     else:
#         return model.I4[t] == float(mu4[3][1])
# model.I4_constr = Constraint(model.t, rule=I4_def)







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
model.b2           = Var(domain = Reals,initialize=0,bounds = (-1,1))

model.node1 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))
model.node2 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))
model.node3 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))
# model.node4 = Var(model.k,model.I,model.t,initialize=0.1,bounds = (-1,1))


model.ud               = Var(domain = Reals, bounds=(0,0.1),  initialize=0)
model.un               = Var(domain = Reals, bounds=(0,5),  initialize=2.8)
model.kn               = Var(domain = Reals, bounds=(0,2),  initialize=1.4)
model.theta            = Var(domain = Reals, bounds=(5,10),  initialize=7.5)
model.gamma            = Var(domain = Reals, bounds=(5,10),  initialize=8.4)
model.epsilon          = Var(domain = Reals, bounds=(0,1),  initialize=0.1)


# Define u
model.u1     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1),initialize=0.05)
model.u2     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1),initialize=0.05)
model.u3     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1),initialize=0.05)
# model.u4     = Var(model.t,domain = NonNegativeReals, bounds=(0,0.1),initialize=0.05)


# # # defining derivatives
# model.x1dot = DerivativeVar(model.x1, wrt=model.t,domain = Reals,initialize = dXdt1[0])
# model.n1dot = DerivativeVar(model.n1, wrt=model.t,domain = Reals,initialize = dXdt1[1])
# model.q1dot = DerivativeVar(model.q1, wrt=model.t,domain = Reals,initialize = dXdt1[2])
# model.f1dot = DerivativeVar(model.f1, wrt=model.t,domain = Reals,initialize = dXdt1[3])
# model.V1dot = DerivativeVar(model.V1, wrt=model.t,domain = Reals,initialize = dXdt1[4])

# # defining derivatives
model.x1dot = DerivativeVar(model.x1, wrt=model.t,domain = Reals,initialize = dXdt1[0])
model.n1dot = DerivativeVar(model.n1, wrt=model.t,domain = Reals,initialize = dXdt1[1])
model.q1dot = DerivativeVar(model.q1, wrt=model.t,domain = Reals,initialize = dXdt1[2])
model.f1dot = DerivativeVar(model.f1, wrt=model.t,domain = Reals,initialize = dXdt1[3])
model.V1dot = DerivativeVar(model.V1, wrt=model.t,domain = Reals,initialize = dXdt1[4])

model.x2dot = DerivativeVar(model.x2, wrt=model.t,domain = Reals,initialize = dXdt2[0])
model.n2dot = DerivativeVar(model.n2, wrt=model.t,domain = Reals,initialize = dXdt2[1])
model.q2dot = DerivativeVar(model.q2, wrt=model.t,domain = Reals,initialize = dXdt2[2])
model.f2dot = DerivativeVar(model.f2, wrt=model.t,domain = Reals,initialize = dXdt2[3])
model.V2dot = DerivativeVar(model.V2, wrt=model.t,domain = Reals,initialize = dXdt2[4])

model.x3dot = DerivativeVar(model.x3, wrt=model.t,domain = Reals,initialize = dXdt3[0])
model.n3dot = DerivativeVar(model.n3, wrt=model.t,domain = Reals,initialize = dXdt3[1])
model.q3dot = DerivativeVar(model.q3, wrt=model.t,domain = Reals,initialize = dXdt3[2])
model.f3dot = DerivativeVar(model.f3, wrt=model.t,domain = Reals,initialize = dXdt3[3])
model.V3dot = DerivativeVar(model.V3, wrt=model.t,domain = Reals,initialize = dXdt3[4])


# model.x4dot = DerivativeVar(model.x4, wrt=model.t,domain = Reals,initialize = dXdt4[0])
# model.n4dot = DerivativeVar(model.n4, wrt=model.t,domain = Reals,initialize = dXdt4[1])
# model.q4dot = DerivativeVar(model.q4, wrt=model.t,domain = Reals,initialize = dXdt4[2])
# model.f4dot = DerivativeVar(model.f4, wrt=model.t,domain = Reals,initialize = dXdt4[3])
# model.V4dot = DerivativeVar(model.V4, wrt=model.t,domain = Reals,initialize = dXdt4[4])
# differential equation for u, x, n#

# EXP1 ------------------------------------
def NN_node11_exp1(model,t):

    return model.node1[1,1,t] == tanh(model.x1[t]/x_max*model.w1[1,1]+model.q1[t]/q_max*model.w1[1,2]+ model.I1[t]/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node11_exp1_cons = Constraint(model.t, rule = NN_node11_exp1)

def NN_node12_exp1(model,t):

    return model.node1[1,2,t] == tanh(model.x1[t]/x_max*model.w1[2,1]+model.q1[t]/q_max*model.w1[2,2]+ model.I1[t]/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node12_exp1_cons = Constraint(model.t, rule = NN_node12_exp1)

def NN_node13_exp1(model,t):
    return model.node1[1,3,t] == tanh(model.x1[t]/x_max*model.w1[3,1]+model.q1[t]/q_max*model.w1[3,2]+ model.I1[t]/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node13_exp1_cons = Constraint(model.t, rule = NN_node13_exp1)

def NN_node14_exp1(model,t):
    return model.node1[1,4,t] == tanh(model.x1[t]/x_max*model.w1[4,1]+model.q1[t]/q_max*model.w1[4,2]+ model.I1[t]/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node14_exp1_cons = Constraint(model.t, rule = NN_node14_exp1)

def NN_node15_exp1(model,t):
    return model.node1[1,5,t] == tanh(model.x1[t]/x_max*model.w1[5,1]+model.q1[t]/q_max*model.w1[5,2]+ model.I1[t]/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node15_exp1_cons = Constraint(model.t, rule = NN_node15_exp1)

def NN_node16_exp1(model,t):
    return model.node1[1,6,t] == tanh(model.x1[t]/x_max*model.w1[6,1]+model.q1[t]/q_max*model.w1[6,2]+ model.I1[t]/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node16_exp1_cons = Constraint(model.t, rule = NN_node16_exp1)

def NN_node17_exp1(model,t):
    return model.node1[1,7,t] == tanh(model.x1[t]/x_max*model.w1[7,1]+model.q1[t]/q_max*model.w1[7,2]+ model.I1[t]/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node17_exp1_cons = Constraint(model.t, rule = NN_node17_exp1)



def u1_con_exp1(model,t):
    return model.u1[t] == model.node1[1,1,t]*model.w2[1,1] + model.node1[1,2,t]*model.w2[1,2] + model.node1[1,3,t]*model.w2[1,3] + model.node1[1,4,t]*model.w2[1,4] + model.node1[1,5,t]*model.w2[1,5] + model.node1[1,6,t]*model.w2[1,6] + model.node1[1,7,t]*model.w2[1,7] + model.b2
model.u1_con_exp1  =Constraint(model.t,rule = u1_con_exp1)

# def u1_con_exp1(model,t):
#     return model.u1[t] == model.node1[1,1,t]*model.w2[1,1]+ model.node1[1,2,t]*model.w2[1,2] + model.b2
# model.u1_con_exp1  =Constraint(model.t,rule = u1_con_exp1)


def u1_e1_const(model, t):
  return model.u1[t] >= 0
model.u1_e1_const = Constraint(model.t, rule = u1_e1_const)


def dxdt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x1dot[t] == -model.F_in1[t]/model.V1[t]*model.x1[t] + model.u1[t] * model.x1[t] - model.ud * model.x1[t]
model.dxdtcon1 = Constraint(model.t, rule = dxdt1)

def dndt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n1dot[t] == model.F_in1[t]*(N_in - model.n1[t])/model.V1[t]-model.un*(model.n1[t]/(model.n1[t]+model.kn))*model.x1[t]
model.dndtcon1 = Constraint(model.t, rule = dndt1)

def dqdt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q1dot[t] == model.F_in1[t]*(N_in - model.n1[t])/(model.V1[t]*model.x1[t]) +model.un*(model.n1[t]/(model.n1[t]+model.kn)) +model.F_in1[t]*model.q1[t]/model.V1[t] - (model.u1[t]-model.ud)*model.q1[t]
model.dqdtcon1 = Constraint(model.t, rule = dqdt1)

def dfdt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f1dot[t] == model.u1[t]*(model.theta*model.q1[t]-model.epsilon*model.f1[t]) - model.gamma*model.un*(model.n1[t]/(model.n1[t]+model.kn)) + model.ud*model.epsilon*model.f1[t]
model.dfdtcon1 = Constraint(model.t, rule = dfdt1)

def dVdt1(model,t):
    if t == 0:
        return Constraint.Skip
    return model.V1dot[t] == model.F_in1[t]
model.dVdtcon1 = Constraint(model.t, rule = dVdt1)


# EXP2 ------------------------------------
def NN_node21_exp2(model,t):

    return model.node2[1,1,t] == tanh(model.x2[t]/x_max*model.w1[1,1]+model.q2[t]/q_max*model.w1[1,2]+ model.I2[t]/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node21_exp2_cons = Constraint(model.t, rule = NN_node21_exp2)

def NN_node22_exp2(model,t):

    return model.node2[1,2,t] == tanh(model.x2[t]/x_max*model.w1[2,1]+model.q2[t]/q_max*model.w1[2,2]+ model.I2[t]/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node22_exp2_cons = Constraint(model.t, rule = NN_node22_exp2)

def NN_node23_exp2(model,t):
    return model.node2[1,3,t] == tanh(model.x2[t]/x_max*model.w1[3,1]+model.q2[t]/q_max*model.w1[3,2]+ model.I2[t]/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node23_exp2_cons = Constraint(model.t, rule = NN_node23_exp2)

def NN_node24_exp2(model,t):
    return model.node2[1,4,t] == tanh(model.x2[t]/x_max*model.w1[4,1]+model.q2[t]/q_max*model.w1[4,2]+ model.I2[t]/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node24_exp2_cons = Constraint(model.t, rule = NN_node24_exp2)

def NN_node25_exp2(model,t):
    return model.node2[1,5,t] == tanh(model.x2[t]/x_max*model.w1[5,1]+model.q2[t]/q_max*model.w1[5,2]+ model.I2[t]/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node25_exp2_cons = Constraint(model.t, rule = NN_node25_exp2)

def NN_node26_exp2(model,t):
    return model.node2[1,6,t] == tanh(model.x2[t]/x_max*model.w1[6,1]+model.q2[t]/q_max*model.w1[6,2]+ model.I2[t]/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node26_exp2_cons = Constraint(model.t, rule = NN_node26_exp2)

def NN_node27_exp2(model,t):
    return model.node2[1,7,t] == tanh(model.x2[t]/x_max*model.w1[7,1]+model.q2[t]/q_max*model.w1[7,2]+ model.I2[t]/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node27_exp2_cons = Constraint(model.t, rule = NN_node27_exp2)


def u2_con_exp2(model,t):
    return model.u2[t] == model.node2[1,1,t]*model.w2[1,1] + model.node2[1,2,t]*model.w2[1,2] + model.node2[1,3,t]*model.w2[1,3] + model.node2[1,4,t]*model.w2[1,4] + model.node2[1,5,t]*model.w2[1,5] + model.node2[1,6,t]*model.w2[1,6] + model.node2[1,7,t]*model.w2[1,7] + model.b2
model.u2_con_exp2  =Constraint(model.t,rule = u2_con_exp2)

# def u2_con_exp2(model,t):
#     return model.u2[t] == model.node2[1,1,t]*model.w2[1,1]+ model.node2[1,2,t]*model.w2[1,2] + model.b2
# model.u2_con_exp2  =Constraint(model.t,rule = u2_con_exp2)

def u2_e2_const(model, t):
  return model.u2[t] >= 0
model.u2_e2_const = Constraint(model.t, rule = u2_e2_const)

def dxdt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x2dot[t] == -model.F_in2[t]/model.V2[t]*model.x2[t] + model.u2[t] * model.x2[t] - model.ud * model.x2[t]
model.dxdtcon2 = Constraint(model.t, rule = dxdt2)

def dndt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n2dot[t] == model.F_in2[t]*(N_in - model.n2[t])/model.V2[t]-model.un*(model.n2[t]/(model.n2[t]+model.kn))*model.x2[t]
model.dndtcon2 = Constraint(model.t, rule = dndt2)

def dqdt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q2dot[t] == model.F_in2[t]*(N_in - model.n2[t])/(model.V2[t]*model.x2[t]) +model.un*(model.n2[t]/(model.n2[t]+model.kn)) +model.F_in2[t]*model.q2[t]/model.V2[t] - (model.u2[t]-model.ud)*model.q2[t]
model.dqdtcon2 = Constraint(model.t, rule = dqdt2)

def dfdt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f2dot[t] == model.u2[t]*(model.theta*model.q2[t]-model.epsilon*model.f2[t]) - model.gamma*model.un*(model.n2[t]/(model.n2[t]+model.kn)) + model.ud*model.epsilon*model.f2[t]
model.dfdtcon2 = Constraint(model.t, rule = dfdt2)

def dVdt2(model,t):
    if t == 0:
        return Constraint.Skip
    return model.V2dot[t] == model.F_in2[t]
model.dVdtcon2 = Constraint(model.t, rule = dVdt2)

# EXP3 ------------------------------------
def NN_node31_exp3(model,t):

    return model.node3[1,1,t] == tanh(model.x3[t]/x_max*model.w1[1,1]+model.q3[t]/q_max*model.w1[1,2]+ model.I3[t]/I_max *model.w1[1,3] +model.b1[1,1])
model.NN_node31_exp3_cons = Constraint(model.t, rule = NN_node31_exp3)

def NN_node32_exp3(model,t):

    return model.node3[1,2,t] == tanh(model.x3[t]/x_max*model.w1[2,1]+model.q3[t]/q_max*model.w1[2,2]+ model.I3[t]/I_max *model.w1[2,3] +model.b1[1,2])
model.NN_node32_exp3_cons = Constraint(model.t, rule = NN_node32_exp3)

def NN_node33_exp3(model,t):
    return model.node3[1,3,t] == tanh(model.x3[t]/x_max*model.w1[3,1]+model.q3[t]/q_max*model.w1[3,2]+ model.I3[t]/I_max *model.w1[3,3] + model.b1[1,3])

model.NN_node33_exp3_cons = Constraint(model.t, rule = NN_node33_exp3)

def NN_node34_exp3(model,t):
    return model.node3[1,4,t] == tanh(model.x3[t]/x_max*model.w1[4,1]+model.q3[t]/q_max*model.w1[4,2]+ model.I3[t]/I_max *model.w1[4,3] + model.b1[1,4])

model.NN_node34_exp3_cons = Constraint(model.t, rule = NN_node34_exp3)

def NN_node35_exp3(model,t):
    return model.node3[1,5,t] == tanh(model.x3[t]/x_max*model.w1[5,1]+model.q3[t]/q_max*model.w1[5,2]+ model.I3[t]/I_max *model.w1[5,3] + model.b1[1,5])

model.NN_node35_exp3_cons = Constraint(model.t, rule = NN_node35_exp3)

def NN_node36_exp3(model,t):
    return model.node3[1,6,t] == tanh(model.x3[t]/x_max*model.w1[6,1]+model.q3[t]/q_max*model.w1[6,2]+ model.I3[t]/I_max *model.w1[6,3] + model.b1[1,6])

model.NN_node36_exp3_cons = Constraint(model.t, rule = NN_node36_exp3)

def NN_node37_exp3(model,t):
    return model.node3[1,7,t] == tanh(model.x3[t]/x_max*model.w1[7,1]+model.q3[t]/q_max*model.w1[7,2]+ model.I3[t]/I_max *model.w1[7,3] + model.b1[1,7])

model.NN_node37_exp3_cons = Constraint(model.t, rule = NN_node37_exp3)

def u3_con_exp3(model,t):
    return model.u3[t] == model.node3[1,1,t]*model.w2[1,1] + model.node3[1,2,t]*model.w2[1,2] + model.node3[1,3,t]*model.w2[1,3] + model.node3[1,4,t]*model.w2[1,4] + model.node3[1,5,t]*model.w2[1,5] + model.node3[1,6,t]*model.w2[1,6] + model.node3[1,7,t]*model.w2[1,7] + model.b2
model.u3_con_exp3  =Constraint(model.t,rule = u3_con_exp3)

# def u3_con_exp3(model,t):
#     return model.u3[t] == model.node3[1,1,t]*model.w2[1,1]+ model.node3[1,2,t]*model.w2[1,2] + model.b2
# model.u3_con_exp3  =Constraint(model.t,rule = u3_con_exp3)

def u3_e3_const(model, t):
  return model.u3[t] >= 0
model.u3_e3_const = Constraint(model.t, rule = u3_e3_const)

def dxdt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.x3dot[t] == -model.F_in3[t]/model.V3[t]*model.x3[t] + model.u3[t] * model.x3[t] - model.ud * model.x3[t]
model.dxdtcon3 = Constraint(model.t, rule = dxdt3)

def dndt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.n3dot[t] == model.F_in3[t]*(N_in - model.n3[t])/model.V3[t]-model.un*(model.n3[t]/(model.n3[t]+model.kn))*model.x3[t]
model.dndtcon3 = Constraint(model.t, rule = dndt3)

def dqdt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.q3dot[t] == model.F_in3[t]*(N_in - model.n3[t])/(model.V3[t]*model.x3[t]) +model.un*(model.n3[t]/(model.n3[t]+model.kn)) +model.F_in3[t]*model.q3[t]/model.V3[t] - (model.u3[t]-model.ud)*model.q3[t]
model.dqdtcon3 = Constraint(model.t, rule = dqdt3)

def dfdt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.f3dot[t] == model.u3[t]*(model.theta*model.q3[t]-model.epsilon*model.f3[t]) - model.gamma*model.un*(model.n3[t]/(model.n3[t]+model.kn)) + model.ud*model.epsilon*model.f3[t]
model.dfdtcon3 = Constraint(model.t, rule = dfdt3)

def dVdt3(model,t):
    if t == 0:
        return Constraint.Skip
    return model.V3dot[t] == model.F_in3[t]
model.dVdtcon3 = Constraint(model.t, rule = dVdt3)


# # EXP4 ------------------------------------
# def NN_node41_exp4(model,t):

#     return model.node4[1,1,t] == tanh(model.x4[t]/x_max*model.w1[1,1]+model.q4[t]/q_max*model.w1[1,2]+ model.I4[t]/I_max *model.w1[1,3] +model.b1[1,1])
# model.NN_node41_exp4_cons = Constraint(model.t, rule = NN_node41_exp4)

# def NN_node42_exp4(model,t):

#     return model.node4[1,2,t] == tanh(model.x4[t]/x_max*model.w1[2,1]+model.q4[t]/q_max*model.w1[2,2]+ model.I4[t]/I_max *model.w1[2,3] +model.b1[1,2])
# model.NN_node42_exp4_cons = Constraint(model.t, rule = NN_node42_exp4)

# def NN_node43_exp4(model,t):
#     return model.node4[1,3,t] == tanh(model.x4[t]/x_max*model.w1[3,1]+model.q4[t]/q_max*model.w1[3,2]+ model.I4[t]/I_max *model.w1[3,3] + model.b1[1,3])

# model.NN_node43_exp4_cons = Constraint(model.t, rule = NN_node43_exp4)

# def NN_node44_exp4(model,t):
#     return model.node4[1,4,t] == tanh(model.x4[t]/x_max*model.w1[4,1]+model.q4[t]/q_max*model.w1[4,2]+ model.I4[t]/I_max *model.w1[4,3] + model.b1[1,4])

# model.NN_node44_exp4_cons = Constraint(model.t, rule = NN_node44_exp4)

# def NN_node45_exp4(model,t):
#     return model.node4[1,5,t] == tanh(model.x4[t]/x_max*model.w1[5,1]+model.q4[t]/q_max*model.w1[5,2]+ model.I4[t]/I_max *model.w1[5,3] + model.b1[1,5])

# model.NN_node45_exp4_cons = Constraint(model.t, rule = NN_node45_exp4)

# def NN_node46_exp4(model,t):
#     return model.node4[1,6,t] == tanh(model.x4[t]/x_max*model.w1[6,1]+model.q4[t]/q_max*model.w1[6,2]+ model.I4[t]/I_max *model.w1[6,3] + model.b1[1,6])

# model.NN_node46_exp4_cons = Constraint(model.t, rule = NN_node46_exp4)

# def NN_node47_exp4(model,t):
#     return model.node4[1,7,t] == tanh(model.x4[t]/x_max*model.w1[7,1]+model.q4[t]/q_max*model.w1[7,2]+ model.I4[t]/I_max *model.w1[7,3] + model.b1[1,7])

# model.NN_node47_exp4_cons = Constraint(model.t, rule = NN_node47_exp4)

# def u4_con_exp4(model,t):
#     return model.u4[t] == model.node4[1,1,t]*model.w2[1,1] + model.node4[1,2,t]*model.w2[1,2] + model.node4[1,3,t]*model.w2[1,3] + model.node4[1,4,t]*model.w2[1,4] + model.node4[1,5,t]*model.w2[1,5] + model.node4[1,6,t]*model.w2[1,6] + model.node4[1,7,t]*model.w2[1,7] + model.b2
# model.u4_con_exp4  =Constraint(model.t,rule = u4_con_exp4)

# # def u4_con_exp4(model,t):
# #     return model.u4[t] == model.node4[1,1,t]*model.w2[1,1]+ model.node4[1,2,t]*model.w2[1,2] + model.b2
# # model.u4_con_exp4  =Constraint(model.t,rule = u4_con_exp4)

# def u4_e4_const(model, t):
#   return model.u4[t] >= 0
# model.u4_e4_const = Constraint(model.t, rule = u4_e4_const)

# def dxdt4(model,t):
#     if t == 0:
#         return Constraint.Skip
#     return model.x4dot[t] == -model.F_in4[t]/model.V4[t]*model.x4[t] + model.u4[t] * model.x4[t] - model.ud * model.x4[t]
# model.dxdtcon4 = Constraint(model.t, rule = dxdt4)

# def dndt4(model,t):
#     if t == 0:
#         return Constraint.Skip
#     return model.n4dot[t] == model.F_in4[t]*(N_in - model.n4[t])/model.V4[t]-model.un*(model.n4[t]/(model.n4[t]+model.kn))*model.x4[t]
# model.dndtcon4 = Constraint(model.t, rule = dndt4)

# def dqdt4(model,t):
#     if t == 0:
#         return Constraint.Skip
#     return model.q4dot[t] == model.F_in4[t]*(N_in - model.n4[t])/(model.V4[t]*model.x4[t]) +model.un*(model.n4[t]/(model.n4[t]+model.kn)) +model.F_in4[t]*model.q4[t]/model.V4[t] - (model.u4[t]-model.ud)*model.q4[t]
# model.dqdtcon4 = Constraint(model.t, rule = dqdt4)

# def dfdt4(model,t):
#     if t == 0:
#         return Constraint.Skip
#     return model.f4dot[t] == model.u4[t]*(model.theta*model.q4[t]-model.epsilon*model.f4[t]) - model.gamma*model.un*(model.n4[t]/(model.n4[t]+model.kn)) + model.ud*model.epsilon*model.f4[t]
# model.dfdtcon4 = Constraint(model.t, rule = dfdt4)

# def dVdt4(model,t):
#     if t == 0:
#         return Constraint.Skip
#     return model.V4dot[t] == model.F_in4[t]
# model.dVdtcon4 = Constraint(model.t, rule = dVdt4)







number_datapoints1 = xobs1.shape[1]
number_datapoints2 = xobs2.shape[1]
number_datapoints3 = xobs3.shape[1]
number_datapoints4 = xobs4.shape[1]


number_spc1 = xobs1.shape[0]
number_spc2 = xobs2.shape[0]
number_spc3 = xobs3.shape[0]
number_spc4 = xobs4.shape[0]
 

def obj(model):

 

    variance1    = (sum((model.x1[t]-model.x1_noise[t])**2 for t in model.tm)+sum((model.n1[t]-model.n1_noise[t])**2 for t in model.tm)+sum((model.q1[t]-model.q1_noise[t])**2 for t in model.tm)+sum((model.f1[t]-model.f1_noise[t])**2 for t in model.tm))/(number_datapoints1 * number_spc1)

    variance2    = (sum((model.x2[t]-model.x2_noise[t])**2 for t in model.tm)+sum((model.n2[t]-model.n2_noise[t])**2 for t in model.tm)+sum((model.q2[t]-model.q2_noise[t])**2 for t in model.tm)+sum((model.f2[t]-model.f2_noise[t])**2 for t in model.tm))/(number_datapoints2 * number_spc2)

    variance3    = (sum((model.x3[t]-model.x3_noise[t])**2 for t in model.tm)+sum((model.n3[t]-model.n3_noise[t])**2 for t in model.tm)+sum((model.q3[t]-model.q3_noise[t])**2 for t in model.tm)+sum((model.f3[t]-model.f3_noise[t])**2 for t in model.tm))/(number_datapoints3 * number_spc3)

    # variance4    = (sum((model.x4[t]-model.x4_noise[t])**2 for t in model.tm)+sum((model.n4[t]-model.n4_noise[t])**2 for t in model.tm)+sum((model.q4[t]-model.q4_noise[t])**2 for t in model.tm)+sum((model.f4[t]-model.f4_noise[t])**2 for t in model.tm))/(number_datapoints4 * number_spc4)

 

    obj1 = (sum((model.x1[t]-model.x1_noise[t])**2 for t in model.tm)+sum((model.n1[t]-model.n1_noise[t])**2 for t in model.tm)+sum((model.q1[t]-model.q1_noise[t])**2 for t in model.tm)+sum((model.f1[t]-model.f1_noise[t])**2 for t in model.tm))/2/(variance1+1e-12) - (number_datapoints1 * number_spc1)*log(1/(sqrt(2*3.14159*(variance1+1e-12))))

    obj2 = (sum((model.x2[t]-model.x2_noise[t])**2 for t in model.tm)+sum((model.n2[t]-model.n2_noise[t])**2 for t in model.tm)+sum((model.q2[t]-model.q2_noise[t])**2 for t in model.tm)+sum((model.f2[t]-model.f2_noise[t])**2 for t in model.tm))/2/(variance2+1e-12) - (number_datapoints2 * number_spc2)*log(1/(sqrt(2*3.14159*(variance2+1e-12))))

    obj3 = (sum((model.x3[t]-model.x3_noise[t])**2 for t in model.tm)+sum((model.n3[t]-model.n3_noise[t])**2 for t in model.tm)+sum((model.q3[t]-model.q3_noise[t])**2 for t in model.tm)+sum((model.f3[t]-model.f3_noise[t])**2 for t in model.tm))/2/(variance3+1e-12) - (number_datapoints3 * number_spc3)*log(1/(sqrt(2*3.14159*(variance3+1e-12))))

    # obj4 = (sum((model.x4[t]-model.x4_noise[t])**2 for t in model.tm)+sum((model.n4[t]-model.n4_noise[t])**2 for t in model.tm)+sum((model.q4[t]-model.q4_noise[t])**2 for t in model.tm)+sum((model.f4[t]-model.f4_noise[t])**2 for t in model.tm))/2/(variance4+1e-12) - (number_datapoints4 * number_spc4)*log(1/(sqrt(2*3.14159*(variance4+1e-12))))


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
instance.V1[0].fix(0.5)

instance.x2[0].fix(0.18)
instance.n2[0].fix(N2)
instance.q2[0].fix(80)
instance.f2[0].fix(120)
instance.V2[0].fix(0.5)

instance.x3[0].fix(0.18)
instance.n3[0].fix(N3)
instance.q3[0].fix(80)
instance.f3[0].fix(120)
instance.V3[0].fix(0.5)

# instance.x4[0].fix(0.18)
# instance.n4[0].fix(N4)
# instance.q4[0].fix(80)
# instance.f4[0].fix(120)
# instance.V4[0].fix(0.5)

discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(instance,nfe=15,ncp=3,wrt=instance.t,scheme='LAGRANGE-RADAU')


    # fix initial value

solver=SolverFactory('ipopt')
solver.options['max_iter'] = 100000
solver.options['tol'] = 1e-5
# solver.options['print_level'] = 5  # Adjust based on your needs

# # solver.options['max_cpu_time'] = 600  # Limit solver CPU time to 10 minutes
# solver.options['print_level'] = 10    # Increase verbosity for more detailed log output
# solver.options['hessian_approximation'] = 'limited-memory'  # Useful for large-scale problems

# results = solver.solve(instance, tee=True, logfile='solver_log.txt')
results = solver.solve(instance, tee=True)


structure_infor = [1,no_euron]

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