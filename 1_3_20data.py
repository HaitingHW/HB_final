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
import sys

eps  = np.finfo(float).eps
# np.random.seed(1)

''' Data treatment'''
def save_pkl(item, fname):
    sn = 'tmp3/' + fname
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


# x = load_pkl(sys.argv[1])

# idx = x['idx']
# input = x['inputs']

sp_list = [1, 0, 0, 0, 1, 0, 0, 0, 1]

xobs1 = load_pkl('data/xobs1.pkl')
xobs2 = load_pkl('data/xobs2.pkl')
xobs3 = load_pkl('data/xobs3.pkl')
xobs4 = load_pkl('data/xobs4.pkl')
Fcn   = load_pkl('data/Fcn.pkl')
tf_N  = load_pkl('data/tf_N.pkl')
operation_con = load_pkl('data/operation_con.pkl')

print(xobs1.shape,'data shape')
tt1 = load_pkl('data/tt1.pkl')
std_value1 = load_pkl('data/std_value1.pkl')
std_value2 = load_pkl('data/std_value2.pkl')
std_value3 = load_pkl('data/std_value3.pkl')
std_value4 = load_pkl('data/std_value4.pkl')
data_init = load_pkl('data/data_init.pkl')

operation_con1 = operation_con[0]
operation_con2 = operation_con[1]
operation_con3 = operation_con[2]
operation_con4 = operation_con[3]

Fcn1 = Fcn[0] 
Fcn2 = Fcn[1] 
Fcn3 = Fcn[2] 
Fcn4 = Fcn[3] 

tf    = 16.*24.
steps_= 20
dt    = tf/steps_


number_datapoints1 = xobs1.shape[1]
number_datapoints2 = xobs2.shape[1]
number_datapoints3 = xobs3.shape[1]
number_datapoints4 = xobs4.shape[1]


number_spc1 = xobs1.shape[0]
number_spc2 = xobs2.shape[0]
number_spc3 = xobs3.shape[0]
number_spc4 = xobs4.shape[0]


num_N = 4
tf_N  = tf/(num_N)
dstep_N = int(tf_N/dt)

def get_grad(x, t):
    dxdt = [[],[],[],[]]
    for n in range(x.shape[0]):
        for i in range(len(x[0]) - 1):
            dxdt[n].append((x[n][i + 1] - x[n][i])/(t[i + 1] - t[i]))
        dxdt[n].append(dxdt[n][-1])
    return dxdt

Xt1   = [to_dict(xobs1[0],dt), to_dict(xobs1[1],dt)] 
dXdt1 = [to_dict(get_grad(xobs1, tt1)[0],dt),to_dict(get_grad(xobs1, tt1)[1],dt)]

Xt2   = [to_dict(xobs2[0],dt), to_dict(xobs2[1],dt)] 
dXdt2 = [to_dict(get_grad(xobs2, tt1)[0],dt),to_dict(get_grad(xobs2, tt1)[1],dt)]

Xt3   = [to_dict(xobs3[0],dt), to_dict(xobs3[1],dt)] 
dXdt3 = [to_dict(get_grad(xobs3, tt1)[0],dt),to_dict(get_grad(xobs3, tt1)[1],dt)]

Xt4   = [to_dict(xobs4[0],dt), to_dict(xobs4[1],dt)] 
dXdt4 = [to_dict(get_grad(xobs4, tt1)[0],dt),to_dict(get_grad(xobs4, tt1)[1],dt)]

x_max = np.max(np.array([xobs1[0],xobs2[0],xobs3[0],xobs4[0]]))
n_max = np.max(np.array([xobs1[1],xobs2[1],xobs3[1],xobs4[1]]))

'''
def feed_gen(step, Fcn):
    Feed = np.zeros((step+1))
    for i in range(5):
        Feed[i] = Fcn[0]

    for i in range(5,9):
        Feed[i] = Fcn[1]

    for i in range(9,13):
        Feed[i] = Fcn[2]

    for i in range(13,17):
        Feed[i] = Fcn[3]
    return Feed
'''
tm = 16*24

def est_para(sp):
    model         = AbstractModel()

    # -- variable definition -- #

    # defining time as continous variable
    model.t       = ContinuousSet(bounds=[0, tm])


    # defining measurement times
    model.tm      = Set(within=model.t)



    # defining measured values as parameters
    model.x1_noise = Param(model.tm)
    model.n1_noise = Param(model.tm)

    model.x2_noise = Param(model.tm)
    model.n2_noise = Param(model.tm)
    
    model.x3_noise = Param(model.tm)
    model.n3_noise = Param(model.tm)
    
    # model.x4_noise = Param(model.tm)
    # model.n4_noise = Param(model.tm)

    # defining state variables
    model.x1 = Var(model.t, within=PositiveReals,initialize=Xt1[0]) 
    model.n1 = Var(model.t, within=PositiveReals,initialize=Xt1[1])
    
    model.x2 = Var(model.t, within=PositiveReals,initialize=Xt2[0]) 
    model.n2 = Var(model.t, within=PositiveReals,initialize=Xt2[1])
    
    model.x3 = Var(model.t, within=PositiveReals,initialize=Xt3[0]) 
    model.n3 = Var(model.t, within=PositiveReals,initialize=Xt3[1])
    
    # model.x4 = Var(model.t, within=PositiveReals,initialize=Xt4[0]) 
    # model.n4 = Var(model.t, within=PositiveReals,initialize=Xt4[1])

    model.Fcn1 = Var(model.t, within=NonNegativeReals,initialize=float(Fcn1[0]))
    model.Fcn2 = Var(model.t, within=NonNegativeReals,initialize=float(Fcn2[0]))
    model.Fcn3 = Var(model.t, within=NonNegativeReals,initialize=float(Fcn3[0]))
    # model.Fcn4 = Var(model.t, within=NonNegativeReals,initialize=float(Fcn4[0]))

    def Fcn1_def(model, t):
        if t <= tf_N*1:
            return model.Fcn1[t] == float(Fcn1[0])
        elif tf_N*1 < t <= tf_N*2:
            return model.Fcn1[t] == float(Fcn1[1])
        
        elif tf_N*2 < t <= tf_N*3:
            return model.Fcn1[t] == float(Fcn1[2])
            
        # elif tf_N*3 < t <= tf_N*4:
        #     return m.Fn[t] == Fcn[3]
        else:
            return model.Fcn1[t] == float(Fcn1[3])
    model.Fn1_constr = Constraint(model.t, rule=Fcn1_def)


    def Fcn2_def(model, t):
        if t <= tf_N*1:
            return model.Fcn2[t] == float(Fcn2[0])
        elif tf_N*1 < t <= tf_N*2:
            return model.Fcn2[t] == float(Fcn2[1])
        
        elif tf_N*2 < t <= tf_N*3:
            return model.Fcn2[t] == float(Fcn2[2])
            
        # elif tf_N*3 < t <= tf_N*4:
        #     return m.Fn[t] == Fcn[3]
        else:
            return model.Fcn2[t] == float(Fcn2[3])
    model.Fn2_constr = Constraint(model.t, rule=Fcn2_def)
    
    def Fcn3_def(model, t):
        if t <= tf_N*1:
            return model.Fcn3[t] == float(Fcn3[0])
        elif tf_N*1 < t <= tf_N*2:
            return model.Fcn3[t] == float(Fcn3[1])
        
        elif tf_N*2 < t <= tf_N*3:
            return model.Fcn3[t] == float(Fcn3[2])
            
        # elif tf_N*3 < t <= tf_N*4:
        #     return m.Fn[t] == Fcn[3]
        else:
            return model.Fcn3[t] == float(Fcn3[3])
    model.Fn3_constr = Constraint(model.t, rule=Fcn3_def)
    
    # def Fcn4_def(model, t):
    #     if t <= tf_N*1:
    #         return model.Fcn4[t] == float(Fcn4[0])
    #     elif tf_N*1 < t <= tf_N*2:
    #         return model.Fcn4[t] == float(Fcn4[1])
        
    #     elif tf_N*2 < t <= tf_N*3:
    #         return model.Fcn4[t] == float(Fcn4[2])
            
    #     # elif tf_N*3 < t <= tf_N*4:
    #     #     return m.Fn[t] == Fcn[3]
    #     else:
    #         return model.Fcn4[t] == float(Fcn4[3])
    # model.Fn4_constr = Constraint(model.t, rule=Fcn4_def)
    # # fix initial value

    # defining parameters to be determined
    model.p0       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01) 
    model.p1       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p2       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p3       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p4       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p5       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p6       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p7       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p8       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)
    model.p9       = Var(domain = Reals, bounds=(-0.1,0.1),  initialize=0.01)

    model.u_d          = Var(domain = Reals, bounds=(0,0.1),  initialize=0.01)
    model.Y_nx         = Var(domain = Reals, bounds=(400,600),initialize=500)


    std_x1        = std_value1[0]  
    std_n1        = std_value1[1]

    std_x2        = std_value2[0]  
    std_n2        = std_value2[1]
    
    std_x3        = std_value3[0]  
    std_n3        = std_value3[1]
    
    std_x4        = std_value4[0]  
    std_n4        = std_value4[1]
    
    # Define u
    model.u1     = Var(model.t,domain = Reals, bounds=(0,0.1))
    model.u2     = Var(model.t,domain = Reals, bounds=(0,0.1))
    model.u3     = Var(model.t,domain = Reals, bounds=(0,0.1))
    model.u4     = Var(model.t,domain = Reals, bounds=(0,0.1))

    # defining derivatives
    model.x1dot = DerivativeVar(model.x1, wrt=model.t,initialize=dXdt1[0])
    model.n1dot = DerivativeVar(model.n1, wrt=model.t,initialize=dXdt1[1])

    model.x2dot = DerivativeVar(model.x2, wrt=model.t,initialize=dXdt2[0])
    model.n2dot = DerivativeVar(model.n2, wrt=model.t,initialize=dXdt2[1])
    
    model.x3dot = DerivativeVar(model.x3, wrt=model.t,initialize=dXdt3[0])
    model.n3dot = DerivativeVar(model.n3, wrt=model.t,initialize=dXdt3[1])

    # model.x4dot = DerivativeVar(model.x4, wrt=model.t,initialize=dXdt4[0])
    # model.n4dot = DerivativeVar(model.n4, wrt=model.t,initialize=dXdt4[1])


    # -- differential equations -- #

    # differential equation for u, x, n#
    #EXP-1
    def h_u1(model,t):
        return model.u1[t] ==  model.p0 +sp[0]* model.p1 * model.x1[t]/x_max + sp[1]*model.p2 * model.n1[t]/n_max + sp[2]*model.p3 * (model.x1[t]/x_max)**2 + sp[3]*model.p4*model.x1[t]/x_max*model.n1[t]/n_max + sp[4]*model.p5*(model.n1[t]/n_max)**2 + sp[5]*model.p6*(model.x1[t]/x_max)**3 +  sp[6]*model.p7 * (model.x1[t]/x_max)**2*model.n1[t]/n_max + sp[7]*model.p8 *model.x1[t]/x_max*(model.n1[t]/n_max)**2 +  sp[8]*model.p9 *(model.n1[t]/n_max)**3
    model.h_u1con = Constraint(model.t, rule = h_u1)

    def x1dot(model,t):
        if t == 0:
            return Constraint.Skip
        return model.x1dot[t] == model.u1[t] * model.x1[t] - model.u_d * model.x1[t]**2
    model.x1dotcon = Constraint(model.t, rule = x1dot)

    def n1dot(model,t):
        if t == 0:
            return Constraint.Skip
        return model.n1dot[t] == -model.Y_nx*model.u1[t]*model.x1[t] + model.Fcn1[t]
    model.n1dotcon = Constraint(model.t, rule = n1dot)


    #EXP-2
    def h_u2(model,t):
        return model.u2[t] ==  model.p0 +sp[0]* model.p1 * model.x2[t]/x_max + sp[1]*model.p2 * model.n2[t]/n_max + sp[2]*model.p3 * (model.x2[t]/x_max)**2 + sp[3]*model.p4*model.x2[t]/x_max*model.n2[t]/n_max + sp[4]*model.p5*(model.n2[t]/n_max)**2 + sp[5]*model.p6*(model.x2[t]/x_max)**3 +  sp[6]*model.p7 * (model.x2[t]/x_max)**2*model.n2[t]/n_max + sp[7]*model.p8 *model.x2[t]/x_max*(model.n2[t]/n_max)**2 +  sp[8]*model.p9 *(model.n2[t]/n_max)**3
    model.h_u2con = Constraint(model.t, rule = h_u2)

    def x2dot(model,t):
        if t == 0:
            return Constraint.Skip
        return model.x2dot[t] == model.u2[t] * model.x2[t] - model.u_d * model.x2[t]**2
    model.x2dotcon = Constraint(model.t, rule = x2dot)

    def n2dot(model,t):
        if t == 0:
            return Constraint.Skip
        return model.n2dot[t] == -model.Y_nx*model.u2[t]*model.x2[t] + model.Fcn2[t]
    model.n2dotcon = Constraint(model.t, rule = n2dot)
    
    
    #EXP-3
    def h_u3(model,t):
        return model.u3[t] ==  model.p0 +sp[0]* model.p1 * model.x3[t]/x_max + sp[1]*model.p2 * model.n3[t]/n_max + sp[2]*model.p3 * (model.x3[t]/x_max)**2 + sp[3]*model.p4*model.x3[t]/x_max*model.n3[t]/n_max + sp[4]*model.p5*(model.n3[t]/n_max)**2 + sp[5]*model.p6*(model.x3[t]/x_max)**3 +  sp[6]*model.p7 * (model.x3[t]/x_max)**2*model.n3[t]/n_max + sp[7]*model.p8 *model.x3[t]/x_max*(model.n3[t]/n_max)**2 +  sp[8]*model.p9 *(model.n3[t]/n_max)**3
    model.h_u3con = Constraint(model.t, rule = h_u3)

    def x3dot(model,t):
        if t == 0:
            return Constraint.Skip
        return model.x3dot[t] == model.u3[t] * model.x3[t] - model.u_d * model.x3[t]**2
    model.x3dotcon = Constraint(model.t, rule = x3dot)

    def n3dot(model,t):
        if t == 0:
            return Constraint.Skip
        return model.n3dot[t] == -model.Y_nx*model.u3[t]*model.x3[t] + model.Fcn3[t]
    model.n3dotcon = Constraint(model.t, rule = n3dot)
    
    
    # #EXP-4
    # def h_u4(model,t):
    #     return model.u4[t] ==  model.p0 +sp[0]* model.p1 * model.x4[t]/x_max + sp[1]*model.p2 * model.n4[t]/n_max + sp[2]*model.p3 * (model.x4[t]/x_max)**2 + sp[3]*model.p4*model.x4[t]/x_max*model.n4[t]/n_max + sp[4]*model.p5*(model.n4[t]/n_max)**2 + sp[5]*model.p6*(model.x4[t]/x_max)**3 +  sp[6]*model.p7 * (model.x4[t]/x_max)**2*model.n4[t]/n_max + sp[7]*model.p8 *model.x4[t]/x_max*(model.n4[t]/n_max)**2 +  sp[8]*model.p9 *(model.n4[t]/n_max)**3
    # model.h_u4con = Constraint(model.t, rule = h_u4)

    # def x4dot(model,t):
    #     if t == 0:
    #         return Constraint.Skip
    #     return model.x4dot[t] == model.u4[t] * model.x4[t] - model.u_d * model.x4[t]**2
    # model.x4dotcon = Constraint(model.t, rule = x4dot)

    # def n4dot(model,t):
    #     if t == 0:
    #         return Constraint.Skip
    #     return model.n4dot[t] == -model.Y_nx*model.u4[t]*model.x4[t] + model.Fcn4[t]
    # model.n4dotcon = Constraint(model.t, rule = n4dot)



    def obj(model):
    
        variance1    = (sum((model.x1[t]-model.x1_noise[t])**2 for t in model.tm)+sum((model.n1[t]-model.n1_noise[t])**2 for t in model.tm))/(number_datapoints1 * number_spc1)+1**-10
        variance2    = (sum((model.x2[t]-model.x2_noise[t])**2 for t in model.tm)+sum((model.n2[t]-model.n2_noise[t])**2 for t in model.tm))/(number_datapoints2 * number_spc2)+1**-10
        variance3    = (sum((model.x3[t]-model.x3_noise[t])**2 for t in model.tm)+sum((model.n3[t]-model.n3_noise[t])**2 for t in model.tm))/(number_datapoints3 * number_spc3)+1**-10
        # variance4    = (sum((model.x4[t]-model.x4_noise[t])**2 for t in model.tm)+sum((model.n4[t]-model.n4_noise[t])**2 for t in model.tm))/(number_datapoints4 * number_spc4)+1**-10

        # Variance already has the square
        obj1 = (sum((model.x1[t]-model.x1_noise[t])**2 for t in model.tm)+sum((model.n1[t]-model.n1_noise[t])**2 for t in model.tm))/2/variance1 - (number_datapoints1 * number_spc1)*log(1/(sqrt(2*3.14159*variance1)))
        obj2 = (sum((model.x2[t]-model.x2_noise[t])**2 for t in model.tm)+sum((model.n2[t]-model.n2_noise[t])**2 for t in model.tm))/2/variance2 - (number_datapoints2 * number_spc2)*log(1/(sqrt(2*3.14159*variance2)))
        obj3 = (sum((model.x3[t]-model.x3_noise[t])**2 for t in model.tm)+sum((model.n3[t]-model.n3_noise[t])**2 for t in model.tm))/2/variance3 - (number_datapoints3 * number_spc3)*log(1/(sqrt(2*3.14159*variance3)))
        # obj4 = (sum((model.x4[t]-model.x4_noise[t])**2 for t in model.tm)+sum((model.n4[t]-model.n4_noise[t])**2 for t in model.tm))/2/variance4 - (number_datapoints4 * number_spc4)*log(1/(sqrt(2*3.14159*variance4)))

        #         # Variance already has the square
        # obj1 = (sum((model.x1[t]-model.x1_noise[t])**2 for t in model.tm)+sum((model.n1[t]-model.n1_noise[t])**2 for t in model.tm))/2
        # obj2 = (sum((model.x2[t]-model.x2_noise[t])**2 for t in model.tm)+sum((model.n2[t]-model.n2_noise[t])**2 for t in model.tm))/2
        # obj3 = (sum((model.x3[t]-model.x3_noise[t])**2 for t in model.tm)+sum((model.n3[t]-model.n3_noise[t])**2 for t in model.tm))/2
        # obj4 = (sum((model.x4[t]-model.x4_noise[t])**2 for t in model.tm)+sum((model.n4[t]-model.n4_noise[t])**2 for t in model.tm))/2
        return obj1+obj2+obj3

    model.obj = Objective(rule=obj)
        # -- model display -- #
    # model.pprint()

    # -- creating optimization problem -- #
    instance = model.create_instance(data_init)
    instance.x1[0].fix(operation_con1[0])
    instance.n1[0].fix(operation_con1[1])
    
    instance.x2[0].fix(operation_con2[0])
    instance.n2[0].fix(operation_con2[1])
    
    instance.x3[0].fix(operation_con3[0])
    instance.n3[0].fix(operation_con3[1])
    
    # instance.x4[0].fix(operation_con4[0])
    # instance.n4[0].fix(operation_con4[1])

    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(instance,nfe=10,ncp=3,wrt=instance.t,scheme='LAGRANGE-RADAU')

        # fix initial value

    solver=SolverFactory('ipopt')
    solver.options['max_iter'] = 100000
    solver.options['tol'] = 1e-5
    results = solver.solve(instance, tee=True)
    
    def save_para(results,sp):
        indices  = ''.join(str(e) for e in sp)
        obj      = results.obj()  
        u_d_o    = results.u_d()
        Y_nx_o    = results.Y_nx()
            
        p0_o     = results.p0()
        p1_o     = results.p1()
        p2_o     = results.p2()
        p3_o     = results.p3()
        p4_o     = results.p4()
        p5_o     = results.p5()
        p6_o     = results.p6()
        p7_o     = results.p7()
        p8_o     = results.p8()
        p9_o     = results.p9()



        df = pd.DataFrame([indices,obj,u_d_o, Y_nx_o,p0_o,p1_o,p2_o,p3_o,p4_o,p5_o,p6_o,p7_o,p8_o,p9_o]).T
        df.columns = ['parameters','obj','u_d','Y_nx','p0', 'p1', 'p2', 'p3','p4', 'p5', 'p6', 'p7','p8','p9']
        return df
    pe_result = save_para(instance,sp)

    return pe_result

test = est_para(sp_list)
# all_pe = []
# for i in range (len(sp_list)):
#     try:
#         test = est_para(sp_list[i])
#         all_pe.append(test)
        
#     except ValueError:
#         pass  # do nothing!

# save_pkl(all_pe,f'output.{idx}')