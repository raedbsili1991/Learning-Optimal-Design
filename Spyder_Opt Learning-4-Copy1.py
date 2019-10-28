# -*- coding: utf-8 -*-

import Performance_Single
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, Image, display, HTML
import time 
from numpy import linalg as LA


##############################################

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score,make_scorer
from scipy.stats import uniform as sp_rand
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

def normalize_x(x):return x/np.radians(85)
time_start = time.clock()


data = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs')
data2 = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs_2')
data3 = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs_3')
data4 = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs_4')
mode = pd.read_csv('6B IsoWS Data Generator/Log_dir/mode_1.csv')

#DATA = data1.dropna()

DATA = (pd.concat([mode.dropna(),data3.dropna(),data4.dropna()])).sample(frac=1)

data_size = 600

X = normalize_x(np.radians(DATA.values[:data_size,1:5]))
#X = normalize(DATA.values[:data_size,1:5],norm = 'l2')
Y = DATA.values[:data_size,5:9]
#%%
#Yf = np.multiply( np.multiply(Y[:,0],Y[:,1]), np.multiply(Y[:,2],Y[:,3]) ) 
#Yf = np.multiply( np.multiply(Y[:,0],Y[:,1]), Y[:,2] ) 
#Yff = np.column_stack(Yf,Y[:,3])
#Y0,Y1,Y2,Y3 = Y[:,0], Y[:,1], Y[:,2], Y[:,3]

Yf = Y

print("best data parameters from Delta min: ", DATA.values[np.where(Yf[:,0] == np.max(Yf[:,0]))[0],:] )
print("---------------------------")
print("best data parameters from OMEGA: ", DATA.values[np.where(Yf[:,3] == np.max(Yf[:,3]))[0],:] )


X_Training,X_Test,Y_Training, Y_Test = train_test_split(X, Yf, test_size=0.25, shuffle = True)


#%%
#######################################################


import warnings
warnings.filterwarnings("ignore")

print("pourcentage of non null Y0:", np.size(np.where(Yf[:,0]>0.01))/np.shape(Yf)[0])
print("pourcentage of non null Y1:", np.size(np.where(Yf[:,1]>0.01))/np.shape(Yf)[0])
print("pourcentage of non null Y2:", np.size(np.where(Yf[:,2]>0.01))/np.shape(Yf)[0])
print("pourcentage of non null Y3:", np.size(np.where(Yf[:,3]>0.01))/np.shape(Yf)[0])


print("Training size: ",data_size)

#
sigma_space = np.logspace(-2,4,num=50)
alpha_space = np.logspace(-5,0,num=50)

hyper_params = [{'kernel': ['rbf'],"alpha": alpha_space ,
                 "gamma": sigma_space,}] 



KR_estimator_0 = GridSearchCV(KernelRidge(),cv =6,
                  param_grid=hyper_params, scoring = 'neg_mean_squared_error')

KR_estimator_1 = GridSearchCV(KernelRidge(),cv =6,
                  param_grid=hyper_params, scoring = 'neg_mean_squared_error')
KR_estimator_2 = GridSearchCV(KernelRidge(),cv =6,
                  param_grid=hyper_params, scoring = 'neg_mean_squared_error')
KR_estimator_3 = GridSearchCV(KernelRidge(),cv =6,
                  param_grid=hyper_params, scoring = 'neg_mean_squared_error')

KR_estimator_0.fit(X_Training, Y_Training[:,0])
KR_estimator_1.fit(X_Training, Y_Training[:,1])
KR_estimator_2.fit(X_Training, Y_Training[:,2])
KR_estimator_3.fit(X_Training, Y_Training[:,3])

score_estimator_0_test = KR_estimator_0.score(X_Test, Y_Test[:,0])
score_estimator_1_test = KR_estimator_1.score(X_Test, Y_Test[:,1])
score_estimator_2_test = KR_estimator_2.score(X_Test, Y_Test[:,2])
score_estimator_3_test = KR_estimator_3.score(X_Test, Y_Test[:,3])

score_estimator_0_training = KR_estimator_0.score(X_Training, Y_Training[:,0])
score_estimator_1_training = KR_estimator_1.score(X_Training, Y_Training[:,1])
score_estimator_2_training = KR_estimator_2.score(X_Training, Y_Training[:,2])
score_estimator_3_training=  KR_estimator_3.score(X_Training, Y_Training[:,3])


ALPHA0,GAMMA0 = KR_estimator_0.best_params_['alpha'],KR_estimator_0.best_params_['gamma']
ALPHA1,GAMMA1 = KR_estimator_1.best_params_['alpha'],KR_estimator_1.best_params_['gamma']
ALPHA2,GAMMA2 = KR_estimator_2.best_params_['alpha'],KR_estimator_2.best_params_['gamma']
ALPHA3,GAMMA3 = KR_estimator_3.best_params_['alpha'],KR_estimator_3.best_params_['gamma']

                                

#print("Best hyperparams:",  KR_estimator.best_params_)
#ALPHAS=  [3.237457542817647e-05, 2.559547922699533e-05, 4.094915062380427e-05, 0.0013894954943731374]
#GAMMAS=  [2.8117686979742307, 0.6866488450043002, 1.2067926406393288, 1.5998587196060574]


#$$
print("ALPHAS: ", [ALPHA0,ALPHA1,ALPHA2,ALPHA3])
print("GAMMAS: ", [GAMMA0,GAMMA1,GAMMA2,GAMMA3])

#ALPHA0, GAMMA0 =  0.00101010101010101, 5.3366992312063095

kr0 = KernelRidge(kernel = 'rbf', alpha = ALPHA0, gamma = GAMMA0)
kr0.fit(X_Training, Y_Training[:,0])

print("Test/Training score here for Y0:", [-score_estimator_0_training,-score_estimator_0_test])
print("-------------")

kr1 = KernelRidge(kernel = 'rbf', alpha = ALPHA1, gamma = GAMMA1)
kr1.fit(X_Training, Y_Training[:,1])
print("Test/Training score here for Y1:", [-score_estimator_1_training,-score_estimator_1_test])
print("-------------")

kr2 = KernelRidge(kernel = 'rbf', alpha = ALPHA2, gamma = GAMMA2)
kr2.fit(X_Training, Y_Training[:,2])
print("Test/Training score here for Y2:", [score_estimator_2_training,score_estimator_2_test])
print("-------------")

kr3 = KernelRidge(kernel = 'rbf', alpha = ALPHA3, gamma = GAMMA3)
kr3.fit(X_Training, Y_Training[:,3])
print("Test/Training score here for Y3:", [score_estimator_3_training,score_estimator_3_test])
print("-------------")



###################################################


#########################################################

#####################
#%%
from array import array 
xk_array = array('d')

def grad_f(x,data):    
    ci, xj, gamma = data
    
    res = np.array([2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,0]-x[0]).T   ),
                    2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,1]-x[1]).T   ),
                    2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,2]-x[2]).T   ),
                    2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,3]-x[3]).T  )])

               
    return res

def Jac_F(x,data): 
    
    w0,w1,w2,w3 = data
    
    x = normalize_x(x) 
    
    grad_fi0 = grad_f(x,[kr0.dual_coef_,kr0.X_fit_,kr0.gamma]) 
    grad_fi1 = grad_f(x,[kr1.dual_coef_,kr1.X_fit_,kr1.gamma])
    grad_fi2 = grad_f(x,[kr2.dual_coef_,kr2.X_fit_,kr2.gamma])
    grad_fi3 = grad_f(x,[kr3.dual_coef_,kr3.X_fit_,kr3.gamma])
    
    x = x.reshape(1, -1)
    fi0 = kr0.predict(x)[0]
    fi1 = kr1.predict(x)[0] 
    fi2 = kr2.predict(x)[0] 
    fi3 = kr3.predict(x)[0]
    
    JF = - ( w0*fi0*grad_fi0 + w1*fi1*grad_fi1 + w2*fi2*grad_fi2 + w3*fi3*grad_fi3  )
    
    return JF/(w0+w1+w2+w3)

def w(x): 
    
    d = kr3.predict(normalize_x(x).reshape(1, -1))[0]
    
    return d - 0.5

def calling(xk):
    #xk_array.append(np.degrees(xk))
    print("xk now = ", np.degrees(xk))
    return 

def h(x): 
    
    f1 = kr0.predict(normalize_x(x).reshape(1, -1))[0]
    f2 = kr1.predict(normalize_x(x).reshape(1, -1))[0]
    f3 = kr2.predict(normalize_x(x).reshape(1, -1))[0]
    f4 = kr3.predict(normalize_x(x).reshape(1, -1))[0]
    
    return f1 - 0.8



def g(x):
    l1,l2,l3,alpha = x[0],x[1],x[2],x[3]
    return np.sin(l1)*np.cos(alpha) - (np.cos(l2)-np.sin(alpha)*np.cos(l1))


def get_x0(Yf,X,data):
    #X is already normalized !!!!!!
    w1,w2,w3,w4 = data
    
    f1,f2,f3,f4 =  Yf[:,0], Yf[:,1], Yf[:,2], Yf[:,3] 
    
    y_sorting = -(w1*f1 + w2*f2 + w3*f3 + w4*f4)/(w1+w2+w3+w4)
    
    
    n_samp = 10 # number of samples for the Pareto front plot
    best_y = Yf[y_sorting.argsort()][:n_samp,0:] # sort samples by corresponding sorting metric and select bests
    best_x = X[y_sorting.argsort()][:n_samp,0:]*np.radians(85) # corresponding best candidates UNORMALIZED!
    
    x0 = best_x[0]
    
    return x0


##############################
    
from scipy.optimize import minimize

upper_limit = 85
lower_limit = 9

l1 = np.random.rand() * (upper_limit - lower_limit) + lower_limit
l2 = np.random.rand() * (upper_limit - lower_limit) + lower_limit
l3 = np.random.rand() * (upper_limit - lower_limit) + lower_limit
alpha = np.random.rand() * (upper_limit - lower_limit) + lower_limit

## CONS & BNDS
bnd = [ (np.radians(9),np.radians(85)),
       (np.radians(9),np.radians(85)),
       (np.radians(9),np.radians(85)),
       (np.radians(9),np.radians(85)) ]

#w_WorkSpace = 0
#w_Uniformity = (1-w_WorkSpace)/3

#x0 = normalize_x(np.array(np.radians([l1,l2,l3,alpha])))
#x0 = normalize_x(np.array(np.radians([33.7, 83, 32.7 , 10.8])))
#x0 = normalize_x(get_x0(Yf))

#print("x0 is ", np.degrees(x0*np.radians(85)))

#print("best data parameters from Delta min: ", DATA.values[np.where(Yf[:,0] == np.max(Yf[:,0]))[0],:] )
#print("---------------------------")
#print("best data parameters from OMEGA: ", DATA.values[np.where(Yf[:,3] == np.max(Yf[:,3]))[0],:] )



cns = ({'type': 'ineq', 'fun':g})

#%%
def F(x,data):
    
    w0,w1,w2,w3 = data
    p = 2
    x = normalize_x(x).reshape(1, -1) 
    
    fi0 = kr0.predict(x)[0]
    fi1 = kr1.predict(x)[0] 
    fi2 = kr2.predict(x)[0] 
    fi3 = kr3.predict(x)[0]
    
    if fi3>1: return 1000
    F1 = - 0.5*( fi0 )**2
    F2 = - 0.5*( fi1 )**2
    F3 = - 0.5*( fi2 )**2
    F4 = - 0.5*( fi3 )**2
    
    #res = w1*F1 + w2*F2 + w3*F3 + w4*F4
    res = w0*(fi0-1)**p+w1*np.abs(fi1-1)**p+w2*np.abs(fi2-1)**p+w3*np.abs(fi3-1)**p
    
    
    return np.power(res,1/p)
data = [1,1,0,10]
x0 = normalize_x(get_x0(Yf,X,data))
res_x = minimize(F,x0, args = data, bounds = bnd, constraints = cns, 
                 tol = 1e-150,options={ 'disp': True, 'maxiter': 10000, 'ftol': 1e-10},
                jac = False,callback=calling)

print("----------- FINAL KERNEL OPTIMIZATION")
print("result is: ",np.degrees((res_x.x)))
print("-----------")
print("predicting proxy f0(x):",kr0.predict(normalize_x(res_x.x).reshape(1, -1))[0])
print("predicting proxy f1(x):",kr1.predict(normalize_x(res_x.x).reshape(1, -1))[0])
print("predicting proxy f2(x):",kr2.predict(normalize_x(res_x.x).reshape(1, -1))[0])
print("predicting proxy f3(x):",kr3.predict(normalize_x(res_x.x).reshape(1, -1))[0])
print("-----------")
y0 = kr0.predict(normalize_x(res_x.x).reshape(1, -1))[0]
y1 = kr1.predict(normalize_x(res_x.x).reshape(1, -1))[0]
y2 = kr2.predict(normalize_x(res_x.x).reshape(1, -1))[0]
y3 = kr3.predict(normalize_x(res_x.x).reshape(1, -1))[0]
print("-----------")
print("y predicted from f*", [y0,y1,y2,y3])
############
y_True = Performance_Single.PlotISO("learned_optimum",65,res_x.x)
print("y true = ", y_True)

#%%
cns = ({'type': 'ineq', 'fun':g})


w3_array = np.logspace(-2,5,num=5)

omega_arr = np.zeros(np.shape(w3_array))
delta_min_arr, delta_arr, delta_max = np.zeros(np.shape(w3_array)), np.zeros(np.shape(w3_array)), np.zeros(np.shape(w3_array))
y0_arr,y1_arr,y2_arr,y3_arr = np.zeros(np.shape(w3_array)),np.zeros(np.shape(w3_array)),np.zeros(np.shape(w3_array)),np.zeros(np.shape(w3_array))
l1_arr,l2_arr,l3_arr,alpha_arr = np.zeros(np.shape(w3_array)),np.zeros(np.shape(w3_array)),np.zeros(np.shape(w3_array)),np.zeros(np.shape(w3_array))

i = 0
for w3 in w3_array: 
    
    data = [1,1,1,w3]
    #x0 = normalize_x(get_x0(Yf,X,data))
    l1 = np.random.rand() * (upper_limit - lower_limit) + lower_limit
    l2 = np.random.rand() * (upper_limit - lower_limit) + lower_limit
    l3 = np.random.rand() * (upper_limit - lower_limit) + lower_limit
    alpha = np.random.rand() * (upper_limit - lower_limit) + lower_limit
    
    x0 = normalize_x(np.array(np.radians([l1,l2,l3,alpha])))
    res_x = minimize(F, x0, args = data, bounds = bnd, constraints = cns, 
                 tol = 1e-150,options={ 'disp': True, 'maxiter': 10000, 'ftol': 1e-10},
                jac = False)
    l1_arr[i],l2_arr[i],l3_arr[i],alpha_arr[i] = np.degrees(res_x.x[0]),np.degrees(res_x.x[1]),np.degrees(res_x.x[2]),np.degrees(res_x.x[3])
    
    
    y0_arr[i] = kr0.predict(normalize_x(res_x.x).reshape(1, -1))[0]
    y1_arr[i] = kr1.predict(normalize_x(res_x.x).reshape(1, -1))[0]
    y2_arr[i] = kr2.predict(normalize_x(res_x.x).reshape(1, -1))[0]
    y3_arr[i] = kr3.predict(normalize_x(res_x.x).reshape(1, -1))[0]    
    
    y_True = Performance_Single.PlotISO("learned_optimum",65,res_x.x)
    print("y true = ", y_True)
    delta_min_arr[i], delta_arr[i], delta_max[i], omega_arr[i] = y_True[0], y_True[1], y_True[2], y_True[3]
    
    i += 1 
    
    
print("computation done: ",(time.clock() - time_start))

def dropnan(arr, *args, **kwarg):
    assert isinstance(arr, np.ndarray)
    dropped=pd.DataFrame(arr).dropna(*args, **kwarg).values
    if arr.ndim==1:
        dropped=dropped.flatten()
    return dropped

from mpl_toolkits.mplot3d import Axes3D
#%%
plt.scatter(y3_arr,y0_arr,label='Learned Front')  
plt.scatter(omega_arr,delta_min_arr,label='True Front')
plt.scatter(Yf[:,0],Yf[:,3],label='Data Front')
#plt.scatter(best[:,3],best[:,2],label='Delta max')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.xlabel("Omega [-]")
plt.ylabel("Deltas [-]")
plt.legend()
plt.show()   

#%%
def ExportTo(filename, dimension, delta_time, kappa2_time,kappaF_time,
             a,b,c,d,e,l1_arr,l2_arr,l3_arr,alpha_arr):
    
    df = pd.DataFrame(data = {
                              'delta min [-]':dimension.T, 
                              'Delta  [-]':delta_time.T, 
                              'delta max [-]':kappa2_time.T,
                              'Omega [-]':kappaF_time.T, 
                              'w_WS/w_Uniformity':a.T, 
                              'predicted delta min [-]':b.T, 
                              'predicted Delta  [-]':c.T, 
                              'predicted delta max [-]':d.T,
                              'predicted Omega [-]':e.T, 
                              'optimum l1': l1_arr,
                              'optimum l2': l2_arr,
                              'optimum l3': l3_arr,
                              'optimum alpha': alpha_arr
                             })
#     df = pd.DataFrame(data = {
#                               'X [mm]':X_.T,
#                               'Y [mm]':Y_.T,
#                               'Z [mm]':Z_.T,
#                              })

    df.to_csv(filename)

    return
# 
#ExportTo("learned_results\FirstRun_learned_pareto_100_2",delta_min_arr, delta_arr, delta_max, omega_arr,w3_array, 
#         y0_arr,y1_arr,y2_arr,y3_arr,
#         l1_arr,l2_arr,l3_arr,alpha_arr
#         )    


#%%




 



#%%

#%%

def ExportTo(filename, dimension, delta_time, kappa2_time,kappaF_time,a):
    
    df = pd.DataFrame(data = {
                              'delta min [-]':dimension.T, 
                              'Delta  [-]':delta_time.T, 
                              'delta max [-]':kappa2_time.T,
                              'Omega [-]':kappaF_time.T, 
                              'w_WS/w_Uniformity':a.T
                             })
#     df = pd.DataFrame(data = {
#                               'X [mm]':X_.T,
#                               'Y [mm]':Y_.T,
#                               'Z [mm]':Z_.T,
#                              })

    df.to_csv(filename)

    return 


ExportTo("learned_pareto_100",delta_min_arr, delta_arr, delta_max, omega_arr,w3_array)

#%% USING ANALYTICAL GRADIENT

from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import check_grad


def grad_f(x,data):    
    ci, xj, gamma = data
    
    res = np.array([2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,0]-x[0])   ),
                    2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,1]-x[1])   ),
                    2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,2]-x[2])   ),
                    2*gamma*np.sum(  ci*rbf_kernel(x.reshape(1, -1),xj,gamma)*(xj[:,3]-x[3])  )])

               
    return res

def Jac_F(x,data): 
    
    w0,w1,w2,w3 = data
    
    x = normalize_x(x) 
    
    grad_fi0 = grad_f(x,[kr0.dual_coef_,kr0.X_fit_,kr0.gamma]) 
    grad_fi1 = grad_f(x,[kr1.dual_coef_,kr1.X_fit_,kr1.gamma])
    grad_fi2 = grad_f(x,[kr2.dual_coef_,kr2.X_fit_,kr2.gamma])
    grad_fi3 = grad_f(x,[kr3.dual_coef_,kr3.X_fit_,kr3.gamma])
    
    x = x.reshape(1, -1)
    fi0 = kr0.predict(x)[0]
    fi1 = kr1.predict(x)[0] 
    fi2 = kr2.predict(x)[0] 
    fi3 = kr3.predict(x)[0]
    
    JF = - ( w0*fi0*grad_fi0 + w1*fi1*grad_fi1 + w2*fi2*grad_fi2 + w3*fi3*grad_fi3  )
    
    return JF/(w0+w1+w2+w3)

def F(x,data):
    
    w1,w2,w3,w4 = data
    
    x = normalize_x(x).reshape(1, -1) 
    
    fi0 = kr0.predict(x)[0]
    fi1 = kr1.predict(x)[0] 
    fi2 = kr2.predict(x)[0] 
    fi3 = kr3.predict(x)[0]
    
    
    F1 = - 0.5*( fi0 )**2
    F2 = - 0.5*( fi1 )**2
    F3 = - 0.5*( fi2 )**2
    F4 = - 0.5*( fi3 )**2
    
    res = w1*F1 + w2*F2 + w3*F3 + w4*F4
    
    #if kr0.predict(normalize_x(x).reshape(1, -1))[0]>1.1: return 10
    
    
    
    return res/(w1+w2+w3+w4)
    
data = (1,1,0,1)
x0 = normalize_x(get_x0(Yf,X,data))
check_grad(F,Jac_F,x0,data)



#%%


##%%
#
#res_x = minimize(F,x0, args = data, bounds = bnd, constraints = cns, 
#                 tol = 1e-150,options={ 'disp': True, 'maxiter': 10000, 'ftol': 1e-10},
#                jac = False,callback=calling)
#
#print("----------- FINAL KERNEL OPTIMIZATION")
#print("result is: ",np.degrees((res_x.x)))
#print("-----------")
#print("predicting proxy f0(x):",kr0.predict(normalize_x(res_x.x).reshape(1, -1))[0])
#print("predicting proxy f1(x):",kr1.predict(normalize_x(res_x.x).reshape(1, -1))[0])
#print("predicting proxy f2(x):",kr2.predict(normalize_x(res_x.x).reshape(1, -1))[0])
#print("predicting proxy f3(x):",kr3.predict(normalize_x(res_x.x).reshape(1, -1))[0])
#print("-----------")
#y0 = kr0.predict(normalize_x(res_x.x).reshape(1, -1))[0]
#y1 = kr1.predict(normalize_x(res_x.x).reshape(1, -1))[0]
#y2 = kr2.predict(normalize_x(res_x.x).reshape(1, -1))[0]
#y3 = kr3.predict(normalize_x(res_x.x).reshape(1, -1))[0]
#print("-----------")
#print("y predicted from f*", [y0,y1,y2,y3])
#
#
################ 
#print("-----------")
#print("Verifying ground truth")
#import Performance_Single
#y_True= Performance_Single.PlotISO("learned_optimum",65,res_x.x)
#print("-----------")
#print("y true from f = ", y_True)
#
##%%
#
#from matplotlib.mlab import griddata
##%matplotlib notebook 
#
#def grid(x, y, z, resX=100, resY=100):
#
#    xi = np.linspace(min(x), max(x), resX)
#    yi = np.linspace(min(y), max(y), resY)
#    Z = griddata(x, y, z, xi, yi,interp='linear')
#    X, Y = np.meshgrid(xi, yi)
#    
#    return X, Y, Z
#
#def show_contours(theta1cr,theta5cr,delta,xtitle,ytitle,title_,N):
#    
#    T1,T5,DELTA = grid(theta1cr,theta5cr,delta, resX=100, resY=100)
#
#    cs = plt.contourf(T1,T5,DELTA,N,cmap=plt.cm.jet)
#    plt.title(title_)
#    plt.colorbar()
#    
#    
#    plt.xscale("log")
#    plt.yscale("log") 
#    plt.xlabel(xtitle)
#    plt.ylabel(ytitle)
#    
#    
#    plt.show()
#
#    #plt.show()
#    return cs
#
#    
#mean_test_0 = KR_estimator_0.cv_results_['mean_test_score']
#mean_test_1 = KR_estimator_1.cv_results_['mean_test_score']
#mean_test_2 = KR_estimator_2.cv_results_['mean_test_score']
#mean_test_3 = KR_estimator_3.cv_results_['mean_test_score']
#
#
#
#gamma_sp = np.logspace(-2,4,num=50)
#alpha_sp = np.logspace(-5,0,num=50)
#
#XX,YY= np.meshgrid(alpha_sp,gamma_sp )
#x_,y_ = np.ravel(XX),np.ravel(YY)
#
#
#plt.figure(1)
#show_contours(x_,y_,mean_test_0,'alphas','gammas','f0 (delta min)',300)
#
#plt.figure(2)
#show_contours(x_,y_,mean_test_1,'alphas','gammas','f1 (delta global)',300)
#
#plt.figure(3)
#show_contours(x_,y_,mean_test_2,'alphas','gammas','f2 (delta max)',300)
#
#plt.figure(4)
#show_contours(x_,y_,mean_test_3,'alphas','gammas','f3 (workspace number)',300)

#def F2(x,data):
#    
#    w1,w2,w3,w4 = data
#    
#    p = 2
#
#    f1 =  np.abs( kr0.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
#    f2 =  np.abs( kr1.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
#    f3 =  np.abs( kr2.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
#    f4 =  np.abs( kr3.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
#    
#    #res = np.average(np.array([f1,f2,f3,f4]),weights = [w1,w2,w3,w4])
#    res = (w1*f1 + w2*f2 + w3*f3 + w4*f4)/4
#    
#    return np.power(res,1/p)
#
#def F3(x,data):
#    
#    w1,w2,w3,w4 = data
#    x = normalize_x(x)
#    
#    f1 = np.abs( kr0.predict(x.reshape(1, -1))[0] - 1)**2
#    f2 = np.abs( kr1.predict(x.reshape(1, -1))[0] - 1)**2
#    f3 = np.abs( kr2.predict(x.reshape(1, -1))[0] - 1)**2
#    f4 = np.abs( kr3.predict(x.reshape(1, -1))[0] - 1)**2
#
#         
#    res = w1*f1 + w2*f2 + w3*f3 + w4*f4
#    
#    return res/(w1+w2+w3+w4)
