# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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

def scoring_rule(y,yhat):
    score = np.abs(y-yhat).max()
    return 1/score


def average_scoring(y,yhat):
    score = np.array([np.average(np.abs(y[:,0]-yhat[:,0])),
                     np.average(np.abs(y[:,1]-yhat[:,1])),
                     np.average(np.abs(y[:,2]-yhat[:,2])),
                     np.average(np.abs(y[:,3]-yhat[:,3]))]) 
    return score


data1 = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs')
data2 = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs_2')
data3 = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs_3')
data4 = pd.read_csv('6B IsoWS Data Generator/Log_dir/data_6b_lhs_4')

#DATA = data1.dropna()

DATA = pd.concat([data3.dropna(),data4.dropna(),data1.dropna()
                 ])

data_size = 600

X = np.radians(DATA.values[:data_size,1:5])/(np.radians(85))
#X = normalize(DATA.values[:data_size,1:5],norm = 'l2')
Y = DATA.values[:data_size,5:9]

#Yf = np.multiply( np.multiply(Y[:,0],Y[:,1]), np.multiply(Y[:,2],Y[:,3]) ) 
#Yf = np.multiply( np.multiply(Y[:,0],Y[:,1]), Y[:,2] ) 
#Yff = np.column_stack(Yf,Y[:,3])
#Y0,Y1,Y2,Y3 = Y[:,0], Y[:,1], Y[:,2], Y[:,3]

Yf = Y

print("best data parameters: ", DATA.values[np.where(Yf[:,0] == np.max(Yf[:,0]))[0],:] )


X_Training,X_Test,Y_Training, Y_Test = train_test_split(X, Yf, test_size=0.25, shuffle = True)




# min_Delta_Test = Y_Test[:,0]
# glob_Delta_Test =Y_Test[:,1]
# max_Delta_Test = Y_Test[:,2]
# WS_Test = Y_Test[:,3]


#######################################################


import warnings
warnings.filterwarnings("ignore")

print("pourcentage of non null Y0:", np.size(np.where(Yf[:,0]>0.01))/np.shape(Yf)[0])
print("pourcentage of non null Y1:", np.size(np.where(Yf[:,1]>0.01))/np.shape(Yf)[0])
print("pourcentage of non null Y2:", np.size(np.where(Yf[:,2]>0.01))/np.shape(Yf)[0])
print("pourcentage of non null Y3:", np.size(np.where(Yf[:,3]>0.01))/np.shape(Yf)[0])


print("Training size: ",data_size)


sigma_space = np.logspace(-2,4,num=50)
alpha_space = np.logspace(-5,0,num=50)

hyper_params = [{'kernel': ['rbf'],"alpha": alpha_space ,
                 "gamma": sigma_space,}] 



KR_estimator_0 = GridSearchCV(KernelRidge(),cv =3,
                  param_grid=hyper_params, scoring = 'r2')

KR_estimator_1 = GridSearchCV(KernelRidge(),cv =3,
                  param_grid=hyper_params, scoring = 'r2')
KR_estimator_2 = GridSearchCV(KernelRidge(),cv =3,
                  param_grid=hyper_params, scoring = 'r2')
KR_estimator_3 = GridSearchCV(KernelRidge(),cv =3,
                  param_grid=hyper_params, scoring = 'r2')

KR_estimator_0.fit(X_Training, Y_Training[:,0])
score_estimator_0 = KR_estimator_0.score(X_Test, Y_Test[:,0])

KR_estimator_1.fit(X_Training, Y_Training[:,1])
score_estimator_1 = KR_estimator_0.score(X_Test, Y_Test[:,1])

KR_estimator_2.fit(X_Training, Y_Training[:,2])
score_estimator_2 = KR_estimator_0.score(X_Test, Y_Test[:,2])

KR_estimator_3.fit(X_Training, Y_Training[:,3])
score_estimator_3 = KR_estimator_3.score(X_Test, Y_Test[:,3])


ALPHA0,GAMMA0 = KR_estimator_0.best_params_['alpha'],KR_estimator_0.best_params_['gamma']
ALPHA1,GAMMA1 = KR_estimator_1.best_params_['alpha'],KR_estimator_1.best_params_['gamma']
ALPHA2,GAMMA2 = KR_estimator_2.best_params_['alpha'],KR_estimator_2.best_params_['gamma']
ALPHA3,GAMMA3 = KR_estimator_3.best_params_['alpha'],KR_estimator_3.best_params_['gamma']

                                
# print("Training score for 0:", KR_estimator.score(X_Training, Y_Training[:,0]))
# print("Test score:", score_estimator_0)

#print("Best hyperparams:",  KR_estimator.best_params_)

print("ALPHAS: ", [ALPHA0,ALPHA1,ALPHA2,ALPHA3])
print("GAMMAS: ", [GAMMA0,GAMMA1,GAMMA2,GAMMA3])

#ALPHA0, GAMMA0 =  0.00101010101010101, 5.3366992312063095

kr0 = KernelRidge(kernel = 'rbf', alpha = ALPHA0, gamma = GAMMA0)
score_trainning_0 = kr0.fit(X_Training, Y_Training[:,0]).score(X_Training, Y_Training[:,0])
score_test_0 = kr0.fit(X_Training, Y_Training[:,0]).score(X_Test, Y_Test[:,0])
learned_coef_0 = kr0.dual_coef_

print("Training score here for Y0:", score_trainning_0)
print("Test score here for Y0:", score_test_0)
print("-------------")

kr1 = KernelRidge(kernel = 'rbf', alpha = ALPHA1, gamma = GAMMA1)
score_trainning_1 = kr1.fit(X_Training, Y_Training[:,1]).score(X_Training, Y_Training[:,1])
score_test_1 = kr1.fit(X_Training, Y_Training[:,1]).score(X_Test, Y_Test[:,1])
learned_coef_1 = kr1.dual_coef_

print("Training score here for Y1:", score_trainning_1)
print("Test score here for Y1:", score_test_1)
print("-------------")

kr2 = KernelRidge(kernel = 'rbf', alpha = ALPHA2, gamma = GAMMA2)

score_trainning_2 = kr2.fit(X_Training, Y_Training[:,2]).score(X_Training, Y_Training[:,2])
score_test_2 = kr2.fit(X_Training, Y_Training[:,2]).score(X_Test, Y_Test[:,2])
learned_coef_2 = kr2.dual_coef_

print("Training score here for Y2:", score_trainning_2)
print("Test score here for Y2:", score_test_2)
print("-------------")

kr3 = KernelRidge(kernel = 'rbf', alpha = ALPHA3, gamma = GAMMA3)
score_trainning_3 = kr3.fit(X_Training, Y_Training[:,3]).score(X_Training, Y_Training[:,3])
score_test_3 = kr3.fit(X_Training, Y_Training[:,3]).score(X_Test, Y_Test[:,3])
learned_coef_3 = kr3.dual_coef_

print("Training score here for Y3:", score_trainning_3)
print("Test score here for Y3:", score_test_3)
print("-------------")



###################################################

from matplotlib.mlab import griddata
#%matplotlib notebook 

def grid(x, y, z, resX=100, resY=100):

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = np.meshgrid(xi, yi)
    
    return X, Y, Z

def show_contours(theta1cr,theta5cr,delta,xtitle,ytitle,title_,N):
    
    T1,T5,DELTA = grid(theta1cr,theta5cr,delta, resX=100, resY=100)

    cs = plt.contourf(T1,T5,DELTA,N,cmap=plt.cm.jet)
    plt.title(title_)
    plt.colorbar()
    
    
    plt.xscale("log")
    plt.yscale("log") 
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    
    
    plt.show()

    #plt.show()
    return cs


#########################################################
    
mean_test_0 = KR_estimator_0.cv_results_['mean_test_score']
mean_test_1 = KR_estimator_1.cv_results_['mean_test_score']
mean_test_2 = KR_estimator_2.cv_results_['mean_test_score']
mean_test_3 = KR_estimator_3.cv_results_['mean_test_score']



gamma_sp = np.logspace(-2,4,num=50)
alpha_sp = np.logspace(-5,0,num=50)

X,Y= np.meshgrid(alpha_sp,gamma_sp )
x_,y_ = np.ravel(X),np.ravel(Y)


plt.figure(1)
show_contours(x_,y_,mean_test_0,'alphas','gammas','f0 (delta min)',300)

plt.figure(2)
show_contours(x_,y_,mean_test_1,'alphas','gammas','f1 (delta global)',300)

plt.figure(3)
show_contours(x_,y_,mean_test_2,'alphas','gammas','f2 (delta max)',300)

plt.figure(4)
show_contours(x_,y_,mean_test_3,'alphas','gammas','f3 (workspace number)',300)



#####################

def f(x,data):
    
    coeffs, xi, gamma = data 
    #C = np.sum(np.multiply(coeffs,k(x,xi,gamma)))
    C = kr.predict((x).reshape(1, -1))[0]
        
    return C

def normalize_x(x):return x/np.radians(85)

def F(x,data):
    
    w1,w2,w3,w4 = data
    

    f1 = - 0.5*( kr0.predict(normalize_x(x).reshape(1, -1))[0] )**2
    f2 = - 0.5*( kr1.predict(normalize_x(x).reshape(1, -1))[0] )**2
    f3 = - 0.5*( kr2.predict(normalize_x(x).reshape(1, -1))[0] )**2
    f4 = - 0.5*( kr3.predict(normalize_x(x).reshape(1, -1))[0] )**2
    
    res = w1*f1 + w2*f2 + w3*f3 + w4*f4
    
    return res/4

def F2(x,data):
    
    w1,w2,w3,w4 = data
    
    p = 2

    f1 =  np.abs( kr0.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
    f2 =  np.abs( kr1.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
    f3 =  np.abs( kr2.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
    f4 =  np.abs( kr3.predict(normalize_x(x).reshape(1, -1))[0] - 1)**2
    
    #res = np.average(np.array([f1,f2,f3,f4]),weights = [w1,w2,w3,w4])
    res = (w1*f1 + w2*f2 + w3*f3 + w4*f4)/4
    
    return np.power(res,1/p)

def F3(x,data):
    
    w1,w2,w3,w4 = data
    

    f1 = - 0.5*( kr0.predict(normalize_x(x).reshape(1, -1))[0] )**2
    f2 = - 0.5*( kr1.predict(normalize_x(x).reshape(1, -1))[0] )**2
    f3 = - 0.5*( kr2.predict(normalize_x(x).reshape(1, -1))[0] )**2
    f4 = - 0.5*( kr3.predict(normalize_x(x).reshape(1, -1))[0] )**2
    
    res = w1*f1 + w2*f2 + w3*f3
    
    return res

def w(x): 
    
    a = kr0.predict(normalize_x(x).reshape(1, -1))[0]
    b = kr1.predict(normalize_x(x).reshape(1, -1))[0]
    c = kr2.predict(normalize_x(x).reshape(1, -1))[0]
    d = kr3.predict(normalize_x(x).reshape(1, -1))[0]
    
    return d - 0.5

def calling(x):
    print(np.degrees(x))
    return 

def h(x): 
    
    a = kr0.predict(normalize_x(x).reshape(1, -1))[0]
    b = kr1.predict(normalize_x(x).reshape(1, -1))[0]
    c = kr2.predict(normalize_x(x).reshape(1, -1))[0]
    d = kr3.predict(normalize_x(x).reshape(1, -1))[0]
    
    return 1 - a*b*c*d



def g(x):
    l1,l2,l3,alpha = x[0],x[1],x[2],x[3]
    return np.sin(l1)*np.cos(alpha) - (np.cos(l2)-np.sin(alpha)*np.cos(l1))

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

x0 = normalize_x(np.array(np.radians([l1,l2,l3,alpha])))
#x0 = normalize_x(np.array(np.radians([45,45,45,45])))

w_WorkSpace = 0
w_Uniformity = (1-w_WorkSpace)/3
data = [w_Uniformity,w_Uniformity,w_Uniformity,w_WorkSpace]

cns = ({'type': 'ineq', 'fun':g},
       {'type': 'ineq', 'fun':w})

## OPT

res_x = minimize(F3,x0, args = data, bounds = bnd, constraints = cns, 
                 tol = 1e-150,options={ 'disp': True, 'maxiter': 10000},
                jac = False,callback=calling)

print("-----------")
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


############### 
import Performance_Single
y_True= Performance_Single.PlotISO("learned_optimum",65,res_x.x)
print("-----------")
print("y true from f = ", y_True)