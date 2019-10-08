## SYMBOLIC COMPUTATIONS OF FORWARD KINEMATICS FOR OMNIWIRST
# Raed Bsili, Istituto Italiano di Tecnologia 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import sympy as sy
from sympy import Matrix, sin, cos, tan, pi, symbols, solve, nsolve
sy.init_printing(use_unicode=True)
from matplotlib.mlab import griddata
from numpy import linalg as LA
import time 
import math
########
from scipy.optimize import root, minimize, differential_evolution 
import scipy.optimize as sc
#######
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#####
###
from scipy.spatial import ConvexHull
###
import pandas as pd

###



#------------------------------------------------------------------------------
# Decalre symbols

l1, l2, l3, alpha = symbols('l1 l2 l3 alpha') # link arc angles

# Inputs angle
theta_L, theta_R = symbols('theta_L theta_R')
#Zeros 
theta_L0, theta_R0 = symbols('theta_L0 theta_R0')
uL0, vL0 = symbols('uL0 vL0')
uR0, vR0 = symbols('uR0 vR0')
# Passive Angles
u_L, u_R, v_L, v_R= symbols('u_L u_R v_L v_R')  #passive angles 
# Euler angles
x, y, z = symbols('x y z') # platform coordinates
theta,psi,phi = symbols('theta psi phi') # platform roll, pitch, yaw 
#------------------------------------------------------------------------------

Rl = symbols('Rl')

#-------------------------------------------------------------------------------

# RotationSym Function
def RotationSym(angle, axis):
    
    if axis == 'Z':
        ROT = sy.Matrix([[cos(angle), -sin(angle), 0, 0], [sin(angle), cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
    elif axis == 'Y':
        ROT = sy.Matrix([[cos(angle), 0, sin(angle), 0], [0, 1, 0, 0], [-sin(angle), 0, cos(angle), 0], [0, 0, 0, 1]])
        
    elif axis == 'X':
        ROT = sy.Matrix([[1, 0, 0, 0], [0, cos(angle), -sin(angle), 0], [0, sin(angle), cos(angle), 0], [0, 0, 0, 1]])
    
    else:
        print('Incorrect axis definition. Use: "X,Y,Z"')
    
    return ROT
#------------------------------------------------------------------------------
# TranslationSym Function
def TranslationSym(position):
    
    x, y, z = position[0], position[1], position[2]
    TRS = sy.Matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    
    return TRS

def VectorSym(position):
    x, y, z = position[0], position[1], position[2]
    return sy.Matrix([[x], [y], [z]])
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# Compute sy.Matrix C for Platform
RZc = RotationSym(phi, 'Z')
RYc = RotationSym(theta, 'Y')
RXc = RotationSym(psi, 'X')

ROTc = RYc*RZc
POSc = TranslationSym([x, y, z])

C = POSc*ROTc
#------------------------------------------------------------------------------




theta_pitch, phi_yaw = symbols('theta_pitch phi_yaw')

RO_L = RotationSym(theta_pitch,'Y')*RotationSym(phi_yaw,'Z')*RotationSym(-pi/2,'Z')*RotationSym(pi/2 - l3,'X')*RotationSym(alpha,'Z')
VL = (RO_L[:3,:3])[:,1]
UL = ((RotationSym(theta_L, 'Y')*RotationSym(l1, 'X'))[:3,:3])[:,1]

#--

RO_R = RotationSym(theta_pitch,'Y')*RotationSym(phi_yaw,'Z')*RotationSym(-pi/2,'Z')*RotationSym(pi/2 - l3,'X')*RotationSym(-alpha,'Z')
VR = (RO_R[:3,:3])[:,1]
UR = ((RotationSym(pi,'Z')*RotationSym(theta_R, 'Y')*RotationSym(l1, 'X'))[:3,:3])[:,1]

fL = (sy.Transpose(VL)*UL)[0]-cos(l2)
fR = (sy.Transpose(VR)*UR)[0]-cos(l2)

fL = sy.collect(sy.trigsimp(fL,deep='True'),sin(theta_L-theta_pitch))
fR = sy.collect(sy.trigsimp(fR,deep='True'),cos(theta_R+theta_pitch))
f3 = (sy.Transpose(VL)*VR)[0]-cos(2*alpha)

dataL = (theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

UL_,VL_ = sy.lambdify(dataL,UL),sy.lambdify(dataL,VL)
UR_,VR_ = sy.lambdify(dataL,UR),sy.lambdify(dataL,VR)

FL_ = sy.lambdify(dataL,fL)
FR_ = sy.lambdify(dataL,fR)
F3_ = sy.lambdify(dataL,f3)


gL = (sy.Transpose(UL.cross(VL))*VectorSym([0,1,0]))[0,0]
gL = sy.simplify(gL)
gR = (sy.Transpose(UR.cross(VR))*VectorSym([0,1,0]))[0,0]
gR = sy.simplify(gR)


jacL_constraint = sy.Matrix([gL]).jacobian([theta_pitch,phi_yaw])
jacR_constraint = sy.Matrix([gR]).jacobian([theta_pitch,phi_yaw])

GL_ = sy.lambdify(dataL,gL)
GR_ = sy.lambdify(dataL,gR)
JLcs_ = sy.lambdify(dataL,jacL_constraint)
JRcs_ = sy.lambdify(dataL,jacR_constraint)

F = pow(fL,2) + pow(fR,2)
Jf = sy.Matrix([F]).jacobian([theta_pitch,phi_yaw])
Hf = sy.hessian(F,(theta_pitch,phi_yaw))

F_ = sy.lambdify(dataL,F)
JF_ = sy.lambdify(dataL,Jf)
HF_ = sy.lambdify(dataL,Hf)

d_FL_L,d_FL_R = sy.diff(fL,theta_L),sy.diff(fL,theta_R)
d_FR_L,d_FR_R = sy.diff(fR,theta_L),sy.diff(fR,theta_R)

d_FL_P,d_FL_Y = sy.simplify(sy.diff(fL,theta_pitch)),sy.diff(fL,phi_yaw)
d_FR_P,d_FR_Y = sy.simplify(sy.diff(fR,theta_pitch)),sy.diff(fR,phi_yaw)

Ji = sy.Matrix([ [d_FL_L,d_FL_R],
        [d_FR_L,d_FR_R] ])
Jo = sy.Matrix([ [d_FL_P,d_FL_Y],
        [d_FR_P,d_FR_Y] ])
dataL = (theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

Ji_ = sy.lambdify(dataL,Ji)
Jo_ = sy.lambdify(dataL,Jo)

J = -Jo.inv()*Ji

J_ = sy.lambdify(dataL,J)

###################

#ISOTROPY

#####################




def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def SolveFK(E,data):
    
    theta_pitch,phi_yaw = E[0],E[1]
    theta_L,theta_R,l1,l2,l3,alpha = data 
    
    cost = [FL_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha),
           FR_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)]
     
    return cost

def SolveFK_min(E,data):
    
    theta_pitch,phi_yaw = E[0],E[1]
    theta_L,theta_R,l1,l2,l3,alpha = data 
    
#     cost = [FL_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha),
#            FR_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha),
#            F3_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)]
    #cost[0]**2 + cost[1]**2 + cost[2]**2
 
    return F_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

def Jac_F(E,data):
    theta_pitch,phi_yaw = E[0],E[1]
    theta_L,theta_R,l1,l2,l3,alpha = data 
    return JF_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

def Hess_F(E,data):
    theta_pitch,phi_yaw = E[0],E[1]
    theta_L,theta_R,l1,l2,l3,alpha = data 
    return F_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

def SolveFK_evo(E,*data):
    
    theta_pitch,phi_yaw = E[0],E[1]
    theta_L,theta_R,l1,l2,l3,alpha = data 
    
    cost = [FL_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha),
           FR_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)]
     
    return cost[0]**2 + cost[1]**2

def Cost(theta_pitch,phi_yaw,data):
    
    theta_L,theta_R,l1,l2,l3,alpha = data 
    
    cost = FL_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)**2 + FR_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)**2+0*F3_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)**2
     
    return cost


def f_cons(E,theta_L,theta_R,l1,l2,l3,alpha):
    
    theta_pitch,phi_yaw = E[0],E[1]
    
    ul = UL_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)
    vl = VL_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

    C = np.cross(ul.reshape(1,3), vl.reshape(1,3))[0][1]
     
    return C

def fl_cons(E,theta_L,theta_R,l1,l2,l3,alpha):
    theta_pitch,phi_yaw = E[0],E[1]
    return GL_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

def fr_cons(E,theta_L,theta_R,l1,l2,l3,alpha):
    theta_pitch,phi_yaw = E[0],E[1]   
    return GR_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

def Jac_fl_cons(E,theta_L,theta_R,l1,l2,l3,alpha):
    theta_pitch,phi_yaw = E[0],E[1]
    return JLcs_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)
def Jac_fr_cons(E,theta_L,theta_R,l1,l2,l3,alpha):
    theta_pitch,phi_yaw = E[0],E[1]
    return JRcs_(theta_pitch,phi_yaw,theta_L,theta_R,l1,l2,l3,alpha)

############################



def GetEuler(thetaL_corr,thetaR_corr,dataP):
    

    #bnd = [ (np.radians(np.min(pitch_cr)),np.radians(np.max(pitch_cr))),
       #(np.radians(np.min(yaw_cr)),np.radians(np.max(yaw_cr))) ]
    
    bnd = [ (np.radians(-180),np.radians(180)),
       (np.radians(-180),np.radians(180)) ]
    
    LL1,LL2,LL3,ALPHA = dataP
    
    m,M = 0,np.size(thetaL_corr)
    pitch_,yaw_ = np.zeros(np.size(np.arange(m,M,1))),np.zeros(np.size(np.arange(m,M,1)))
    
    guess0 = np.radians([0,0])
    
    for k in np.arange(m,M,1):

        datak = [thetaL_corr[k],thetaR_corr[k],LL1,LL2,LL3,ALPHA]
   
        cns = ({'type': 'ineq', 'fun':fl_cons,'jac':Jac_fl_cons,'args':datak},
              {'type': 'ineq', 'fun':fr_cons,'jac':Jac_fr_cons,'args':datak})

        sol = minimize(SolveFK_min,guess0,args=datak,jac=Jac_F,
                       bounds=bnd,constraints=cns,tol=1e-10)

        pitch,yaw = sol.x[0],sol.x[1]


        pitch_[k],yaw_[k] = pitch,yaw
        
    return pitch_,yaw_

def SolveThetas0(theta_L0,data):
    
    l1,l2,l3,alpha = data
    fL0 = np.sin(alpha)*np.cos(l1) + np.sin(l1)*np.sin(l3)*np.sin(theta_L0)*np.cos(alpha) + np.sin(l1)*np.cos(alpha)*np.cos(l3)*np.cos(theta_L0) - np.cos(l2)
    
    return fL0

def GetThetas0(data):
    
    LL1,LL2,LL3,ALPHA = data
    sol = sc.root(SolveThetas0,0,args=[LL1,LL2,LL3,ALPHA],method='broyden2')
    THETAL0 = math.fmod(sol.x,math.pi)
    THETAR0 = - THETAL0
    
    return THETAL0,THETAR0
        

###################


import matplotlib.colors as colors

def Get_delta(J): 
    m = 2
    M = np.power(LA.det(np.dot(J,J.T)),1/m)
    P = (1/m)*np.trace(np.dot(J,J.T))
    return M/P

def show_contours(theta1cr,theta5cr,delta,xtitle,ytitle,title_,N):
#     #,
#                       norm=colors.BoundaryNorm(boundaries=np.linspace(0,1,25)
#                                                , ncolors=256)
    T1,T5,DELTA = grid(theta1cr,theta5cr,delta, resX=100, resY=100)
    CSS = plt.contourf(T1,T5,DELTA,N,cmap=cm.coolwarm,vmin=.8, vmax=1)

    plt.title(title_)
    plt.colorbar()
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()

    return CSS

def get_contours(theta1cr,theta5cr,delta,N):
    
    T1,T5,DELTA = grid(theta1cr,theta5cr,delta, resX=100, resY=100)
    CS = plt.contourf(T1,T5,DELTA,N,cmap=plt.cm.jet)
    

    return CS

def Get_Area(x,y):
    return np.abs(0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y)))

def GetGlobalIsotropy(cs):
    
    d = cs.levels
    ro, d_areas = np.zeros(np.size(d)), np.zeros(np.size(d))

    for i in range(np.size(d)-1): 
            x= cs.collections[i].get_paths()[0].vertices[:,0]
            y = cs.collections[i].get_paths()[0].vertices[:,1]
            #d_areas[i] = Get_Area(x,y)
            points = np.array([x,y])
            d_areas[i] = ConvexHull(points.T).volume
                
    ro = np.sum(np.multiply(d,d_areas))/np.sum(d_areas)
    
    return ro 

def GetIsotropy(pitch_,yaw_,thetaL_corr,thetaR_corr,data):
    
    LL1,LL2,LL3,ALPHA = data
    m,M = 0,np.size(pitch_)
    delta_ = np.zeros(np.size(np.arange(m,M,1)))

    for k in np.arange(m,M,1):
        Jac = J_(pitch_[k],yaw_[k],thetaL_corr[k],thetaR_corr[k],LL1,LL2,LL3,ALPHA)
        delta_[k] = Get_delta(Jac)
        
    return delta_

def Generate_Input(M,n):
    
    tL,tR =  np.radians(np.linspace(-M,M,n)),np.radians(np.linspace(-M,M,n))
    # -------------------
    TETAL, TETAR = np.meshgrid(tL,tR,indexing='ij')
    tL_,tR_  = np.ravel(TETAL),np.ravel(TETAR)
   
    return np.degrees(tL_),np.degrees(tR_)

def Show_Cartesian_Isotropy(pitch_,yaw_,delta_):


    L = 1 #156.005 #171.245 #Distance from the center to 

    X_ = L*np.cos(pitch_)*np.cos(yaw_)
    Y_ = L*np.sin(yaw_)
    Z_ = -L*(np.sin(pitch_)*np.cos(yaw_))
    
#     fig = plt.figure(3)
#     ax = fig.add_subplot(111, projection='3d')
#     title = ax.set_title('Cartesian Isotropy')
#     ax.set_xlabel('X [mm]')
#     ax.set_ylabel('Y [mm]')
#     ax.set_zlabel('Z [mm]')
#     scatter_ = ax.scatter(X_,Y_,Z_,c=delta_,cmap=cm.coolwarm,vmin=.3, vmax=1)
#     fig.colorbar(scatter_)
#     plt.show()   
    
    return X_,Y_,Z_ 

def ExportTo(filename,pitch_,yaw_,X_,Y_,Z_,delta_):
    
    df = pd.DataFrame(data = {'Pitch [°]':pitch_.T,
                              'Yaw [°]':yaw_.T,
                              'X [mm]':X_.T,
                              'Y [mm]':Y_.T,
                              'Z [mm]':Z_.T,
                              'Delta [-]':delta_.T
                             })
#     df = pd.DataFrame(data = {
#                               'X [mm]':X_.T,
#                               'Y [mm]':Y_.T,
#                               'Z [mm]':Z_.T,
#                              })

    df.to_csv(filename)

    return 

def PlotISO(filename,Input,P):
    
    thetaL_cr,thetaR_cr = Generate_Input(Input,50)
    LL1,LL2,LL3,ALPHA = P[0],P[1],P[2],P[3]
    dataP = (LL1,LL2,LL3,ALPHA)
    N = 50 
    Sj = (2*Input)**2

    THETAL0,THETAR0 = GetThetas0([LL1,LL2,LL3,ALPHA])
    
    print("ThetaL0,ThetaR0 = ",np.degrees([THETAL0,THETAR0]))
    

    iThetaL,iThetaR = np.radians(thetaL_cr), np.radians(thetaR_cr)
    iThetaL_corr,iThetaR_corr = iThetaL+THETAL0 , iThetaR+THETAR0

    time_start = time.clock()

    pitch_,yaw_ = GetEuler(iThetaL_corr,iThetaR_corr,dataP)
     
        

    delta_ = GetIsotropy(pitch_,yaw_,iThetaL_corr,iThetaR_corr,dataP)
    
    
    points = np.array([np.degrees(pitch_),np.degrees(yaw_)])
    WS_Area = ConvexHull(points.T).volume

    print("Gridded Input:",[-Input,Input])
    print("computation done: ",(time.clock() - time_start))

    #%matplotlib notebook 
    plt.figure(1)

    print("For l1 l2 l3 alpha: ",np.degrees([LL1,LL2,LL3,ALPHA]))
    print("\n Min/Max Pitch: ",[np.min(np.degrees(pitch_)),np.max(np.degrees(pitch_))])
    print("\n Min/Max Yaw: ",[np.min(np.degrees(yaw_)),np.max(np.degrees(yaw_))])
    print("\n Min/Max Delta is:",[np.min(delta_),np.max(delta_)])

    csi = show_contours(np.degrees(iThetaL),np.degrees(iThetaR),delta_,
                  r'$\theta_L$ [°]',r'$\theta_R$ [°]',
                  r'$\Delta$ [-] Contour ',N)

    plt.figure(2)
    cso = show_contours(np.degrees(pitch_),np.degrees(yaw_),delta_,              
                  r'$\Theta_p$ [°]',r'$\phi_y$ [°]',
                  r'$\Delta$ [-] Contour ',N)


    print('Global Delta Input',GetGlobalIsotropy(cso))
    print('Workspace euler area and number: ',[WS_Area,WS_Area/(Sj)])    
    print('Residual: ',1-np.min(delta_)*GetGlobalIsotropy(csi)*np.max(delta_)*(WS_Area/(Sj))) 

    X_,Y_,Z_  = Show_Cartesian_Isotropy(pitch_,yaw_,delta_)
    points = np.array([X_,Y_,Z_])
    hull = ConvexHull(points.T)
    print("Workspace cartesian area: ",hull.area)
    
#     ExportTo(filename+str(Input),np.degrees(pitch_),np.degrees(yaw_),X_,Y_,Z_,delta_)
    
    return [np.min(delta_),GetGlobalIsotropy(csi),np.max(delta_),WS_Area/(Sj)]



