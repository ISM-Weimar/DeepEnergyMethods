"""
script for wave equation in 1D with Neumann boundary conditions at left end
Governing equation: u_tt(x,t) = u_xx(x,t) for x \in (0,1), u(0)=u(1)=0
Exact solution u(x,t) = 2\alpha/pi for x<\alpha*(t-1)
                      = \alpha/pi*[1-cos(pi*(t-x/alpha))], for \alpha(t-1)<=x<=alpha*t
                      = 0 for \alpha*t < x
which satisfies the inital and Neumann boundary conditions:
    u(x,0) = 0
    u_t(x,0) = 0
    u_x(0,t) = -sin(pi*t) for 0<=t<=1
             = 0 for 1<t
(Example pages 89-92 in Bedford - Introduction to Elastic Wave Propagation
 https://www.researchgate.net/publication/269575415_Introduction_to_Elastic_Wave_Propagation)
"""

import tensorflow as tf
import numpy as np
import time
import os
from utils.PINN_wave import WaveEquation
from utils.gridPlot import plotConvergence
from utils.gridPlot import plot1d
from utils.gridPlot import createFolder
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def compExactDispVel(XT):
    # Computes the exact solution for Bedford beam
    alpha = 1
    numData = np.shape(XT)[0]
    u = np.zeros([numData,1])
    v = np.zeros([numData,1])
    for i in range(numData):
        x = XT[i,0]
        t = XT[i,1]
        if x<alpha*(t-1):
            u[i] = 2*alpha/np.pi
            v[i] = 0
        elif x<=alpha*t:
            u[i] = alpha/np.pi * (1-np.cos(np.pi*(t-x/alpha)))
            v[i] = alpha/np.pi * np.pi * np.sin(np.pi*(t-x/alpha))
    return u, v

#Main program
figHeight = 5
figWidth = 5

originalDir = os.getcwd()
foldername = 'Wave1D_results'    
createFolder('./'+ foldername + '/')
os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
         
T = 2.
x_min = 0
x_max = 4

# Domain bounds
lb = np.array([x_min, 0.0])  #left boundary (x=x_min, t=0)
rb = np.array([x_max, T])  #right boundary (x=x_max, t=T)

layers = [2, 30, 30, 30, 1]                
    
# generate the interior collocation points
numTimePts = 201
numSpacePts = 201
t_int = np.linspace(0, T, numTimePts)
x_pts = np.linspace(x_min, x_max, numSpacePts)

#trim endpoints
t_int = t_int[1:]
x_int = x_pts[1:-1]

#generate grid of points
[xGrid, tGrid] = np.meshgrid(x_int, t_int)
xGrid = np.ndarray.flatten(xGrid)[np.newaxis]
tGrid = np.ndarray.flatten(tGrid)[np.newaxis]
X_f = np.concatenate((xGrid.T, tGrid.T), axis=1)
    
# generate the boundary collocation points for x=x_min, x=x_max   
xValLeft = np.zeros_like(t_int)
uxValLeft = np.where(t_int<=1, -np.sin(np.pi*t_int), 0)
xValRight = np.ones_like(t_int)
uValRight = np.zeros_like(t_int)

X_b_left = np.concatenate((xValLeft[np.newaxis].T, t_int[np.newaxis].T,
                           uxValLeft[np.newaxis].T), axis=1)


X_b_right = np.concatenate((xValRight[np.newaxis].T, t_int[np.newaxis].T, 
                            uValRight[np.newaxis].T), axis=1)

# Generate the initial displacement and velocity boundary conditions
tVal = np.zeros_like(x_pts)
uVal = np.zeros_like(x_pts) 
vVal = np.zeros_like(x_pts)
X_init = np.concatenate((x_pts[np.newaxis].T, tVal[np.newaxis].T, uVal[np.newaxis].T, 
                        vVal[np.newaxis].T), axis=1)

model = WaveEquation(lb, rb, X_f, X_b_left, X_init, layers)        
start_time = time.time()      
num_train_its = 2000
model.train(num_train_its)
elapsed = time.time() - start_time                
print('Training time: %.4f' % (elapsed))

# Training data
nPred = 1001
TPred = T
X_star = np.linspace(x_min, x_max, nPred)[np.newaxis]
X_star = X_star.T
T_star = TPred * np.ones_like(X_star)

XT_star = np.concatenate((X_star, T_star), axis=1)
u_pred, v_pred = model.predict(XT_star)
u_exact, v_exact = compExactDispVel(XT_star)
u_pred_err = u_exact-u_pred
v_pred_err = v_exact-v_pred

error_u = np.linalg.norm(u_exact-u_pred,2)/np.linalg.norm(u_exact,2)
error_v = np.linalg.norm(v_exact-v_pred,2)/np.linalg.norm(v_exact,2)
print('Relative error u: %e' % (error_u))       
print('Relative error v: %e' % (error_v))       

# Plot results
plot1d(u_pred_err,X_star,u_pred,u_exact,v_pred,v_exact,v_pred_err,figHeight,figWidth)

adam_buff = model.loss_adam_buff
lbfgs_buff = model.lbfgs_buffer        
plotConvergence(num_train_its,adam_buff,lbfgs_buff,figHeight,figWidth)    
    

    