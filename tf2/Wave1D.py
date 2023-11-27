#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script for wave equation in 1D with Neumann boundary conditions at left end
Governing equation: u_tt(x,t) = u_xx(x,t) for (x,t) \in \Omegax(0,T)
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
@author: cosmin
"""
import tensorflow as tf
import numpy as np
import time
import scipy.optimize
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.scipy_loss import scipy_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Wave1D
from utils.Plotting import plot_pts, plot_convergence_semilog

np.random.seed(42)
tf.random.set_seed(42)

    
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

    
#define the input and output data set
xmin = 0
xmax = 4
tmin = 0
tmax = 2
domainCorners = np.array([[xmin,tmin], [xmin,tmax], [xmax,tmin], [xmax,tmax]])
domainGeom = Quadrilateral(domainCorners)

numPtsU = 101
numPtsV = 101
xPhys, yPhys = domainGeom.getUnifIntPts(numPtsU,numPtsV,[0,0,0,0])
data_type = "float32"

Xint = np.concatenate((xPhys,yPhys),axis=1).astype(data_type)
Yint = np.zeros_like(xPhys).astype(data_type)

#boundary conditions at x=0
xPhysBnd, tPhysBnd, _, _ = domainGeom.getUnifEdgePts(numPtsU, numPtsV, [0,0,0,1])
Xbnd = np.concatenate((xPhysBnd, tPhysBnd), axis=1).astype(data_type)
Ybnd = np.where(tPhysBnd<=1, -np.sin(np.pi*tPhysBnd), 0).astype(data_type)

#initial conditions (displacement and velocity) for t=0
xPhysInit, tPhysInit, _, _ = domainGeom.getUnifEdgePts(numPtsU, numPtsV, [1,0,0,0])
Xinit = np.concatenate((xPhysInit, tPhysInit), axis=1).astype(data_type)
Yinit = np.zeros_like(Xinit)


#plot the collocation points
plot_pts(Xint, Xbnd)

#define the model 
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(30, "tanh")
l2 = tf.keras.layers.Dense(30, "tanh")
l3 = tf.keras.layers.Dense(30, "tanh")
#l4 = tf.keras.layers.Dense(20, "tanh")
#l5 = tf.keras.layers.Dense(20, "tanh")
l4 = tf.keras.layers.Dense(1, None)
train_op = tf.keras.optimizers.Adam()
train_op2 = "BFGS-B"
num_epoch = 1000
print_epoch = 100
pred_model = Wave1D([l1, l2, l3, l4], train_op, num_epoch, print_epoch)

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Yint_tf = tf.convert_to_tensor(Yint)
Xbnd_tf = tf.convert_to_tensor(Xbnd)
Ybnd_tf = tf.convert_to_tensor(Ybnd)
Xinit_tf = tf.convert_to_tensor(Xinit)
Yinit_tf = tf.convert_to_tensor(Yinit)

#training
print("Training (ADAM)...")
t0 = time.time()
pred_model.network_learn(Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf, Xinit_tf, Yinit_tf)
t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")

if train_op2=="SciPy-LBFGS-B":
    print("Training (SciPy-LBFGS-B)...")
    loss_func = scipy_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
    init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables).numpy()
    results = scipy.optimize.minimize(fun=loss_func, x0=init_params, jac=True, method='L-BFGS-B',
                options={'disp': None, 'maxls': 50, 'iprint': -1, 
                'gtol': 1e-6, 'eps': 1e-6, 'maxiter': 50000, 'ftol': 1e-6, 
                'maxcor': 50, 'maxfun': 50000})
    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    loss_func.assign_new_model_parameters(results.x)
else:            
    print("Training (TFP-BFGS)...")

    loss_func = tfp_function_factory(pred_model, 
                                     Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf, Xinit_tf, Yinit_tf)
    # convert initial model parameters to a 1D tf.Tensor
    init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)
    # train the model with L-BFGS solver
    results = tfp.optimizer.bfgs_minimize(
        value_and_gradients_function=loss_func, initial_position=init_params,
              max_iterations=50000, tolerance=1e-14)#num_correction_pairs=100, tolerance=1e-14)  
    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    loss_func.assign_new_model_parameters(results.position)    
t2 = time.time()
print("Time taken (LBFGS)", t2-t1, "seconds")
print("Time taken (all)", t2-t0, "seconds")
print("Testing...")
numPtsUTest = 2*numPtsU
numPtsVTest = 2*numPtsV
xPhysTest, tPhysTest = domainGeom.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
XTest = np.concatenate((xPhysTest,tPhysTest),axis=1).astype(data_type)
XTest_tf = tf.convert_to_tensor(XTest)
uTest = pred_model(XTest_tf).numpy()    
_, vTest = pred_model.du(XTest_tf[:,0:1], XTest_tf[:,1:2])
vTest = vTest.numpy()
uExact, vExact = compExactDispVel(XTest)

xPhysTest2D = np.resize(XTest[:,0], [numPtsUTest, numPtsVTest])
yPhysTest2D = np.resize(XTest[:,1], [numPtsUTest, numPtsVTest])
uExact2D = np.resize(uExact, [numPtsUTest, numPtsVTest])
vExact2D = np.resize(vExact, [numPtsUTest, numPtsVTest])
uTest2D = np.resize(uTest, [numPtsUTest, numPtsVTest])
vTest2D = np.resize(vTest, [numPtsUTest, numPtsVTest])
plt.contourf(xPhysTest2D, yPhysTest2D, uExact2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("Exact displacement")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, uTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("Computed displacement")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, uExact2D-uTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("Error: U_exact-U_computed")
plt.show()

plt.contourf(xPhysTest2D, yPhysTest2D, vExact2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("Exact velocity")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, vTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("Computed velocity")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, vExact2D-vTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("Error: V_exact-V_computed")
plt.show()

errDisp = uExact - uTest
print("L2-error norm for displacement: {}".format(np.linalg.norm(errDisp)/np.linalg.norm(uExact)))

errVel = vExact - vTest
print("L2-error norm for velocity: {}".format(np.linalg.norm(errVel)/np.linalg.norm(vExact)))

# plot the loss convergence
plot_convergence_semilog(pred_model.adam_loss_hist, loss_func.history)

   