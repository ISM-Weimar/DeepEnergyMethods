#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson equation example
Solve the equation -\Delta u(x) = f(x) for x\in\Omega with Dirichlet boundary conditions u(x)=u0 for x\in\partial\Omega
@author: cosmin
"""
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Poisson2D_coll
from utils.Plotting import plot_convergence_semilog

tf.random.set_seed(42)
 


#define the RHS function f(x)
kx = 1
ky = 1
def rhs_fun(x,y):
    f = (kx**2+ky**2)*np.pi**2*np.sin(kx*np.pi*x)*np.sin(ky*np.pi*y)
    return f

def exact_sol(x,y):
    u = np.sin(kx*np.pi*x)*np.sin(ky*np.pi*y)
    return u

def deriv_exact_sol(x,y):
    du = kx*np.pi*np.cos(kx*np.pi*x)*np.sin(ky*np.pi*y)
    dv = ky*np.pi*np.sin(kx*np.pi*x)*np.cos(ky*np.pi*y)
    return du, dv

#define the input and output data set
xmin = 0
xmax = 1
ymin = 0
ymax = 1
domainCorners = np.array([[xmin,ymin], [xmin,ymax], [xmax,ymin], [xmax,ymax]])
myQuad = Quadrilateral(domainCorners)

numPtsU = 28
numPtsV = 28
xPhys, yPhys = myQuad.getUnifIntPts(numPtsU, numPtsV, [0,0,0,0])
data_type = "float64"

Xint = np.concatenate((xPhys,yPhys),axis=1).astype(data_type)
Yint = rhs_fun(Xint[:,[0]], Xint[:,[1]])

xPhysBnd, yPhysBnd, _, _s = myQuad.getUnifEdgePts(numPtsU, numPtsV, [1,1,1,1])
Xbnd = np.concatenate((xPhysBnd, yPhysBnd), axis=1).astype(data_type)
Ybnd = exact_sol(Xbnd[:,[0]], Xbnd[:,[1]])

#define the model 
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(20, "tanh")
l2 = tf.keras.layers.Dense(20, "tanh")
l3 = tf.keras.layers.Dense(20, "tanh")
l4 = tf.keras.layers.Dense(1, None)
train_op = tf.keras.optimizers.Adam()
num_epoch = 1000
print_epoch = 100
pred_model = Poisson2D_coll([l1, l2, l3, l4], train_op, num_epoch, print_epoch)

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Yint_tf = tf.convert_to_tensor(Yint)
Xbnd_tf = tf.convert_to_tensor(Xbnd)
Ybnd_tf = tf.convert_to_tensor(Ybnd)

#training
print("Training (ADAM)...")
t0 = time.time()
pred_model.network_learn(Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")
print("Training (LBFGS)...")

loss_func = tfp_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
#loss_func = scipy_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)#.numpy()
# train the model with BFGS solver
results = tfp.optimizer.bfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
          max_iterations=1000, tolerance=1e-14)  
# results = scipy.optimize.minimize(fun=loss_func, x0=init_params, jac=True, method='L-BFGS-B',
#                 options={'disp': None, 'maxls': 50, 'iprint': -1, 
#                 'gtol': 1e-12, 'eps': 1e-12, 'maxiter': 50000, 'ftol': 1e-12, 
#                 'maxcor': 50, 'maxfun': 50000})
# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
loss_func.assign_new_model_parameters(results.position)
#loss_func.assign_new_model_parameters(results.x)
t2 = time.time()
print("Time taken (LBFGS)", t2-t1, "seconds")
print("Time taken (all)", t2-t0, "seconds")
print("Testing...")
numPtsUTest = 2*numPtsU
numPtsVTest = 2*numPtsV
xPhysTest, yPhysTest = myQuad.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
XTest = np.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
XTest_tf = tf.convert_to_tensor(XTest)
YTest = pred_model(XTest_tf).numpy()    
YExact = exact_sol(XTest[:,[0]], XTest[:,[1]])

xPhysTest2D = np.resize(XTest[:,0], [numPtsUTest, numPtsVTest])
yPhysTest2D = np.resize(XTest[:,1], [numPtsUTest, numPtsVTest])
YExact2D = np.resize(YExact, [numPtsUTest, numPtsVTest])
YTest2D = np.resize(YTest, [numPtsUTest, numPtsVTest])
plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Exact solution")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed solution")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D-YTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Error: U_exact-U_computed")
plt.show()

err = YExact - YTest
print("L2-error norm: {}".format(np.linalg.norm(err)/np.linalg.norm(YTest)))

# plot the loss convergence
plot_convergence_semilog(pred_model.adam_loss_hist, loss_func.history)
