#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson equation example
Solve the equation -\Delta u(x) = f(x) for x\in\Omega 
with Dirichlet boundary conditions u(x)=u0 for x\in\partial\Omega
For this example
    u(x,y) = 0, for (x,y) \in \partial \Omega, where \Omega:=[0,1]x[0,1]
    Exact solution: u(x,y) = x(1-x)y(1-y) corresponding to 
    f(x,y) = 2(x-x^2+y-y^2)   (defined in rhs_fun(x,y))
Implement Deep Energy Method
"""
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Poisson2D_DEM
from utils.Plotting import plot_convergence_dem

tf.random.set_seed(42)

    
#define the RHS function f(x)
def rhs_fun(x,y):
    f = 2*(x-x**2+y-y**2)
    return f

def exact_sol(x,y):
    u = x*(1-x)*y*(1-y)
    return u

def deriv_exact_sol(x,y):
    dudx = (1-2*x)*(y-y**2)
    dudy = (x-x**2)*(1-2*y)
    return dudx, dudy
    
#define the input and output data set
xmin = 0
xmax = 1
ymin = 0
ymax = 1
domainCorners = np.array([[xmin,ymin], [xmin,ymax], [xmax,ymin], [xmax,ymax]])
myQuad = Quadrilateral(domainCorners)

numPtsU = 80
numPtsV = 80
numElemU = 20
numElemV = 20
numGauss = 4
boundary_weight = 1e4

xPhys, yPhys, Wint = myQuad.getQuadIntPts(numElemU, numElemV, numGauss)
data_type = "float64"

Xint = np.concatenate((xPhys,yPhys),axis=1).astype(data_type)
Wint = Wint.astype(data_type)
Yint = rhs_fun(Xint[:,[0]], Xint[:,[1]])

xPhysBnd, yPhysBnd, _, _ = myQuad.getUnifEdgePts(numPtsU, numPtsV, [1,1,1,1])
Xbnd = np.concatenate((xPhysBnd, yPhysBnd), axis=1).astype(data_type)
Ybnd = exact_sol(Xbnd[:,[0]], Xbnd[:,[1]])
Wbnd = boundary_weight*np.ones_like(Ybnd).astype(data_type)


#plot the boundary and interior points
plt.scatter(Xint[:,0], Xint[:,1], s=0.5)
plt.scatter(Xbnd[:,0], Xbnd[:,1], s=1, c='red')
plt.title("Boundary collocation and interior integration points")
plt.show()

#define the model 
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(20, "tanh")
l2 = tf.keras.layers.Dense(20, "tanh")
l3 = tf.keras.layers.Dense(20, "tanh")
l4 = tf.keras.layers.Dense(1, None)
train_op = tf.keras.optimizers.Adam()
num_epoch = 1000
print_epoch = 100
pred_model = Poisson2D_DEM([l1, l2, l3, l4], train_op, num_epoch, print_epoch)

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Yint_tf = tf.convert_to_tensor(Yint)
Wint_tf = tf.convert_to_tensor(Wint)
Xbnd_tf = tf.convert_to_tensor(Xbnd)
Wbnd_tf = tf.convert_to_tensor(Wbnd)
Ybnd_tf = tf.convert_to_tensor(Ybnd)

#training
print("Training (ADAM)...")
t0 = time.time()
pred_model.network_learn(Xint_tf, Wint_tf, Yint_tf, Xbnd_tf, Wbnd_tf, Ybnd_tf)
t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")
print("Training (BFGS)...")

loss_func = tfp_function_factory(pred_model, Xint_tf, Wint_tf, Yint_tf, 
                                 Xbnd_tf, Wbnd_tf, Ybnd_tf)
#loss_func = scipy_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)#.numpy()
# train the model with BFGS solver
results = tfp.optimizer.bfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
          max_iterations=1000, tolerance=1e-14)  

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
print("Relative L2-error norm: {}".format(np.linalg.norm(err)/np.linalg.norm(YTest)))

# Compute the error with integration
xPhys, yPhys, Wint = myQuad.getQuadIntPts(2*numElemU, 2*numElemV, 2*numGauss)
Xint_test = np.concatenate((xPhys,yPhys),axis=1).astype(data_type)
Yint_test = pred_model(Xint_test)
dYint_dx_test, dYint_dy_test = pred_model.du(Xint_test[:,0:1], Xint_test[:,1:2])
Yint_exact = exact_sol(xPhys, yPhys)
dYint_dx_exact, dYint_dy_exact = deriv_exact_sol(xPhys, yPhys)
rel_l2_err = np.sqrt(np.sum(((Yint_exact-Yint_test)**2*Wint))/np.sum(Yint_exact**2*Wint))
dYint_dx_err = dYint_dx_exact - dYint_dx_test
dYint_dy_err = dYint_dy_exact - dYint_dy_test
h1_err = np.sum((dYint_dx_err**2 + dYint_dy_err**2)*Wint)
h1_norm = np.sum((dYint_dx_exact**2 + dYint_dy_exact**2)*Wint)
rel_h1_err = np.sqrt(h1_err/h1_norm)                
print("Relative L2-error norm (integration): ", rel_l2_err)
print("Relative H1-error norm (integration): ", rel_h1_err)

# plot the loss convergence
plot_convergence_dem(pred_model.adam_loss_hist, loss_func.history, percentile=95.)


   