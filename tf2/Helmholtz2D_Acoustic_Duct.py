#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement a Helmholtz 2D problem for the acoustic duct:
    Here we solve for both the real and imaginary part
    \Delta w(x,y) +k^2w(x,y) = 0 for (x,y) \in \Omega:= (0,2)x(0,1)
with Neumann and Robin boundary conditions
    \partial u / \partial n = cos(m*pi*x), for x = 0;
    \partial u / \partial n = -iku, for x = 2; 
    \partial u / \partial n = 0, for y=0 and y=1
        
    Exact solution: u(x,y) = cos(m*pi*y)*(A_1*exp(-i*k_x*x) + A_2*exp(i*k_x*x))
    where A_1 and A_2 are obtained by solving the 2x2 linear system:
    [i*k_x                               -i*k_x          ] [A_1]  = [1]
    [i*(k-k_x)*exp(-2*i*k_x)       i*(k+k_x)*exp(2*i*k_x)] [A_2]    [0]
        
"""
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Helmholtz2D_coll
from utils.Plotting import plot_convergence_semilog

tf.random.set_seed(42)
 
#define model parameters
m = 1 #mode number
k = 4 #wave number
alpha = 1j*k # coefficient for Robin boundary conditions of the form 
             #   du/dn + alpha*u = g

#solve for the constants A1 and A2 in the exact solution
kx = np.sqrt(k ** 2 - (m * np.pi) ** 2)
LHS = np.array(
    [[1j*kx , -1j * kx], [1j*(k - kx) * np.exp(-2j * kx), 1j*(k + kx) * np.exp(2j * kx)]]
)
RHS = np.array([1.0, 0.0])
A = np.linalg.solve(LHS, RHS)


# The exact solution for error norm computations
def exact_sol(x, y):
    return np.cos(m * np.pi * y) * (
        A[0] * np.exp(-1j * kx * x) + A[1] * np.exp(1j * kx * x)
    )

def deriv_exact_sol(x, y):
    return [
        np.cos(m * np.pi * y)
        * (
            A[0] * (-1j) * kx * np.exp(-1j * kx * x)
            + A[1] * 1j * kx * np.exp(1j * kx * x)
        ),
        -np.sin(m * np.pi * y)
        * (m * np.pi)
        * (A[0] * np.exp(-1j * kx * x) + A[1] * np.exp(1j * kx * x)),
    ]

# Define the boundary conditions
def u_bound_left(x, y):
    return np.cos(m * np.pi * y)


def u_bound_right(x, y):
    return 0.0

def u_bound_up_down(x,y):
    return np.zeros_like(x)

    
#define the input and output data set
xmin = 0
xmax = 2
ymin = 0
ymax = 1
domainCorners = np.array([[xmin,ymin], [xmin,ymax], [xmax,ymin], [xmax,ymax]])
myQuad = Quadrilateral(domainCorners)

numPtsU = 28
numPtsV = 28
xPhys, yPhys = myQuad.getUnifIntPts(numPtsU, numPtsV, [0,0,0,0])
data_type = "float64"

Xint = np.concatenate((xPhys,yPhys),axis=1).astype(data_type)
Yint = np.zeros_like(Xint)

# Generate the training data for the Neumann boundary
xPhysNeu, yPhysNeu, xNormNeu, yNormNeu = myQuad.getUnifEdgePts(numPtsU, numPtsV, [1,0,1,1])
XbndNeu = np.concatenate((xPhysNeu, yPhysNeu, xNormNeu, yNormNeu), axis=1).astype(data_type)
dwdx_neu, dwdy_neu = deriv_exact_sol(xPhysNeu, yPhysNeu)

Ybnd_neu_real = np.real(dwdx_neu*XbndNeu[:, 2:3] + dwdy_neu*XbndNeu[:, 3:4])
Ybnd_neu_imag = np.imag(dwdx_neu*XbndNeu[:, 2:3] + dwdy_neu*XbndNeu[:, 3:4])
YbndNeu = np.concatenate((Ybnd_neu_real, Ybnd_neu_imag), axis=1).astype(data_type)

# Generating the training data for the Robin boundary
xPhysRobin, yPhysRobin, xNormRobin, yNormRobin = myQuad.getUnifEdgePts(numPtsU, numPtsV, [0,1,0,0])
XbndRobin = np.concatenate((xPhysRobin, yPhysRobin, xNormRobin, yNormRobin), axis=1).astype(data_type)
Ybnd_robin_real = u_bound_up_down(xPhysRobin, yPhysRobin)
Ybnd_robin_imag = u_bound_up_down(xPhysRobin, yPhysRobin)
YbndRobin = np.concatenate((Ybnd_robin_real, Ybnd_robin_imag), axis=1).astype(data_type)

#define the model 
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(20, "tanh")
l2 = tf.keras.layers.Dense(20, "tanh")
l3 = tf.keras.layers.Dense(20, "tanh")
l4 = tf.keras.layers.Dense(2, None)
train_op = tf.keras.optimizers.Adam()
num_epoch = 5000
print_epoch = 100
alpha_real = np.real([alpha]).astype(data_type)[0]
alpha_imag = np.imag([alpha]).astype(data_type)[0]
pred_model = Helmholtz2D_coll([l1, l2, l3, l4], train_op, num_epoch,
                                    print_epoch, k, alpha_real, alpha_imag)

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Yint_tf = tf.convert_to_tensor(Yint)
XbndNeu_tf = tf.convert_to_tensor(XbndNeu)
YbndNeu_tf = tf.convert_to_tensor(YbndNeu)
XbndRobin_tf = tf.convert_to_tensor(XbndRobin)
YbndRobin_tf = tf.convert_to_tensor(YbndRobin)

#training
print("Training (ADAM)...")
t0 = time.time()
pred_model.network_learn(Xint_tf, Yint_tf, XbndNeu_tf, YbndNeu_tf, XbndRobin_tf, YbndRobin_tf)
t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")
print("Training (BFGS)...")

loss_func = tfp_function_factory(pred_model, Xint_tf, Yint_tf, 
                                 XbndNeu_tf, YbndNeu_tf, XbndRobin_tf, YbndRobin_tf)
#loss_func = scipy_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)
# train the model with BFGS solver
results = tfp.optimizer.bfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
          max_iterations=3000, tolerance=1e-14)

# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
loss_func.assign_new_model_parameters(results.position)
#loss_func.assign_new_model_parameters(results.x)
t2 = time.time()
print("Time taken (BFGS)", t2-t1, "seconds")
print("Time taken (all)", t2-t0, "seconds")
print("Testing...")
numPtsUTest = 2*numPtsU
numPtsVTest = 2*numPtsV
xPhysTest, yPhysTest = myQuad.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
XTest = np.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
XTest_tf = tf.convert_to_tensor(XTest)
YTest = pred_model(XTest_tf).numpy()
YTest_real = YTest[:, 0:1]
YTest_imag = YTest[:, 1:2]
YExact = exact_sol(XTest[:,[0]], XTest[:,[1]])
YExact_real = np.real(YExact)
YExact_imag = np.imag(YExact)

xPhysTest2D = np.resize(XTest[:,0], [numPtsUTest, numPtsVTest])
yPhysTest2D = np.resize(XTest[:,1], [numPtsUTest, numPtsVTest])
YExact2D_real = np.resize(YExact_real, [numPtsUTest, numPtsVTest])
YExact2D_imag = np.resize(YExact_imag, [numPtsUTest, numPtsVTest])

YTest2D_real = np.resize(YTest_real, [numPtsUTest, numPtsVTest])
YTest2D_imag = np.resize(YTest_imag, [numPtsUTest, numPtsVTest])

# Plot the real part of the solution and errors
plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_real, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Exact solution (real)")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D_real, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed solution (real)")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_real-YTest2D_real, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Error: U_exact-U_computed (real)")
plt.show()

# Plot the imaginary part of the solution and errors
plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_imag, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Exact solution (imag)")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D_imag, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed solution (imag)")
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_imag-YTest2D_imag, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Error: U_exact-U_computed (imag)")
plt.show()

# Compute relative errors
err_real = YExact_real - YTest_real
err_imag = YExact_imag - YTest_imag
print("L2-error norm (real): {}".format(np.linalg.norm(err_real)/np.linalg.norm(YTest_real)))
print("L2-error norm (imag): {}".format(np.linalg.norm(err_imag)/np.linalg.norm(YTest_imag)))

# plot the loss convergence
plot_convergence_semilog(pred_model.adam_loss_hist, loss_func.history)
   