#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson equation example
Solve the equation -u''(x) = f(x) for x\in(a,b) with Dirichlet boundary conditions u(a)=u0, u(b)=1
Implementation with Deep Energy Method with adaptive activation function 
(see https://doi.org/10.1016/j.jcp.2019.109136 )
"""
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.Plotting import plot_convergence_dem


tf.random.set_seed(42)

class model(tf.keras.Model): 
    def __init__(self, train_op, num_epoch, print_epoch, data_type):
        super(model, self).__init__()
        l1 = tf.keras.layers.Dense(64, self.adaptive_tanh)
        l2 = tf.keras.layers.Dense(64, self.adaptive_tanh)
        l3 = tf.keras.layers.Dense(1, None)
        self.model_layers = [l1, l2, l3]
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.a = tf.Variable(0.1, dtype=data_type)
        self.adam_loss_hist = []
    
    def adaptive_tanh(self, x):
        return tf.math.tanh(10*self.a*x)
        
    def call(self, X):
        return self.u(X)
    
    
    
    # Running the model
    def u(self,X):
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X
    
    # Return the first derivative
    def du(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            u_val = self.u(X)
        du_val = tape.gradient(u_val, X)
        return du_val    
    
    @tf.function
    def get_loss(self,Xint, Yint, Wint, Xbnd, Ybnd):
        int_loss, bnd_loss_dir = self.get_all_losses(Xint, Yint, Wint, Xbnd, Ybnd)
        return int_loss+1e3*bnd_loss_dir
         
    #Custom loss function
    def get_all_losses(self,Xint, Yint, Wint, XbndDir, YbndDir):
        u_val_bnd_dir = self.u(XbndDir)        
        u_val_int = self.u(Xint)
        du_val_int = self.du(Xint)
        
        int_loss = tf.reduce_sum((1/2*du_val_int**2 - u_val_int*Yint)*Wint)
        bnd_loss_dir = tf.reduce_mean(tf.math.square(u_val_bnd_dir - YbndDir))        
        return int_loss, bnd_loss_dir
      
    # get gradients
    def get_grad(self, Xint, Yint, Wint, XbndDir, YbndDir):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Wint, XbndDir, YbndDir)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self,Xint,Yint, Wint, XbndDir, YbndDir):
        self.bounds = {"lb" : tf.math.reduce_min(Xint),
                       "ub" : tf.math.reduce_max(Xint)}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Wint, XbndDir, YbndDir)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                int_loss, bnd_loss_dir = self.get_all_losses(Xint, Yint, Wint, Xbnd, Ybnd)
                L = int_loss + bnd_loss_dir
                print("Epoch {} loss: {}, int_loss_x: {}, bnd_loss_dir: {}, a: {}".format(i, 
                                                        L, int_loss, bnd_loss_dir, self.a.numpy()))                

def generate_quad_pts_weights_1d(x_min=0, x_max=1, num_elem=10, num_gauss_pts=4):
    """
    Generates the Gauss points and weights on a 1D interval (x_min, x_max), split
    into num_elem equal-length subintervals, each with num_gauss_pts quadature
    points per element.
    
    Note: the sum of the weights should equal to the length of the domain

    Parameters
    ----------
    x_min : (scalar)
        lower bound of the 1D domain.
    x_max : (scalar)
        upper bound of the 1D domain.
    num_elem : (integer)
        number of subdivision intervals or elements.
    num_gauss_pts : (integer)
        number of Gauss points in each element

    Returns
    -------
    pts : (1D array)
        coordinates of the integration points.
    weights : (1D array)
        weights corresponding to each point.

    """
    x_pts = np.linspace(x_min, x_max, num=num_elem+1)
    pts = np.zeros(num_elem*num_gauss_pts)
    weights = np.zeros(num_elem*num_gauss_pts)
    pts_ref, weights_ref = np.polynomial.legendre.leggauss(num_gauss_pts)
    for i in range(num_elem):
        x_min_int = x_pts[i]
        x_max_int = x_pts[i+1]        
        jacob_int = (x_max_int-x_min_int)/2
        pts_int = jacob_int*pts_ref + (x_max_int+x_min_int)/2
        weights_int = jacob_int * weights_ref
        pts[i*num_gauss_pts:(i+1)*num_gauss_pts] = pts_int
        weights[i*num_gauss_pts:(i+1)*num_gauss_pts] = weights_int        
        
    return pts, weights

 

#define the RHS function f(x)
k = 4
def rhs_fun(x):
    f = k**2*np.pi**2*np.sin(k*np.pi*x)
    return f

def exact_sol(x):
    y = np.sin(k*np.pi*x)
    return y

def deriv_exact_sol(x):
    output = k*np.pi*np.cos(k*np.pi*x)    
    return output

#define the input and output data set
xmin = 0
xmax = 1
data_type = "float64"

Xint, Wint = generate_quad_pts_weights_1d(x_min=xmin, x_max=xmax, num_elem=50, num_gauss_pts=4)
Xint = np.array(Xint)[np.newaxis].T.astype(data_type)
Yint = rhs_fun(Xint)
Wint = np.array(Wint)[np.newaxis].T.astype(data_type)

Xbnd = np.array([[xmin],[xmax]]).astype(data_type)
Ybnd = exact_sol(Xbnd)


#define the model 
tf.keras.backend.set_floatx(data_type)

train_op = tf.keras.optimizers.Adam()
num_epoch = 3000
print_epoch = 100
pred_model = model(train_op, num_epoch, print_epoch, data_type)

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Yint_tf = tf.convert_to_tensor(Yint)
Wint_tf = tf.convert_to_tensor(Wint)
XbndDir_tf = tf.convert_to_tensor(Xbnd)
YbndDir_tf = tf.convert_to_tensor(Ybnd)

#training
print("Training (ADAM)...")
t0 = time.time()
pred_model.network_learn(Xint_tf, Yint_tf, Wint_tf, XbndDir_tf, YbndDir_tf)
t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")
print("Training (LBFGS)...")

loss_func = tfp_function_factory(pred_model, Xint_tf, Yint_tf, Wint_tf, XbndDir_tf, 
                                  YbndDir_tf)
# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)
# train the model with BFGS solver
results = tfp.optimizer.bfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
          max_iterations=300, tolerance=1e-14)  
# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
loss_func.assign_new_model_parameters(results.position)
#loss_func.assign_new_model_parameters(results.x)
t2 = time.time()
print("Time taken (BFGS)", t2-t1, "seconds")
print("Time taken (all)", t2-t0, "seconds")
print("Testing...")
numPtsTest = 2*len(Xint)
x_test = np.linspace(xmin, xmax, numPtsTest).astype(data_type)
x_test = np.array(x_test)[np.newaxis].T
x_tf = tf.convert_to_tensor(x_test)

y_test = pred_model.u(x_tf)    
y_exact = exact_sol(x_test)
dy_test = pred_model.du(x_tf)


plt.plot(x_test, y_test, label='Predicted')
plt.plot(x_test, y_exact, label='Exact')
plt.legend()
plt.show()
plt.plot(x_test, y_exact-y_test)
plt.title("Error")
plt.show()
err = y_exact - y_test
print("L2-error norm: {}".format(np.linalg.norm(err)/np.linalg.norm(y_exact)))

dy_exact = deriv_exact_sol(x_test)
plt.plot(x_test, dy_test, label='Predicted')
plt.plot(x_test, dy_exact, label='Exact')
plt.legend()
plt.title("First derivative")
plt.show()

err_deriv = dy_exact - dy_test
plt.plot(x_test, err_deriv)
plt.title("Error for first derivative")
plt.show
print("Relative error for first derivative: {}".format(np.linalg.norm(err_deriv)/np.linalg.norm(dy_exact)))

# plot the loss convergence
plot_convergence_dem(pred_model.adam_loss_hist, loss_func.history, percentile=95.)