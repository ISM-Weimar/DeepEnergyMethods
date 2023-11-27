#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation example
Inpterpolates the function given by exact_sol(x) at a discrete set of points
Created on Fri Sep 11 15:57:07 2020

@author: cosmin
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.Plotting import plot_convergence_semilog

tf.random.set_seed(42)

class model(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(model, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.adam_loss_hist = []
    
    def call(self, X):
        return self.u(X)
    
    # Running the model
    def u(self,X):
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
         
    #Custom loss function
    def get_loss(self,Xint, Yint):
        u_val_int=self.u(Xint)
        int_loss = tf.reduce_mean(tf.math.square(u_val_int - Yint))
        return int_loss
      
    # get gradients
    def get_grad(self, Xint, Yint):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self,Xint,Yint):
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))


    
#define the function ot be interpolated
k = 4
def exact_sol(input):
    output = np.sin(k*np.pi*input)
    return output

#define the input and output data set
xmin = -1
xmax = 1
numPts = 201
data_type = "float64"

Xint = np.linspace(xmin, xmax, numPts).astype(data_type)
Xint = np.array(Xint)[np.newaxis].T
Yint = exact_sol(Xint)

#define the model 
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(64, "tanh")
l2 = tf.keras.layers.Dense(64, "tanh")
l3 = tf.keras.layers.Dense(1, None)
train_op = tf.keras.optimizers.Adam()
num_epoch = 10000
print_epoch = 100
pred_model = model([l1, l2, l3], train_op, num_epoch, print_epoch)

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Yint_tf = tf.convert_to_tensor(Yint)

#training
print("Training (ADAM)...")
pred_model.network_learn(Xint_tf, Yint_tf)
print("Training (LBFGS)...")
loss_func = tfp_function_factory(pred_model, Xint_tf, Yint_tf)
# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)
# train the model with L-BFGS solver
results = tfp.optimizer.lbfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
         max_iterations=4000, num_correction_pairs=50, tolerance=1e-14)  
# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
loss_func.assign_new_model_parameters(results.position)

print("Testing...")
numPtsTest = 2*numPts
x_test = np.linspace(xmin, xmax, numPtsTest)    
x_test = np.array(x_test)[np.newaxis].T
x_tf = tf.convert_to_tensor(x_test)

y_test = pred_model.u(x_tf)    
y_exact = exact_sol(x_test)

plt.plot(x_test, y_test, x_test, y_exact)
plt.show()
plt.plot(x_test, y_exact-y_test)
plt.title("Error")
plt.show()
err = y_exact - y_test
print("L2-error norm: {}".format(np.linalg.norm(err)/np.linalg.norm(y_exact)))

# plot the loss convergence
plot_convergence_semilog(pred_model.adam_loss_hist, loss_func.history)
