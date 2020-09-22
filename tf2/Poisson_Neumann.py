#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson equation example
Solve the equation -u''(x) = f(x) for x\in(a,b) with Dirichlet boundary conditions u(a)=u0
and Neumann boundary condition u'(b) = u1
@author: cosmin
"""
import tensorflow as tf
import numpy as np
import time
from utils.tfp_loss import tfp_function_factory
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
#make figures bigger on HiDPI monitors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
tf.random.set_seed(42)

class model(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(model, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
            
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
    
    # Return the second derivative
    def d2u(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            du_val = self.du(X)
        d2u_val = tape.gradient(du_val, X)
        return d2u_val
         
    #Custom loss function
    def get_loss(self,Xint, Yint, XbndDir, YbndDir, XbndNeu, YbndNeu):
        u_val_bnd_dir = self.u(XbndDir)
        du_val_bnd_neu = self.du(XbndNeu)
        d2u_val_int=self.d2u(Xint)
        int_loss = tf.reduce_mean(tf.math.square(d2u_val_int - Yint))
        bnd_loss_dir = tf.reduce_mean(tf.math.square(u_val_bnd_dir - YbndDir))
        bnd_loss_neu = tf.reduce_mean(tf.math.square(du_val_bnd_neu - YbndNeu))
        return int_loss+bnd_loss_dir+bnd_loss_neu
      
    # get gradients
    def get_grad(self, Xint, Yint, XbndDir, YbndDir, XbndNeu, YbndNeu):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, XbndDir, YbndDir, XbndNeu, YbndNeu)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self,Xint,Yint, XbndDir, YbndDir, XbndNeu, YbndNeu):
        self.bounds = {"lb" : tf.math.reduce_min(Xint),
                       "ub" : tf.math.reduce_max(Xint)}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, XbndDir, YbndDir, XbndNeu, YbndNeu)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))

 
if __name__ == "__main__":
    
    #define the RHS function f(x)
    k = 4
    def rhs_fun(x):
        f = -k**2*np.pi**2*np.sin(k*np.pi*x)
        return f
    
    def exact_sol(x):
        y = np.sin(k*np.pi*x)
        return y
    
    def deriv_exact_sol(input):
        output = k*np.pi*np.cos(k*np.pi*input)
        return output
    
    #define the input and output data set
    xmin = 0
    xmax = 1
    numPts = 201
    data_type = "float64"
    
    Xint = np.linspace(xmin, xmax, numPts).astype(data_type)
    Xint = np.array(Xint)[np.newaxis].T
    Yint = rhs_fun(Xint)
    
    XbndDir = np.array([[xmin]]).astype(data_type)
    YbndDir = exact_sol(XbndDir)
    XbndNeu = np.array([[xmin]]).astype(data_type)
    YbndNeu = deriv_exact_sol(XbndNeu)
    
    #define the model 
    tf.keras.backend.set_floatx(data_type)
    l1 = tf.keras.layers.Dense(64, "tanh")
    l2 = tf.keras.layers.Dense(64, "tanh")
    l3 = tf.keras.layers.Dense(1, None)
    train_op = tf.keras.optimizers.Adam()
    num_epoch = 5000
    print_epoch = 100
    pred_model = model([l1, l2, l3], train_op, num_epoch, print_epoch)
    
    #convert the training data to tensors
    Xint_tf = tf.convert_to_tensor(Xint[1:-1])
    Yint_tf = tf.convert_to_tensor(Yint[1:-1])
    XbndDir_tf = tf.convert_to_tensor(XbndDir)
    YbndDir_tf = tf.convert_to_tensor(YbndDir)
    XbndNeu_tf = tf.convert_to_tensor(XbndNeu)
    YbndNeu_tf = tf.convert_to_tensor(YbndNeu)

    #training
    print("Training (ADAM)...")
    t0 = time.time()
    pred_model.network_learn(Xint_tf, Yint_tf, XbndDir_tf, YbndDir_tf, XbndNeu_tf, YbndNeu_tf)
    t1 = time.time()
    print("Time taken (ADAM)", t1-t0, "seconds")
    print("Training (LBFGS)...")
    
    loss_func = tfp_function_factory(pred_model, Xint_tf, Yint_tf, XbndDir_tf, 
                                     YbndDir_tf, XbndNeu_tf, YbndNeu_tf)
    # convert initial model parameters to a 1D tf.Tensor
    init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)
    # train the model with L-BFGS solver
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=loss_func, initial_position=init_params,
              max_iterations=1000, num_correction_pairs=100, tolerance=1e-14)  
    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    loss_func.assign_new_model_parameters(results.position)
    #loss_func.assign_new_model_parameters(results.x)
    t2 = time.time()
    print("Time taken (LBFGS)", t2-t1, "seconds")
    print("Time taken (all)", t2-t0, "seconds")
    print("Testing...")
    numPtsTest = 2*numPts
    x_test = np.linspace(xmin, xmax, numPtsTest).astype(data_type)
    x_test = np.array(x_test)[np.newaxis].T
    x_tf = tf.convert_to_tensor(x_test)

    y_test = pred_model.u(x_tf)    
    y_exact = exact_sol(x_test)
    dy_test = pred_model.du(x_tf)


    plt.plot(x_test, y_test, x_test, y_exact)
    plt.show()
    plt.plot(x_test, y_exact-y_test)
    plt.title("Error")
    plt.show()
    err = y_exact - y_test
    print("L2-error norm: {}".format(np.linalg.norm(err)/np.linalg.norm(y_exact)))
    
    dy_exact = deriv_exact_sol(x_test)
    plt.plot(x_test, dy_test, x_test, dy_exact)
    plt.title("First derivative")
    plt.show()
    
    err_deriv = dy_exact - dy_test
    plt.plot(x_test, err_deriv)
    plt.title("Error for first derivative")
    plt.show
    print("Relative error for first derivative: {}".format(np.linalg.norm(err_deriv)/np.linalg.norm(dy_exact)))