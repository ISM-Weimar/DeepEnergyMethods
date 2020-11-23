#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:54:28 2020

@author: cosmin
"""
import tensorflow as tf

class Poisson2D_coll(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Poisson2D_coll, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
            
    def call(self, X):
        return self.u(X[:,0:1], X[:,1:2])
    
    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X
    
    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val = self.u(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)
        dudy_val = tape.gradient(u_val, yPhys)
        del tape
        return dudx_val, dudy_val
    
    # Return the second derivative
    @tf.function
    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val, dudy_val = self.du(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)
        d2udy2_val = tape.gradient(dudy_val, yPhys)
        del tape        
        return d2udx2_val, d2udy2_val
         
    #Custom loss function
    @tf.function
    def get_loss(self,Xint, Yint, Xbnd, Ybnd):
        u_val_bnd = self.call(Xbnd)
        xPhys = Xint[:,0:1]
        yPhys = Xint[:,1:2]
        
        d2udx2_val_int, d2udy2_val_int = self.d2u(xPhys, yPhys)
        f_val_int = -(d2udx2_val_int + d2udy2_val_int)
        int_loss = tf.reduce_mean(tf.math.square(f_val_int - Yint))
        bnd_loss = tf.reduce_mean(tf.math.square(u_val_bnd - Ybnd))
        return int_loss+bnd_loss
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd, Ybnd):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd, Ybnd)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self,Xint,Yint, Xbnd, Ybnd):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
                
class Wave1D(tf.keras.Model): 
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Wave1D, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
            
    def call(self, X):
        return self.u(X[:,0:1], X[:,1:2])
    
    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X
    
    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val = self.u(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)
        dudy_val = tape.gradient(u_val, yPhys)
        del tape
        return dudx_val, dudy_val
    
    # Return the second derivative
    @tf.function
    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val, dudy_val = self.du(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)
        d2udy2_val = tape.gradient(dudy_val, yPhys)
        del tape        
        return d2udx2_val, d2udy2_val
         
    #Custom loss function
    @tf.function
    def get_loss(self,Xint, Yint, Xbnd, Ybnd, Xinit, Yinit):
        
        xPhys = Xint[:,0:1]
        tPhys = Xint[:,1:2]
        
        d2udx2_val_int, d2udt2_val_int = self.d2u(xPhys, tPhys)
        f_val_int = d2udx2_val_int - d2udt2_val_int
        int_loss = tf.reduce_mean(tf.math.square(f_val_int - Yint))
        
        u_val_init = self.call(Xinit)
        xPhysInit = Xinit[:,0:1]
        yPhysInit = Xinit[:,1:2]
        _, dudt_val_init = self.du(xPhysInit, yPhysInit)
        init_loss =  tf.reduce_mean(tf.math.square(u_val_init - Yinit[:,0:1])) + \
                     tf.reduce_mean(tf.math.square(dudt_val_init - Yinit[:,1:2]))
        
        xPhysBnd = Xbnd[:,0:1]
        yPhysBnd = Xbnd[:,1:2]
        
        dudx_val_bnd, _ = self.du(xPhysBnd, yPhysBnd)
        bnd_loss = tf.reduce_mean(tf.math.square(dudx_val_bnd - Ybnd))
        return int_loss+init_loss+bnd_loss#+tf.sqrt(tf.square(int_loss-bnd_loss)+0.01)
      
    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd, Ybnd, Xinit, Yinit):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd, Ybnd, Xinit, Yinit)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
      
    # perform gradient descent
    def network_learn(self, Xint,Yint, Xbnd, Ybnd, Xinit, Yinit):
        xmin = tf.math.reduce_min(Xint[:,0])
        ymin = tf.math.reduce_min(Xint[:,1])
        xmax = tf.math.reduce_max(Xint[:,0])
        ymax = tf.math.reduce_max(Xint[:,1])
        print("xmin =", xmin)
        print("ymin =", ymin)
        print("xmax =", xmax)
        print("ymax =", ymax)
        self.bounds = {"lb" : tf.reshape(tf.concat([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.concat([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd, Xinit, Yinit)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                print("Epoch {} loss: {}".format(i, L))
            