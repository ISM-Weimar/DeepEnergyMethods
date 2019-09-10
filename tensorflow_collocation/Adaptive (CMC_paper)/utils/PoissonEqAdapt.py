# -*- coding: utf-8 -*-
"""
Class file for Poisson equation with collocation
Takes interior training data in the train method

"""
import tensorflow as tf
import numpy as np
import time


class PoissonEquationColl:
    '''
      Solve Poisson and Helmholtz equations of the form 
    -\Delta u(x,y) + alpha*u(x,y) = f(x,y) for (x,y) \in \Omega
    u(x,y) = u_bar for (x,y) \in \partial\Omega_{Dir}
    du(x,y)/dn =  
    '''
    
    # Initialize the class
    def __init__(self, dirichlet_bnd, neumann_bnd, alpha, layers, data_type, pen_dir, pen_neu):
      
        
        
        #Import parameters
        self.x_dirichlet = dirichlet_bnd[:,0:1]
        self.y_dirichlet = dirichlet_bnd[:,1:2]
        self.u_dirichlet = dirichlet_bnd[:,2:3]

        self.x_neumann = neumann_bnd[:,0:1]
        self.y_neumann = neumann_bnd[:,1:2]
        self.x_normal = neumann_bnd[:,2:3]
        self.y_normal = neumann_bnd[:,3:4]
        self.flux_val = neumann_bnd[:,4:5]
        
        #print(self.flux_val)
        
       

        self.alpha = alpha
        self.data_type = data_type
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        #Initialize bounds
        self.lb = np.concatenate((min(min(self.x_dirichlet), min(self.x_neumann)),
                            min(min(self.y_dirichlet), min(self.y_neumann))))    
        self.ub = np.concatenate((max(max(self.x_dirichlet), max(self.x_neumann)),
                            max(max(self.y_dirichlet), max(self.y_neumann))))
        

    
        # tf Placeholders
        self.x_dirichlet_tf = tf.placeholder(data_type, shape=[None, self.x_dirichlet.shape[1]])
        self.y_dirichlet_tf = tf.placeholder(data_type)
        self.u_dirichlet_tf = tf.placeholder(data_type)
       
        self.x_neumann_tf = tf.placeholder(data_type, shape=[None, self.x_neumann.shape[1]])
        self.y_neumann_tf = tf.placeholder(data_type, shape=[None, self.y_neumann.shape[1]])
        self.x_normal_tf = tf.placeholder(data_type, shape=[None, self.x_normal.shape[1]])
        self.y_normal_tf = tf.placeholder(data_type, shape=[None, self.y_normal.shape[1]])
        self.flux_val_tf = tf.placeholder(data_type, shape=[None, self.flux_val.shape[1]])
                
        self.x_int_tf = tf.placeholder(data_type)
        self.y_int_tf = tf.placeholder(data_type)
        self.f_int_tf = tf.placeholder(data_type)

        # tf Graphs
        self.u_dirichlet_pred = self.net_u(self.x_dirichlet_tf, self.y_dirichlet_tf)
        self.flux_pred = self.net_flux(self.x_neumann_tf, self.y_neumann_tf, \
                                       self.x_normal_tf, self.y_normal_tf)
        self.u_int_pred = self.net_u(self.x_int_tf,self.y_int_tf)
        self.f_int_pred = self.net_f_u(self.x_int_tf,self.y_int_tf)

        # Loss        
        self.loss_dir = tf.reduce_mean(tf.square(self.u_dirichlet_tf - self.u_dirichlet_pred))
        self.loss_neu = tf.reduce_mean(tf.square(self.flux_val_tf - self.flux_pred))
        self.loss_int = tf.reduce_mean(tf.square(self.f_int_tf - self.f_int_pred))                
        
        # Multiply by penalty parameters
        self.loss_dir = self.loss_dir * pen_dir
        self.loss_neu = self.loss_neu * pen_neu
        
        self.loss = self.loss_dir + self.loss_neu + self.loss_int
        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'eps': 1e-10,
                                                                         'maxcor': 100,
                                                                         'maxls': 50,                                                                  
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.lbfgs_buffer = []
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.data_type), dtype=self.data_type)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.data_type), dtype=self.data_type)
    
    def swish(self,x):
        y = tf.sigmoid(x)*x
        return y
    
    def neural_net(self,X,weights,biases):
#       
        num_layers = len(weights) + 1
        print(X)
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self,x,y):
        X = tf.concat([x,y],-1)
        u = self.neural_net(X,self.weights,self.biases)

        return u
    
    def net_u_x(self,x,y):
        # compute the x derivative
        u = self.net_u(x,y)
        u_x = tf.gradients(u,x)[0]        
        
        return u_x
    
    def net_u_y(self,x,y):
        # compute the y derivative
        u = self.net_u(x,y)
        u_y = tf.gradients(u,y)[0]        
        
        return u_y
    
    def net_flux(self,x,y,normal_x,normal_y):
        #compute the normal derivative
        u = self.net_u(x,y)
        u_x = tf.gradients(u,x)[0]  
        u_y = tf.gradients(u,y)[0]
        flux = u_x*normal_x + u_y*normal_y
        return flux

    def net_f_u(self,x,y):

        u = self.net_u(x,y)
        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x,x)[0]        
        u_y = tf.gradients(u,y)[0]
        u_yy = tf.gradients(u_y,y)[0]
        f_u = -(u_xx + u_yy) + self.alpha*u

        return f_u


    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
#        print('Loss:', loss)

    def train(self, X_int, nIter):
        
        self.x_int = X_int[:,0:1]
        self.y_int = X_int[:,1:2]
        self.f_int = X_int[:,2:3]

        tf_dict = {self.x_dirichlet_tf: self.x_dirichlet,
                   self.y_dirichlet_tf: self.y_dirichlet,
                   self.u_dirichlet_tf: self.u_dirichlet,
                   self.x_neumann_tf: self.x_neumann,
                   self.y_neumann_tf: self.y_neumann,
                   self.x_normal_tf: self.x_normal,
                   self.y_normal_tf: self.y_normal,
                   self.flux_val_tf: self.flux_val,
                   self.x_int_tf: self.x_int,
                   self.y_int_tf: self.y_int,
                   self.f_int_tf: self.f_int}                   

        start_time = time.time()
        
        self.loss_adam_buff = np.zeros(nIter)
        
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            
            
            #print(flux_pred_value)
            self.loss_adam_buff[it] = loss_value
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value_dir = self.sess.run(self.loss_dir, tf_dict)
                loss_value_neu = self.sess.run(self.loss_neu, tf_dict)
                print('It: %d, Total Loss: %.3e, Dirichlet Loss: %.3e, Neumann Loss: %.3e, Time: %.2f' %
                      (it, loss_value, loss_value_dir, loss_value_neu, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2]}
        u_star = self.sess.run(self.u_int_pred, tf_dict)

        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2]}
        f_u_star = self.sess.run(self.f_int_pred, tf_dict)

        return u_star, f_u_star
    
    def getWeightsBiases(self):
        weights =  self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases
