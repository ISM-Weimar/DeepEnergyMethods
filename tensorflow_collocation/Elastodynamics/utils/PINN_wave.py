# PINN file for Wave equation

import tensorflow as tf
import numpy as np
import time

class WaveEquation:
    # Initialize the class
    def __init__(self, lb, rb, X_f, X_b, X_init, layers):
            
        self.lb = lb             
        self.rb = rb
        
        #unpack interior collocation space-time points
        self.x_int = X_f[:,0:1]
        self.t_int = X_f[:,1:2]
        
        #unpack boundary space-time points and displacement values
        self.x_bnd = X_b[:,0:1]
        self.t_bnd = X_b[:,1:2]
        self.u_x_bnd = X_b[:,2:3]
            
        #unpack point location and intitial displacement and velocity values
        self.x_init = X_init[:,0:1]
        self.t_init = X_init[:,1:2]
        self.u_init = X_init[:,2:3]
        self.v_init = X_init[:,3:4]
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
                
        # tf Placeholders                
        self.x_bnd_tf = tf.placeholder(tf.float32)
        self.t_bnd_tf = tf.placeholder(tf.float32)
        self.u_x_bnd_tf = tf.placeholder(tf.float32)
        self.x_init_tf = tf.placeholder(tf.float32)
        self.t_init_tf = tf.placeholder(tf.float32)
        self.u_init_tf = tf.placeholder(tf.float32)
        self.v_init_tf = tf.placeholder(tf.float32)
        self.x_int_tf = tf.placeholder(tf.float32)
        self.t_int_tf = tf.placeholder(tf.float32)
        
        # tf Graphs
        _, self.v_bnd_pred, self.u_x_bnd_pred = self.net_u(self.x_bnd_tf, self.t_bnd_tf)
        self.u_init_pred, self.v_init_pred, _ = self.net_u(self.x_init_tf, self.t_init_tf)
        self.u_f_pred, self.v_f_pred, _ = self.net_u(self.x_int_tf, self.t_int_tf)
        self.f_u_pred = self.net_f_u(self.x_int_tf, self.t_int_tf)
        
        # Loss
        self.loss_bnd = tf.reduce_mean(tf.square(self.u_x_bnd_tf - self.u_x_bnd_pred))
        self.loss_u_init = tf.reduce_mean(tf.square(self.u_init_tf - self.u_init_pred))                    
        self.loss_v_init = tf.reduce_mean(tf.square(self.v_init_tf - self.v_init_pred))
        self.loss_resid = tf.reduce_mean(tf.square(self.f_u_pred))
        
        self.loss = self.loss_bnd + self.loss_u_init + self.loss_v_init + self.loss_resid
        
        self.lbfgs_buffer = []

        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 100,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(0.1*tf.ones([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.rb - self.lb) - 1.0
                                
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]   
            H = tf.nn.tanh(tf.add(tf.matmul(H,W), b))
            #H = tf.nn.relu(tf.add(tf.matmul(H, W), b))**2
 
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, x, t):
        
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)               
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]

        return u, u_t, u_x

    def net_f_u(self, x, t):
        u, u_t, u_x = self.net_u(x, t)
                
        u_xx = tf.gradients(u_x, x)[0]
        u_tt = tf.gradients(u_t, t)[0]
        
        f_u = u_xx - u_tt 
        
        return f_u
    
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x_bnd_tf: self.x_bnd,
        self.t_bnd_tf: self.t_bnd,
        self.u_x_bnd_tf: self.u_x_bnd,
        self.x_init_tf: self.x_init,
        self.t_init_tf: self.t_init,
        self.u_init_tf: self.u_init,
        self.v_init_tf: self.v_init,
        self.x_int_tf: self.x_int,
        self.t_int_tf:  self.t_int}                
        
        start_time = time.time()
        self.loss_adam_buff = np.zeros(nIter)
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                
                loss_bnd_value = self.sess.run(self.loss_bnd, tf_dict)
                loss_u_init_value = self.sess.run(self.loss_u_init, tf_dict)
                loss_v_init_value = self.sess.run(self.loss_v_init, tf_dict)
                                
                print('It: %d, Loss: %.3e, Bnd Loss: %.3e, u_init_loss: %.3e, v_init_loss, %.3e, Time: %.2f' % 
                      (it, loss_value, loss_bnd_value, loss_u_init_value, 
                       loss_v_init_value, elapsed))
                start_time = time.time()
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, XT_star):
        
        X_star = XT_star[:, 0:1]
        T_star = XT_star[:, 1:2]
        
        tf_dict = {self.x_int_tf: X_star, self.t_int_tf: T_star}        
        u_star = self.sess.run(self.u_f_pred, tf_dict)
        v_star = self.sess.run(self.v_f_pred, tf_dict)
               
        return u_star, v_star
    
    
    def getWeightsBiases(self):
        weights =  self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases

