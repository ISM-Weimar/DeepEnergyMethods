# Class file for Physics Informed Neural Network

import tensorflow as tf
import numpy as np
import time

class Elasticity2D:
    # Initialize the class
    def __init__(self, model_data, model_pts, NN_param):
        
        #initialize fields for model data
        self.E = model_data['E']
        self.nu = model_data['nu']
        
        #plane stress        
        self.c11 = self.E/(1-self.nu**2)
        self.c22 = self.E/(1-self.nu**2)
        self.c12 = self.E*self.nu/(1-self.nu**2)
        self.c21 = self.E*self.nu/(1-self.nu**2)
        self.c31 = 0.0
        self.c32 = 0.0
        self.c13 = 0.0
        self.c23 = 0.0
        self.c33 = self.E/(2*(1+self.nu))
        
        #initialized fields for model points
        X_int = model_pts['X_int']
        X_bnd = model_pts['X_bnd']

        self.x_int = X_int[:, 0:1]
        self.y_int = X_int[:, 1:2]
        self.wt_int = X_int[:, 2:3]
        
        self.x_bnd = X_bnd[:, 0:1]
        self.y_bnd = X_bnd[:, 1:2]        
        self.wt_bnd = X_bnd[:, 2:3]
        self.trac_x_bnd = X_bnd[:, 3:4]
        self.trac_y_bnd = X_bnd[:, 4:5]                             
        
        self.lb = np.array([np.min(self.x_int), np.min(self.y_int)])
        self.ub = np.array([np.max(self.x_int), np.max(self.y_int)])
        
        self.layers = NN_param['layers']
        self.data_type = NN_param['data_type']
        self.weights, self.biases = self.initialize_NN(self.layers)

        # tf Placeholders
        self.x_int_tf = tf.placeholder(self.data_type)
        self.y_int_tf = tf.placeholder(self.data_type)
        self.wt_int_tf = tf.placeholder(self.data_type)
        
        self.x_bnd_tf = tf.placeholder(self.data_type)
        self.y_bnd_tf = tf.placeholder(self.data_type)
        self.wt_bnd_tf = tf.placeholder(self.data_type)
        self.trac_x_bnd_tf = tf.placeholder(self.data_type)
        self.trac_y_bnd_tf = tf.placeholder(self.data_type)

        # tf Graphs
        self.u_pred, self.v_pred = self.net_uv(self.x_int_tf,self.y_int_tf)
        self.fx_pred, self.fy_pred = self.net_traction(self.x_bnd_tf, self.y_bnd_tf,                                               
                                               self.trac_x_bnd_tf, self.trac_y_bnd_tf)
        self.energy_pred, self.sigma_x_pred, self.sigma_y_pred, \
              self.tau_xy_pred = self.net_energy(self.x_int_tf,self.y_int_tf)

        # Loss
        self.loss_neu = tf.reduce_sum((self.fx_pred+self.fy_pred)*self.wt_bnd)
        self.loss_int = tf.reduce_sum(self.energy_pred*self.wt_int)
        
        self.loss = self.loss_int  - self.loss_neu                                    

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 100,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
    
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session()

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

    def neural_net(self,X,weights,biases):
        num_layers = len(weights) + 1
		
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        #H = tf.sigmoid(tf.add(tf.matmul(H, weights[0]), biases[0]))
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            #H = tf.tanh(tf.add(tf.matmul(H, W), b))
            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))**2
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self,x,y):

        X = tf.concat([x,y],1)

        uv = self.neural_net(X,self.weights,self.biases)
        
        u = uv[:,0:1]
        v = uv[:,1:2]

        return u, v

    def net_traction(self,x,y,tracX,tracY):

        u, v = self.net_uv(x,y)
        
        trX = u*tracX
        trY = v*tracY
        
        return trX, trY
    
    def net_energy(self,x,y):

        u, v = self.net_uv(x,y)
        u_x = tf.gradients(u,x)[0]
        v_y = tf.gradients(v,y)[0]
        u_y = tf.gradients(u,y)[0]
        v_x = tf.gradients(v,x)[0]
        u_xy = (u_y + v_x)
        
        sigmaX = self.c11*u_x + self.c12*v_y
        sigmaY = self.c21*u_x + self.c22*v_y
        tauXY = self.c33*u_xy
        
        energy = 0.5*(sigmaX*u_x + sigmaY*v_y + tauXY*u_xy)
        
        return energy, sigmaX, sigmaY, tauXY

    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        print('Loss:', loss)

    def train(self, nIter):
        
        tf_dict = {self.x_bnd_tf: self.x_bnd, 
                   self.y_bnd_tf: self.y_bnd, 
                   self.wt_bnd_tf: self.wt_bnd,
                   self.trac_x_bnd_tf: self.trac_x_bnd,
                   self.trac_y_bnd_tf: self.trac_y_bnd,                   
                   self.x_int_tf: self.x_int, 
                   self.y_int_tf: self.y_int, 
                   self.wt_int_tf: self.wt_int}

        start_time = time.time()
        self.loss_adam_buff = np.zeros(nIter)
        
        for it in range(nIter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_neu = self.sess.run(self.loss_neu, tf_dict)
                loss_int = self.sess.run(self.loss_int, tf_dict)
                print('It: %d, Total Loss: %.3e, Int Loss: %.3e, Neumann Loss: %.3e, Time: %.2f' %
                      (it, loss_value, loss_int, loss_neu, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)            
        
    def predict(self, X_star):

        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        energy_star = self.sess.run(self.energy_pred, tf_dict)
        sigma_x_star = self.sess.run(self.sigma_x_pred, tf_dict)
        sigma_y_star = self.sess.run(self.sigma_y_pred, tf_dict)
        tau_xy_star = self.sess.run(self.tau_xy_pred, tf_dict)

        return u_star, v_star, energy_star, sigma_x_star, sigma_y_star, tau_xy_star
   
class Elasticity3D:
    # Initialize the class
    def __init__(self, model_data, model_pts, NN_param):
        
        #initialize fields for model data
        self.E = model_data['E']
        self.nu = model_data['nu']
        
        self.c11 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c22 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c33 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c12 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c13 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c21 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c23 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c31 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c32 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c44 = self.E/(2*(1+self.nu))
        self.c55 = self.E/(2*(1+self.nu))
        self.c66 = self.E/(2*(1+self.nu))       
        
        # Initialized fields for model points
        X_int = model_pts['X_int']
        X_bnd = model_pts['X_bnd']

        self.x_int = X_int[:, 0:1]
        self.y_int = X_int[:, 1:2]
        self.z_int = X_int[:, 2:3]
        self.wt_int = X_int[:, 3:4]
        
        self.x_bnd = X_bnd[:, 0:1]
        self.y_bnd = X_bnd[:, 1:2]
        self.z_bnd = X_bnd[:, 2:3]
        self.wt_bnd = X_bnd[:, 3:4]
        self.trac_x_bnd = X_bnd[:, 4:5]
        self.trac_y_bnd = X_bnd[:, 5:6]
        self.trac_z_bnd = X_bnd[:, 6:7]
        
        self.lb = np.array([np.min(self.x_int), np.min(self.y_int), np.min(self.z_int)])
        self.ub = np.array([np.max(self.x_int), np.max(self.y_int), np.max(self.z_int)])
        
        self.layers = NN_param['layers']
        self.data_type = NN_param['data_type']
        self.weights, self.biases = self.initialize_NN(self.layers)

        # tf Placeholders
        self.x_int_tf = tf.placeholder(self.data_type)
        self.y_int_tf = tf.placeholder(self.data_type)
        self.z_int_tf = tf.placeholder(self.data_type)
        self.wt_int_tf = tf.placeholder(self.data_type)
        
        self.x_bnd_tf = tf.placeholder(self.data_type)
        self.y_bnd_tf = tf.placeholder(self.data_type)
        self.z_bnd_tf = tf.placeholder(self.data_type)
        self.wt_bnd_tf = tf.placeholder(self.data_type)
        self.trac_x_bnd_tf = tf.placeholder(self.data_type)
        self.trac_y_bnd_tf = tf.placeholder(self.data_type)
        self.trac_z_bnd_tf = tf.placeholder(self.data_type)

        # tf Graphs
        self.u_pred, self.v_pred, self.w_pred = self.net_uvw(self.x_int_tf, 
                                                             self.y_int_tf,
                                                             self.z_int_tf)
        self.fx_pred, self.fy_pred, self.fz_pred = self.net_traction(self.x_bnd_tf, 
                           self.y_bnd_tf, self.z_bnd_tf, self.trac_x_bnd_tf, 
                           self.trac_y_bnd_tf, self.trac_z_bnd_tf)
        self.energy_pred, self.sigma_x_pred, self.sigma_y_pred, \
                self.sigma_z_pred, self.tau_xy_pred, self.tau_yz_pred, \
                self.tau_zx_pred = self.net_energy(self.x_int_tf, self.y_int_tf, self.z_int_tf)

        # Loss
        self.loss_neu = tf.reduce_sum((self.fx_pred+self.fy_pred+self.fz_pred)*self.wt_bnd)
        self.loss_int = tf.reduce_sum(self.energy_pred*self.wt_int)
        
        self.loss = self.loss_int  - self.loss_neu                                    

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 100,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
    
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session()

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

    def neural_net(self,X,weights,biases):
        num_layers = len(weights) + 1
		
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            #H = tf.tanh(tf.add(tf.matmul(H, W), b))
            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))**2
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uvw(self,x,y):

        X = tf.concat([x,y],1)

        uvw = self.neural_net(X,self.weights,self.biases)
        
        u = uvw[:,0:1]
        v = uvw[:,1:2]
        w = uvw[:,2:3]

        return u, v, w

    def net_traction(self,x,y,z,tracX,tracY,tracZ):

        u, v, w = self.net_uvw(x,y,z)
        
        trX = u*tracX
        trY = v*tracY
        trZ = w*tracZ
        
        return trX, trY, trZ
    
    def net_energy(self, x, y, z):
        u, v, w = self.net_uvw(x, y, z)
               
        u_x = tf.gradients(u,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_z = tf.gradients(u,z)[0]
        v_x = tf.gradients(v,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_z = tf.gradients(v,z)[0]
        w_x = tf.gradients(w,x)[0]
        w_y = tf.gradients(w,y)[0]
        w_z = tf.gradients(w,z)[0]
        u_xy = (u_y + v_x)
        u_yz = (v_z + w_y)
        u_zx = (u_z + w_x)                
        
        sigmaX = self.c11*u_x + self.c12*v_y + self.c13*w_z
        sigmaY = self.c21*u_x + self.c22*v_y + self.c23*w_z
        sigmaZ = self.c31*u_x + self.c32*v_y + self.c33*w_z
        tauYZ = self.c44*u_yz
        tauZX = self.c55*u_zx
        tauXY = self.c66*u_xy
        
        energy = 0.5*(sigmaX*u_x + sigmaY*v_y + sigmaZ*w_z + tauYZ*u_yz + \
                          tauZX*u_zx + tauXY*u_xy)
        
        return energy, sigmaX, sigmaY, sigmaZ, tauXY, tauYZ, tauZX    
        
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)

    def train(self, nIter):
        
        tf_dict = {self.x_bnd_tf: self.x_bnd, 
                   self.y_bnd_tf: self.y_bnd,
                   self.z_bnd_tf: self.z_bnd,
                   self.wt_bnd_tf: self.wt_bnd,
                   self.trac_x_bnd_tf: self.trac_x_bnd,
                   self.trac_y_bnd_tf: self.trac_y_bnd,
                   self.trac_z_bnd_tf: self.trac_z_bnd,
                   self.x_int_tf: self.x_int,
                   self.y_int_tf: self.y_int,
                   self.z_int_tf: self.z_int,
                   self.wt_int_tf: self.wt_int}

        start_time = time.time()
        self.loss_adam_buff = np.zeros(nIter)
        
        for it in range(nIter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_neu = self.sess.run(self.loss_neu, tf_dict)
                loss_int = self.sess.run(self.loss_int, tf_dict)
                print('It: %d, Total Loss: %.3e, Int Loss: %.3e, Neumann Loss: %.3e, Time: %.2f' %
                      (it, loss_value, loss_int, loss_neu, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)            
        
    def predict(self, X_star):

        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2], 
                   self.z_int_tf: X_star[:,2:3]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        sigma_x_star = self.sess.run(self.sigma_x_pred, tf_dict)
        sigma_y_star = self.sess.run(self.sigma_y_pred, tf_dict)
        sigma_z_star = self.sess.run(self.sigma_z_pred, tf_dict)
        tau_xy_star = self.sess.run(self.tau_xy_pred, tf_dict)
        tau_yz_star = self.sess.run(self.tau_yz_pred, tf_dict)
        tau_zx_star = self.sess.run(self.tau_zx_pred, tf_dict)
                
        energy_star = self.sess.run(self.energy_pred, tf_dict)

        return u_star, v_star, w_star, energy_star, sigma_x_star, sigma_y_star, \
            sigma_z_star, tau_xy_star, tau_yz_star, tau_zx_star
                