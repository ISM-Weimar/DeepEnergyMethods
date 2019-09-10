"""
script for Poisson's equation in 1D with Dirichlet boundary conditions at both ends
Governing equation: -u''(x) = k^2*pi^2*sin(k*pi*x) for x \in (0,1), u(0)=u(1)=0
Exact solution: u(x)=sin(k*pi*x)
"""


import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from pyDOE import lhs
import time


np.random.seed(1234)
tf.set_random_seed(1234)


#make figures bigger on HiDPI monitors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, lb, u_lb, rb, u_rb, X_f, layers, k):
            
        self.lb = lb             
        self.rb = rb
        
        self.x_lb = lb
        self.x_rb = rb
        self.u_lb = u_lb
        self.u_rb = u_rb
            
        self.x_f = X_f
        self.k = k
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
                
        # tf Placeholders        
        
        self.x_lb_tf = tf.placeholder(tf.float32)
        self.u_lb_tf = tf.placeholder(tf.float32)
        self.x_rb_tf = tf.placeholder(tf.float32)
        self.u_rb_tf = tf.placeholder(tf.float32)
        
        self.x_f_tf = tf.placeholder(tf.float32)

        # tf Graphs
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf)
        self.u_rb_pred, self.u_x_rb_pred = self.net_u(self.x_rb_tf)
        self.u_f_pred, self.u_x_f_pred = self.net_u(self.x_f_tf)
        self.f_u_pred = self.net_f_u(self.x_f_tf)
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_lb_tf - self.u_lb_pred)) + \
                    tf.reduce_mean(tf.square(self.u_rb_tf - self.u_rb_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred))
        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
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
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
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
        
        #use tf.multiply for the first iteration/layers
        W = weights[0]
        b = biases[0]
        H = tf.tanh(tf.add(tf.multiply(H,W), b))
        
        #use tf.matmul for the remanining layers
        for l in range(1,num_layers-2):
            W = weights[l]
            b = biases[l]   
            H = tf.tanh(tf.add(tf.matmul(H,W), b))
 
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, x):
        
        u = self.neural_net(x, self.weights, self.biases)               
        u_x = tf.gradients(u, x)[0]

        return u, u_x

    def net_f_u(self, x):
        u, u_x = self.net_u(x)
        
        u_xx = tf.gradients(u_x, x)[0]        
        pi = tf.constant(np.pi)
        f_u =u_xx + self.k**2*pi**2*tf.sin(self.k*pi*x) 
        
        return f_u
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x_lb_tf: self.x_lb,
                   self.u_lb_tf: self.u_lb,
                   self.x_rb_tf: self.x_rb,
                   self.u_rb_tf: self.u_rb,
                   self.x_f_tf: self.x_f}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
        
        tf_dict = {self.x_f_tf: X_star}        
        u_star = self.sess.run(self.u_f_pred, tf_dict)
       
        tf_dict = {self.x_f_tf: X_star}        
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
               
        return u_star, f_u_star
    
    
    def getWeightsBiases(self):
        weights =  self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases
    
if __name__ == "__main__": 
     
    noise = 0.0        
    
    # Domain bounds
    lb = 0.0  #left boundary
    rb = 1.0  #right boundary

    N_f = 1000  #number of interior points
    layers = [1, 400, 1]
    
    #constant in the exact solution and RHS
    k = 4;
    
    
    ###########################
    #Dirichlet data
    u_left = 0;
    u_right = 0;
    
    
    # generate the collocation points
    X_f = lb + (rb-lb)*lhs(1, N_f)
            
    model = PhysicsInformedNN(lb, u_left, rb, u_right, X_f, layers, k)
             
    start_time = time.time()                
    model.train(1000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    
    # test data
    nPred = 1001
    X_star = np.linspace(lb, rb, nPred)[np.newaxis]
    X_star = X_star.T
    
    u_pred, f_u_pred = model.predict(X_star)
    
    u_exact = np.sin(k*np.pi*X_star);
    u_pred_err = u_exact-u_pred
    
    error_u = (np.linalg.norm(u_exact-u_pred,2)/np.linalg.norm(u_exact,2))
    print('Relative error u: %e' % (error_u))       
    
    np.savetxt('uPred.out', u_pred, delimiter=',')
    np.savetxt('f_uPred.out', f_u_pred, delimiter=',')
    
    
    plt.plot(X_star,u_pred, X_star, np.sin(k*np.pi*X_star), '--')
    plt.title('$u_{comp}$ and $u_{exact}$')
    plt.show()
    plt.plot(X_star, u_pred_err, '--')
    plt.title('$u_{exact} - u_{comp}$')
    plt.show()
    
    weights, biases = model.getWeightsBiases()
    #print(weights)
    #print(biases)
    