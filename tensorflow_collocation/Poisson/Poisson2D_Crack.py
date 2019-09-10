'''
Implement a Poisson 2D problem on a cracked domain with pure Dirichlet boundary conditions:
    - \Delta u(x,y) = f(x,y) for (x,y) \in \Omega:= (-1,1)x(-1,1) \ (0,1)x{0}
    u(x,y) = 0, for (x,y) \in \partial \Omega
    f(x,y) = 1
    Problem from: Weinan E and Bing Yu - The Deep Ritz method: A deep learning-based
    numerical algorithm for solving variational problems, Section 3.1
    Note: this will take ~1 hour to run on a quad-core machine! Further optimizations would be welcome.
'''


import tensorflow as tf
import numpy as np

#make figures bigger on HiDPI monitors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import matplotlib.pyplot as plt
from pyDOE import lhs

import time

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_b_left, u_b_left, X_b_bottom, u_b_bottom, X_b_top, u_b_top, X_b_right, u_b_right, X_b_crack, u_b_crack, X_f, layers, lb, ub):
      
        self.lb = lb
        self.ub = ub

        self.x_b_left = X_b_left[:,0:1]
        self.y_b_left = X_b_left[:,1:2]
        self.u_b_left = u_b_left

        self.x_b_bottom = X_b_bottom[:,0:1]
        self.y_b_bottom = X_b_bottom[:,1:2]
        self.u_b_bottom = u_b_bottom

        self.x_b_top = X_b_top[:,0:1]
        self.y_b_top = X_b_top[:,1:2]
        self.u_b_top = u_b_top
        
        self.x_b_right = X_b_right[:,0:1]
        self.y_b_right = X_b_right[:,1:2]
        self.u_b_right = u_b_right
        
        self.x_b_crack = X_b_crack[:,0:1]
        self.y_b_crack = X_b_crack[:,1:2]
        self.u_b_crack = u_b_crack


        self.x_f = X_f[:,0:1]
        self.y_f = X_f[:,1:2]

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.x_b_left_tf = tf.placeholder(tf.float32, shape=[None, self.x_b_left.shape[1]])
        self.y_b_left_tf = tf.placeholder(tf.float32, shape=[None, self.y_b_left.shape[1]])
        self.u_b_left_tf = tf.placeholder(tf.float32, shape=[None, self.u_b_left.shape[1]])
        self.x_b_bottom_tf = tf.placeholder(tf.float32, shape=[None, self.x_b_bottom.shape[1]])
        self.y_b_bottom_tf = tf.placeholder(tf.float32, shape=[None, self.y_b_bottom.shape[1]])
        self.u_b_bottom_tf = tf.placeholder(tf.float32, shape=[None, self.u_b_bottom.shape[1]])
        self.x_b_top_tf = tf.placeholder(tf.float32, shape=[None, self.x_b_top.shape[1]])
        self.y_b_top_tf = tf.placeholder(tf.float32, shape=[None, self.y_b_top.shape[1]])
        self.u_b_top_tf = tf.placeholder(tf.float32, shape=[None, self.u_b_top.shape[1]])
        self.x_b_right_tf = tf.placeholder(tf.float32, shape=[None, self.x_b_right.shape[1]])
        self.y_b_right_tf = tf.placeholder(tf.float32, shape=[None, self.y_b_right.shape[1]])
        self.u_b_right_tf = tf.placeholder(tf.float32, shape=[None, self.u_b_right.shape[1]])
        self.x_b_crack_tf = tf.placeholder(tf.float32, shape=[None, self.x_b_crack.shape[1]])
        self.y_b_crack_tf = tf.placeholder(tf.float32, shape=[None, self.y_b_crack.shape[1]])
        self.u_b_crack_tf = tf.placeholder(tf.float32, shape=[None, self.u_b_crack.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])

        # tf Graphs
        self.u_b_left_pred = self.net_u(self.x_b_left_tf,self.y_b_left_tf)
        self.u_f_pred = self.net_u(self.x_f_tf,self.y_f_tf)
        self.u_b_bottom_pred = self.net_u(self.x_b_bottom_tf, self.y_b_bottom_tf)
        self.u_b_top_pred = self.net_u(self.x_b_top_tf,self.y_b_top_tf)
        self.u_b_crack_pred = self.net_u(self.x_b_crack_tf,self.y_b_crack_tf)
        self.u_b_right_pred = self.net_u(self.x_b_right_tf,self.y_b_right_tf)
        self.f_u_pred = self.net_f_u(self.x_f_tf,self.y_f_tf)

        # Loss

        self.loss_bound = tf.reduce_mean(tf.square(self.u_b_left_tf - self.u_b_left_pred)) + \
                    tf.reduce_mean(tf.square(self.u_b_bottom_tf - self.u_b_bottom_pred)) + \
                    tf.reduce_mean(tf.square(self.u_b_top_tf - self.u_b_top_pred)) + \
                    tf.reduce_mean(tf.square(self.u_b_right_tf - self.u_b_right_pred)) + \
                    tf.reduce_mean(tf.square(self.u_b_crack_tf - self.u_b_crack_pred))
        self.loss_int = tf.reduce_mean(tf.square(self.f_u_pred))
        
        #multiply by a penalty factor for the boundary term 
        self.loss_bound = 10*self.loss_bound
        
        self.loss = self.loss_bound + self.loss_int

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self,X,weights,biases):
#       
        num_layers = len(weights) + 1
        
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

        X = tf.concat([x,y],1)

        u = self.neural_net(X,self.weights,self.biases)
        #u = uvphi[:,0:1]
        #v = uvphi[:,1:2]

        return u

    def net_f_u(self,x,y):

        u = self.net_u(x,y)
        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x,x)[0]        
        u_y = tf.gradients(u,y)[0]
        u_yy = tf.gradients(u_y,y)[0]

        f_u = u_xx + u_yy + 1

        return f_u


    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):

        tf_dict = {self.x_b_left_tf: self.x_b_left, self.y_b_left_tf: self.y_b_left, self.u_b_left_tf: self.u_b_left,
                   self.x_b_bottom_tf: self.x_b_bottom, self.y_b_bottom_tf: self.y_b_bottom,
                   self.u_b_bottom_tf: self.u_b_bottom, self.x_b_top_tf: self.x_b_top, self.y_b_top_tf: self.y_b_top,
                   self.x_b_right_tf: self.x_b_right, self.y_b_right_tf: self.y_b_right, self.u_b_right_tf: self.u_b_right,
                   self.x_b_crack_tf: self.x_b_crack, self.y_b_crack_tf: self.y_b_crack, self.u_b_crack_tf: self.u_b_crack,
                   self.u_b_top_tf: self.u_b_top, self.x_f_tf: self.x_f, self.y_f_tf: self.y_f}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bound_value = self.sess.run(self.loss_bound, tf_dict)
                loss_int_value = self.sess.run(self.loss_int, tf_dict)
                print('It: %d, Total Loss: %.3e, Boundary Loss: %.3e, Interior Loss: %.3e, Time: %.2f' %
                      (it, loss_value, loss_bound_value, loss_int_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

       # tf_dict = {self.x_bLeft_tf: X_star[:,0:1], self.y_bLeft_tf: X_star[:,1:2]}
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2]}

       # u_star = self.sess.run(self.u_bLeft_pred, tf_dict)
       # v_star = self.sess.run(self.v_bLeft_pred, tf_dict)
       
        u_star = self.sess.run(self.u_f_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2]}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)

        return u_star, f_u_star


if __name__ == "__main__":
    

    layers = [2, 200, 200, 200, 1] #number of neurons in each layer
    num_train_its = 100000        #number of training iterations
    
    
    # Domain bounds
    LB_Plate = np.array([-1.0,-1.0])  #lower bound of the plate
    UB_Plate = np.array([1.0,1.0])  #upper bound of the plate
    
    N_b = 200 # number of boundary points per edge
    N_f = 875 # number of collocation points

   
    X_f = LB_Plate + (UB_Plate - LB_Plate) * lhs(2, N_f) #Generating collocation points


    #TODO: express these in terms of LB_Plate, UB_Plate
    x_left = -np.ones((N_b,1),dtype = np.float32)
    y_left = np.array([np.linspace(-1.0,1.0,N_b)])
    X_b_left = np.concatenate((x_left, y_left.T), axis=1)
    u_b_left = np.zeros((N_b,1),dtype = np.float32)

    x_bottom = np.array([np.linspace(-1.0, 1.0, N_b)])
    y_bottom = -np.ones((N_b,1),dtype = np.float32)
    X_b_bottom = np.concatenate((x_bottom.T, y_bottom), axis=1)
    u_b_bottom = np.zeros((N_b,1),dtype = np.float32)

    x_top = np.array([np.linspace(-1.0, 1.0, N_b)])
    y_top = np.ones((N_b,1),dtype = np.float32)
    X_b_top = np.concatenate((x_top.T, y_top), axis=1)
    u_b_top = np.zeros((N_b,1),dtype = np.float32)
    
    x_right = np.ones((N_b,1),dtype = np.float32)
    y_right = np.array([np.linspace(-1.0,1.0,N_b)])
    X_b_right = np.concatenate((x_right, y_right.T), axis=1)
    u_b_right = np.zeros((N_b,1),dtype = np.float32)
    
    x_crack = np.array([np.linspace(0, 1.0, int(N_b/2))])
    y_crack = np.zeros((int(N_b/2),1),dtype = np.float32)
    X_b_crack = np.concatenate((x_crack.T, y_crack), axis=1)
    u_b_crack = np.zeros((int(N_b/2),1),dtype = np.float32)
    
    nPred = 81
    xPred = np.linspace(-1.0, 1.0, nPred)
    yPred = np.linspace(-1.0, 1.0, nPred)
    xGrid, yGrid = np.meshgrid(xPred, yPred)
    xGrid = np.array([xGrid.flatten()])
    yGrid = np.array([yGrid.flatten()])
    Grid = np.concatenate((xGrid.T,yGrid.T),axis=1)
    np.savetxt('Grid.out', Grid, delimiter=',')
    
    model = PhysicsInformedNN(X_b_left, u_b_left, X_b_bottom, u_b_bottom,  X_b_top, u_b_top, X_b_right, u_b_right, X_b_crack, u_b_crack, X_f, layers, LB_Plate, UB_Plate)

    start_time = time.time()
    model.train(num_train_its)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, f_uPred = model.predict(Grid)
    #print X_star
    #np.savetxt('inp.out', X_star, delimiter=',')
    #print u_pred
    np.savetxt('uPred.out', u_pred, delimiter=',')
    np.savetxt('f_uPred.out', f_uPred, delimiter=',')
    
    #print(u_pred)
    #print(len(u_pred))
    
    #plot the u displacements
    u_pred = np.resize(u_pred, [nPred, nPred])
    CS = plt.contour(xPred, yPred, u_pred,15, linewidths=0.5, colors='k')
    CS = plt.contourf(xPred, yPred, u_pred, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    plt.title('u')
    plt.show()
    
    
    #print f_u_pred


