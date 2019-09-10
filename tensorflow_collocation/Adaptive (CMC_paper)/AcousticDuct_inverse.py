'''
Implement a Helmholtz 2D problem for the acoustic duct:
    \Delta u(x,y) +k^2u(x,y) = 0 for (x,y) \in \Omega:= (0,2)x(0,1)
    \partial u / \partial n = cos(m*pi*x), for x = 0;
    \partial u / \partial n = -iku, for x = 2;
    \partial u / \partial n = 0, for y=0 and y=1
    
    
    Exact solution: u(x,y) = cos(m*pi*y)*(A_1*exp(-i*k_x*x) + A_2*exp(i*k_x*x))corresponding to 
    where A_1 and A_2 are obtained by solving the 2x2 linear system:
    [i*k_x                               -i*k_x      ] [A_1]  = [1]
    [(k-k_x)*exp(-2*i*k_x)       (k+k_x)*exp(2*i*k_x)] [A_2]    [0]
    
    Writes output for TensorBoard
    Uniformly spaced points
    Inverse problem where we solve for k from the exact solution
    
'''

import tensorflow as tf
import numpy as np

#make figures bigger on HiDPI monitors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import matplotlib.pyplot as plt

import time
from datetime import datetime

tf.reset_default_graph()   # To clear the defined variables and operations of the previous run
np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_b_left, u_x_b_left, X_b_bottom, u_y_b_bottom, X_b_top, u_y_b_top, X_b_right, u_x_b_right, X_f, T_f, layers, lb, ub):
      
        self.lb = lb
        self.ub = ub
        


        self.x_b_left = X_b_left[:,0:1]
        self.y_b_left = X_b_left[:,1:2]
        self.u_x_b_left = u_x_b_left

        self.x_b_bottom = X_b_bottom[:,0:1]
        self.y_b_bottom = X_b_bottom[:,1:2]
        self.u_y_b_bottom = u_y_b_bottom

        self.x_b_top = X_b_top[:,0:1]
        self.y_b_top = X_b_top[:,1:2]
        self.u_y_b_top = u_y_b_top
        
        self.x_b_right = X_b_right[:,0:1]
        self.y_b_right = X_b_right[:,1:2]
        self.u_x_b_right = u_x_b_right


        self.x_f = X_f[:,0:1]
        self.y_f = X_f[:,1:2]
        self.t_f = T_f
        

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.x_b_left_tf = tf.placeholder(tf.float64, shape=[None, self.x_b_left.shape[1]])
        self.y_b_left_tf = tf.placeholder(tf.float64, shape=[None, self.y_b_left.shape[1]])
        self.u_x_b_left_tf = tf.placeholder(tf.float64, shape=[None, self.u_x_b_left.shape[1]])
        self.x_b_bottom_tf = tf.placeholder(tf.float64, shape=[None, self.x_b_bottom.shape[1]])
        self.y_b_bottom_tf = tf.placeholder(tf.float64, shape=[None, self.y_b_bottom.shape[1]])
        self.u_y_b_bottom_tf = tf.placeholder(tf.float64, shape=[None, self.u_y_b_bottom.shape[1]])
        self.x_b_top_tf = tf.placeholder(tf.float64, shape=[None, self.x_b_top.shape[1]])
        self.y_b_top_tf = tf.placeholder(tf.float64, shape=[None, self.y_b_top.shape[1]])
        self.u_y_b_top_tf = tf.placeholder(tf.float64, shape=[None, self.u_y_b_top.shape[1]])
        self.x_b_right_tf = tf.placeholder(tf.float64, shape=[None, self.x_b_right.shape[1]])
        self.y_b_right_tf = tf.placeholder(tf.float64, shape=[None, self.y_b_right.shape[1]])
        self.u_x_b_right_tf = tf.placeholder(tf.float64, shape=[None, self.u_x_b_right.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float64, shape=[None, self.y_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])


        self.k = tf.Variable([1.0], dtype=tf.float64)


        # tf Graphs
         # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        
        
        self.u_x_b_left_pred = self.net_u_x(self.x_b_left_tf,self.y_b_left_tf)
        #self.u_int_pred = self.net_u(self.x_f_tf,self.y_f_tf)
        self.u_y_b_bottom_pred = self.net_u_y(self.x_b_bottom_tf, self.y_b_bottom_tf)
        self.u_y_b_top_pred = self.net_u_y(self.x_b_top_tf,self.y_b_top_tf)
        self.u_x_b_right_pred = self.net_u_x(self.x_b_right_tf,self.y_b_right_tf)
        self.u_int_pred, self.f_u_pred = self.net_f_u(self.x_f_tf,self.y_f_tf)
        #self.u_int_pred = self.net_u(self.x_f_tf,self.y_f_tf)

        # Loss        
        self.loss_bound = tf.reduce_mean(tf.square(self.u_x_b_left_tf - self.u_x_b_left_pred)) + \
                   tf.reduce_mean(tf.square(self.u_y_b_bottom_tf - self.u_y_b_bottom_pred)) + \
                   tf.reduce_mean(tf.square(self.u_y_b_top_tf - self.u_y_b_top_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_b_right_tf - self.u_x_b_right_pred))
        self.loss_int = tf.reduce_mean(tf.square(self.u_int_pred-self.t_f_tf)) + tf.reduce_mean(tf.square(self.f_u_pred))
        
        #apply penalty term
        self.loss_bound = 100*self.loss_bound
        #self.loss = self.loss_bound + self.loss_int
        self.loss = self.loss_int
                    

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1e-10})


    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

       
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    def neural_net(self,X,weights,biases):
#       
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        #H = tf.sigmoid(tf.add(tf.matmul(H, weights[0]), biases[0]))
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            #H = tf.add(tf.matmul(H, W), b)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self,x,y):

        X = tf.concat([x,y],1)

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
    

    def net_f_u(self,x,y):
        k = self.k
        X = tf.concat([x,y],1)
        u = self.neural_net(X,self.weights,self.biases)
        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x,x)[0]        
        u_y = tf.gradients(u,y)[0]
        u_yy = tf.gradients(u_y,y)[0]

        f_u = (u_xx + u_yy) + (k**2)*u

        return u, f_u



    def callback(self, loss, k):
        print('Loss: %.3e, k: %.3f' % (loss, k))

    def train(self, nIter):

        tf_dict = {self.x_b_left_tf: self.x_b_left, self.y_b_left_tf: self.y_b_left, self.u_x_b_left_tf: self.u_x_b_left,
                   self.x_b_bottom_tf: self.x_b_bottom, self.y_b_bottom_tf: self.y_b_bottom,
                   self.u_y_b_bottom_tf: self.u_y_b_bottom, self.x_b_top_tf: self.x_b_top, self.y_b_top_tf: self.y_b_top,
                   self.x_b_right_tf: self.x_b_right, self.y_b_right_tf: self.y_b_right, self.u_x_b_right_tf: self.u_x_b_right,
                   self.u_y_b_top_tf: self.u_y_b_top, self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.t_f_tf: self.t_f}

        start_time = time.time()

        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bound_value = self.sess.run(self.loss_bound, tf_dict)
                loss_int_value = self.sess.run(self.loss_int, tf_dict)
                k_value = self.sess.run(self.k)
                
                print('It: %d, Total Loss: %.3e, Boundary Loss: %.3e, Interior Loss: %.3e, k value: %.9e, Time: %.2f' %
                      (it, loss_value, loss_bound_value, loss_int_value, k_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.k],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2]}
       
        u_star = self.sess.run(self.u_int_pred, tf_dict)

        f_u_star = self.sess.run(self.f_u_pred, tf_dict)

        return u_star, f_u_star


if __name__ == "__main__":
    N_b = 800 # number of boundary points
    #N_f = 5000 # number of collocation points

    #layers = [2, 20, 20, 20, 1] #number of neurons in each layer
    num_train_its = 5000        #number of training iterations
    

    layers = [2, 30, 30, 30, 30, 30, 1]
    LB_Plate = np.array([0.0,0.0])
    UB_Plate = np.array([2.0,1.0])
    #X_f = LB_Plate + (UB_Plate - LB_Plate) * lhs(2, N_f) #Generating collocation points

    # create points
    # define domain and collocation points
    numPtsX = 151
    numPtsY = 31
    Length = UB_Plate[0]-LB_Plate[0]
    Width = UB_Plate[1]-LB_Plate[1]
    offsetX = Length/(numPtsX-1)
    offsetY = Width/(numPtsY-1)
    x_dom = offsetX, Length-offsetX, numPtsX-2
    y_dom = offsetY, Width-offsetY, numPtsY-2
        
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    X_f = np.zeros((x_dom[2]*y_dom[2],2))
    c = 0
    for x in np.nditer(lin_x):
        tb = y_dom[2]*c
        te = tb + y_dom[2]
        c += 1
        X_f[tb:te,0] = x
        X_f[tb:te,1] = lin_y


    #define model parameters
    m = 1; #mode number
    k = 4; #wave number
    
    #solve for the constants A1 and A2 in the exact solution
    kx = np.sqrt(k**2 - (m*np.pi)**2);
    alpha = -k**2;

    LHS = np.array([[1j*kx, -1j*kx], [(k-kx)*np.exp(-2*1j*kx), (k+kx)*np.exp(2*1j*kx)]])
    RHS = np.array([[1],[0]])
    
    A = np.linalg.solve(LHS, RHS)
    
    T_f = np.real(np.cos(m*np.pi*X_f[:,1])*(A[0]*np.exp(-1j*kx*X_f[:,0])+A[1]*np.exp(1j*kx*X_f[:,0])))[np.newaxis].T

    

    #TODO: express these in terms of LB_Plate, UB_Plate
    x_left = np.zeros((int(N_b/4),1),dtype = np.float64)
    y_left = np.array([np.linspace(0.0,1.0,int(N_b/4))])
    X_b_left = np.concatenate((x_left, y_left.T), axis=1)
    u_x_b_left = np.float64(-np.cos(m*np.pi*y_left))
    u_x_b_left = u_x_b_left.T
    

    x_bottom = np.array([np.linspace(0.0, 2.0, int(N_b/4))])
    y_bottom = np.zeros((int(N_b/4),1),dtype = np.float64)
    X_b_bottom = np.concatenate((x_bottom.T, y_bottom), axis=1)
    u_y_b_bottom = np.zeros((int(N_b/4),1),dtype = np.float64)

    x_top = np.array([np.linspace(0.0, 2.0, int(N_b/4))])
    y_top = np.ones((int(N_b/4),1),dtype = np.float64)
    X_b_top = np.concatenate((x_top.T, y_top), axis=1)
    u_y_b_top = np.zeros((int(N_b/4),1),dtype = np.float64)
    
    x_right = 2*np.ones((int(N_b/4),1),dtype = np.float64)
    y_right = np.array([np.linspace(0.0,1.0,int(N_b/4))])
    X_b_right = np.concatenate((x_right, y_right.T), axis=1)
    x_right_scal = 2;
    u_x_b_right = np.float64(np.real(np.cos(m*np.pi*y_right)*(A[0]*(-1j)*kx*np.exp(-1j*kx*x_right_scal)+A[1]*1j*kx*np.exp(1j*kx*x_right_scal))))
    u_x_b_right = u_x_b_right.T
        
    nPred = 160
    xPred = np.linspace(0.0, 2.0, nPred)
    yPred = np.linspace(0.0, 1.0, nPred)
    xGrid, yGrid = np.meshgrid(xPred, yPred)
    xGrid = np.array([xGrid.flatten()])
    yGrid = np.array([yGrid.flatten()])
    Grid = np.concatenate((xGrid.T,yGrid.T),axis=1)    
        
    x_grid = Grid[:,0:1]
    y_grid = Grid[:,1:2]
    
    print('Defining model...')
    model = PhysicsInformedNN(X_b_left, u_x_b_left, X_b_bottom, u_y_b_bottom,  X_b_top, u_y_b_top, X_b_right, u_x_b_right, X_f, T_f, layers,  LB_Plate, UB_Plate)

    start_time = time.time()
    print('Starting training...')
    model.train(num_train_its)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, f_uPred = model.predict(Grid)    
    u_exact = np.real(np.cos(m*np.pi*y_grid)*(A[0]*np.exp(-1j*kx*x_grid)+A[1]*np.exp(1j*kx*x_grid)))            
    u_pred_err = u_exact-u_pred
    
    error_u = (np.linalg.norm(u_exact-u_pred,2)/np.linalg.norm(u_exact,2))
    print('Relative error u: %e' % (error_u))       
    
#    #plot the computed solution u_comp
    u_pred = np.resize(u_pred, [nPred, nPred])    
    CS = plt.contourf(xPred, yPred, u_pred, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    plt.title('$u_{comp}$')
    plt.show()
    
    #plot the exact solution u_ex 
    u_exact = np.resize(u_exact, [nPred, nPred])
    CS = plt.contourf(xPred, yPred, u_exact, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    plt.title('$u_{ex}$')
    plt.show()
    
#    #plot the error u_ex - u_comp
    u_pred_err = np.resize(u_pred_err, [nPred, nPred])
    CS = plt.contourf(xPred, yPred, u_pred_err, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    plt.title('$u_{ex}-u_{comp}$')
    plt.show()
            