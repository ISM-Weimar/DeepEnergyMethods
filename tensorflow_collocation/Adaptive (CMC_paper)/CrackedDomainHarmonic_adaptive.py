'''
Implement a Poisson 2D problem on a cracked domain with pure Dirichlet boundary conditions:
    - \Delta u(x,y) = f(x,y) for (x,y) \in \Omega:= (-1,1)x(-1,1) \ (0,1)x{0}
    u(r,\theta) = r^(1/2)*sin(\theta/2), for (r,\theta) \in \partial \Omega (with polar coordinates)
    f(x,y) = 0
    Problem from: Weinan E and Bing Yu - The Deep Ritz method: A deep learning-based
    numerical algorithm for solving variational problems, Section 3.1
    Use adaptivity
    
'''


import tensorflow as tf
import numpy as np

from utils.PoissonEqAdapt import PoissonEquationColl
from utils.Geometry import QuadrilateralGeom

#make figures bigger on HiDPI monitors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import matplotlib.pyplot as plt

import time

print("Initializing domain...")
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

#problem parameters
alpha = 0


#model paramaters
layers = [2, 30, 30, 30, 30, 1] #number of neurons in each layer
num_train_its = 10000        #number of training iterations
data_type = tf.float32
pen_dir = 500
pen_neu = 0


numIter = 3
numBndPts = 81
numIntPtsX = 21
numIntPtsY = 21
   
#generate points 
domainSECorners = np.array([[0,-1],[1,-1],[1,0],[0,0]])
domainNECorners = np.array([[0,0],[1,0],[1,1],[0,1]])
domainWCorners = np.array([[-1,-1],[0,-1],[0,1],[-1,1]])

domainSEGeom = QuadrilateralGeom(domainSECorners)
domainNEGeom = QuadrilateralGeom(domainNECorners)
domainWGeom = QuadrilateralGeom(domainWCorners)

dirichlet_bottom_e_x, dirichlet_bottom_e_y, _, _ = domainSEGeom.getBottomPts(numBndPts)
dirichlet_right_s_x, dirichlet_right_s_y, _, _ = domainSEGeom.getRightPts(numBndPts)
dirichlet_crack_x, dirichlet_crack_y, _, _ = domainNEGeom.getBottomPts(numBndPts)
dirichlet_right_n_x, dirichlet_right_n_y, _, _ = domainNEGeom.getRightPts(numBndPts)
dirichlet_top_e_x, dirichlet_top_e_y, _, _, = domainNEGeom.getTopPts(numBndPts)
dirichlet_bottom_w_x, dirichlet_bottom_w_y, _, _ = domainWGeom.getBottomPts(numBndPts)
dirichlet_top_w_x, dirichlet_top_w_y, _, _ = domainWGeom.getTopPts(numBndPts)
dirichlet_left_x, dirichlet_left_y, _, _ = domainWGeom.getLeftPts(2*numBndPts)

interior_se_x, interior_se_y = domainSEGeom.getUnifIntPts(numIntPtsX, numIntPtsY, [0,0,0,1])
interior_ne_x, interior_ne_y = domainNEGeom.getUnifIntPts(numIntPtsX, numIntPtsY, [0,0,0,1])
interior_w_x, interior_w_y = domainWGeom.getUnifIntPts(numIntPtsX, 2*numIntPtsY, [0,0,0,0])

def compExSol(x,y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    t = np.where(t<0, t+2*np.pi, t)
    u = np.sqrt(r)*np.sin(t/2)
    return u

#generate boundary values
dirichlet_bottom_e_u = compExSol(dirichlet_bottom_e_x, dirichlet_bottom_e_y)
dirichlet_right_s_u = compExSol(dirichlet_right_s_x, dirichlet_right_s_y)
dirichlet_crack_u = compExSol(dirichlet_crack_x, dirichlet_crack_y)
dirichlet_right_n_u = compExSol(dirichlet_right_n_x, dirichlet_right_n_y)
dirichlet_top_e_u = compExSol(dirichlet_top_e_x, dirichlet_top_e_y)
dirichlet_bottom_w_u = compExSol(dirichlet_bottom_w_x, dirichlet_bottom_w_y)
dirichlet_top_w_u = compExSol(dirichlet_top_w_x, dirichlet_top_w_y)
dirichlet_left_u = compExSol(dirichlet_left_x, dirichlet_left_y)

#combine points
dirichlet_bottom_e_bnd = np.concatenate((dirichlet_bottom_e_x, dirichlet_bottom_e_y, 
                                         dirichlet_bottom_e_u), axis=1)
dirichlet_right_s_bnd = np.concatenate((dirichlet_right_s_x, dirichlet_right_s_y, 
                                         dirichlet_right_s_u), axis=1)
dirichlet_crack_bnd = np.concatenate((dirichlet_crack_x, dirichlet_crack_y, 
                                         dirichlet_crack_u), axis=1)
dirichlet_right_n_bnd = np.concatenate((dirichlet_right_n_x, dirichlet_right_n_y, 
                                         dirichlet_right_n_u), axis=1)
dirichlet_top_e_bnd = np.concatenate((dirichlet_top_e_x, dirichlet_top_e_y, 
                                         dirichlet_top_e_u), axis=1)
dirichlet_bottom_w_bnd = np.concatenate((dirichlet_bottom_w_x, dirichlet_bottom_w_y, 
                                         dirichlet_bottom_w_u), axis=1)
dirichlet_top_w_bnd = np.concatenate((dirichlet_top_w_x, dirichlet_top_w_y, 
                                         dirichlet_top_w_u), axis=1)
dirichlet_left_bnd = np.concatenate((dirichlet_left_x, dirichlet_left_y, 
                                         dirichlet_left_u), axis=1)

dirichlet_bnd = np.concatenate((dirichlet_bottom_e_bnd, dirichlet_right_s_bnd, 
                                dirichlet_crack_bnd, dirichlet_right_n_bnd,
                                dirichlet_top_e_bnd, dirichlet_bottom_w_bnd, 
                                dirichlet_top_w_bnd, dirichlet_left_bnd), axis=0)

neumann_bnd = np.zeros((1,5))

interior_se_x_flat = np.ndarray.flatten(interior_se_x)[np.newaxis]
interior_se_y_flat = np.ndarray.flatten(interior_se_y)[np.newaxis]
interior_ne_x_flat = np.ndarray.flatten(interior_ne_x)[np.newaxis]
interior_ne_y_flat = np.ndarray.flatten(interior_ne_y)[np.newaxis]
interior_w_x_flat = np.ndarray.flatten(interior_w_x)[np.newaxis]
interior_w_y_flat = np.ndarray.flatten(interior_w_y)[np.newaxis]

interior_x_flat = np.concatenate((interior_se_x_flat, interior_ne_x_flat, interior_w_x_flat), axis=1)
interior_y_flat = np.concatenate((interior_se_y_flat, interior_ne_y_flat, interior_w_y_flat), axis=1)

#generate interior values (f(x,y))
f_val = np.zeros_like(interior_x_flat)
X_int = np.concatenate((interior_x_flat.T, interior_y_flat.T, f_val.T), axis=1)
top_pred_X = np.zeros([0,3])

#adaptivity loop

rel_err = np.zeros(numIter)
rel_est_err = np.zeros(numIter)
numPts = np.zeros(numIter)

print('Defining model...')
model = PoissonEquationColl(dirichlet_bnd, neumann_bnd, alpha, layers, data_type, pen_dir, pen_neu)

for i in range(numIter):
    
    #training part    
    X_int = np.concatenate((X_int, top_pred_X))
    
    print('Domain geometry')
    plt.scatter(neumann_bnd[:,0], neumann_bnd[:,1],s=0.5,c='g')
    plt.scatter(dirichlet_bnd[:,0], dirichlet_bnd[:,1],s=0.5,c='r')
    plt.scatter(X_int[:,0], X_int[:,1], s=0.5, c='b')
    plt.show()

    start_time = time.time()
    print('Starting training...')
    model.train(X_int, num_train_its)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    #generate points for evaluating the model
    print('Evaluating model...')
    numPredPtsX = 2*numIntPtsX
    numPredPtsY = 2*numIntPtsY
    
    pred_interior_se_x, pred_interior_se_y = domainSEGeom.getUnifIntPts(numPredPtsX, numPredPtsY, [1,1,1,1])
    pred_interior_ne_x, pred_interior_ne_y = domainNEGeom.getUnifIntPts(numPredPtsX, numPredPtsY, [1,1,1,1])
    pred_interior_w_x, pred_interior_w_y = domainWGeom.getUnifIntPts(numPredPtsX, 2*numPredPtsY, [1,1,1,1])
    
    pred_interior_se_x_flat = np.ndarray.flatten(pred_interior_se_x)[np.newaxis]
    pred_interior_se_y_flat = np.ndarray.flatten(pred_interior_se_y)[np.newaxis]
    pred_interior_ne_x_flat = np.ndarray.flatten(pred_interior_ne_x)[np.newaxis]
    pred_interior_ne_y_flat = np.ndarray.flatten(pred_interior_ne_y)[np.newaxis]
    pred_interior_w_x_flat = np.ndarray.flatten(pred_interior_w_x)[np.newaxis]
    pred_interior_w_y_flat = np.ndarray.flatten(pred_interior_w_y)[np.newaxis]
    
    
    pred_se_X = np.concatenate((pred_interior_se_x_flat.T, pred_interior_se_y_flat.T), axis=1)
    pred_ne_X = np.concatenate((pred_interior_ne_x_flat.T, pred_interior_ne_y_flat.T), axis=1)
    pred_w_X = np.concatenate((pred_interior_w_x_flat.T, pred_interior_w_y_flat.T), axis=1)
    u_pred_se, f_pred_se = model.predict(pred_se_X)
    u_pred_ne, f_pred_ne = model.predict(pred_ne_X)
    u_pred_w, f_pred_w = model.predict(pred_w_X)
    
    u_pred = np.concatenate((u_pred_se, u_pred_ne, u_pred_w), axis=0)
    f_pred = np.concatenate((f_pred_se, f_pred_ne, f_pred_w), axis=0)
    #define exact solution
    u_exact_se = compExSol(pred_interior_se_x_flat.T, pred_interior_se_y_flat.T)
    u_exact_ne = compExSol(pred_interior_ne_x_flat.T, pred_interior_ne_y_flat.T)
    u_exact_w = compExSol(pred_interior_w_x_flat.T, pred_interior_w_y_flat.T)
    u_exact = np.concatenate((u_exact_se, u_exact_ne, u_exact_w), axis=0)
    
    u_pred_err = u_exact-u_pred
    error_u = (np.linalg.norm(u_exact-u_pred,2)/np.linalg.norm(u_exact,2))
    print('Relative error u: %e' % (error_u))       
    
    
    def plotSol(pred_interior_se_x, pred_interior_ne_x, pred_interior_w_x,
                pred_interior_se_y, pred_interior_ne_y, pred_interior_w_y,
                val_se, val_ne, val_w, numPredPtsX, numPredPtsY):
        
        min_val = min(min(val_se), min(val_ne), min(val_w))[0]
        max_val = max(max(val_se), max(val_ne), max(val_w))[0]
        
        val_se = np.resize(val_se, [numPredPtsY, numPredPtsX])
        val_ne = np.resize(val_ne, [numPredPtsY, numPredPtsX])
        val_w = np.resize(val_w, [2*numPredPtsY, numPredPtsX])
        
        plt.contourf(pred_interior_se_x, pred_interior_se_y, val_se, 255, vmin=min_val, vmax=max_val, cmap=plt.cm.jet)
        plt.contourf(pred_interior_ne_x, pred_interior_ne_y, val_ne, 255, vmin=min_val, vmax=max_val, cmap=plt.cm.jet)
        plt.contourf(pred_interior_w_x, pred_interior_w_y, val_w, 255, vmin=min_val, vmax=max_val, cmap=plt.cm.jet)    
        plt.colorbar() # draw colorbar    
        plt.show()
    
    #   plot the solution u_comp
    print('$u_{comp}$')
    plotSol(pred_interior_se_x, pred_interior_ne_x, pred_interior_w_x,
            pred_interior_se_y, pred_interior_ne_y, pred_interior_w_y,
            u_pred_se, u_pred_ne, u_pred_w, numPredPtsX, numPredPtsY)
    
    #  plot the exact solution
    print('$u_{ex}$')
    plotSol(pred_interior_se_x, pred_interior_ne_x, pred_interior_w_x,
            pred_interior_se_y, pred_interior_ne_y, pred_interior_w_y,
            u_exact_se, u_exact_ne, u_exact_w, numPredPtsX, numPredPtsY)
    
    #  plot the error u_ex - u_comp
    u_err_se = u_exact_se - u_pred_se
    u_err_ne = u_exact_ne - u_pred_ne
    u_err_w = u_exact_w - u_pred_w
    #print('u_ex - u_comp')
    plotSol(pred_interior_se_x, pred_interior_ne_x, pred_interior_w_x,
            pred_interior_se_y, pred_interior_ne_y, pred_interior_w_y,
            u_err_se, u_err_ne, u_err_w, numPredPtsX, numPredPtsY)
    
    print('Loss convergence')    
    range_adam = np.arange(1,num_train_its+1)
    range_lbfgs = np.arange(num_train_its+2, num_train_its+2+len(model.lbfgs_buffer))
    ax0, = plt.semilogy(range_adam, model.loss_adam_buff, label='Adam')
    ax1, = plt.semilogy(range_lbfgs,  model.lbfgs_buffer, label='L-BFGS')
    plt.legend(handles=[ax0,ax1])
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.show()
    
    
    print('Residual f_pred')
    plotSol(pred_interior_se_x, pred_interior_ne_x, pred_interior_w_x,
            pred_interior_se_y, pred_interior_ne_y, pred_interior_w_y,
            f_pred_se, f_pred_ne, f_pred_w, numPredPtsX, numPredPtsY)
    
    #generate interior points for evaluating the model
    numPredPtsX = (i+2)*numIntPtsX
    numPredPtsY = (i+2)*numIntPtsY
    
    pred_interior_se_x, pred_interior_se_y = domainSEGeom.getUnifIntPts(numPredPtsX, numPredPtsY, [0,0,0,1])
    pred_interior_ne_x, pred_interior_ne_y = domainNEGeom.getUnifIntPts(numPredPtsX, numPredPtsY, [0,0,0,1])
    pred_interior_w_x, pred_interior_w_y = domainWGeom.getUnifIntPts(numPredPtsX, 2*numPredPtsY, [0,0,0,0])
    
    pred_interior_se_x_flat = np.ndarray.flatten(pred_interior_se_x)[np.newaxis]
    pred_interior_se_y_flat = np.ndarray.flatten(pred_interior_se_y)[np.newaxis]
    pred_interior_ne_x_flat = np.ndarray.flatten(pred_interior_ne_x)[np.newaxis]
    pred_interior_ne_y_flat = np.ndarray.flatten(pred_interior_ne_y)[np.newaxis]
    pred_interior_w_x_flat = np.ndarray.flatten(pred_interior_w_x)[np.newaxis]
    pred_interior_w_y_flat = np.ndarray.flatten(pred_interior_w_y)[np.newaxis]
    
    pred_se_X = np.concatenate((pred_interior_se_x_flat.T, pred_interior_se_y_flat.T), axis=1)
    pred_ne_X = np.concatenate((pred_interior_ne_x_flat.T, pred_interior_ne_y_flat.T), axis=1)
    pred_w_X = np.concatenate((pred_interior_w_x_flat.T, pred_interior_w_y_flat.T), axis=1)
    _, f_pred_se = model.predict(pred_se_X)
    _, f_pred_ne  = model.predict(pred_ne_X)
    _, f_pred_w = model.predict(pred_w_X)
    
    f_pred = np.concatenate((f_pred_se, f_pred_ne, f_pred_w), axis=0)
    
    f_val = np.zeros_like(f_pred)
    f_err = f_val - f_pred
    f_err_rel = (np.linalg.norm(f_err,2))
    
    rel_err[i] = error_u
    rel_est_err[i] = f_err_rel
    numPts[i] = len(X_int)
    
    print('Estimated error f_int-f_pred: %e' %  (f_err_rel))
    
    #pick the top N percent interior points with highest error
    N = 15
    pred_X = np.concatenate((pred_se_X, pred_ne_X, pred_w_X), axis=0)
    ntop =  np.int(np.round(len(f_err)*N/100))
    index_f_err = np.argsort(-np.abs(f_err),axis=0)
    top_pred_xy = np.squeeze(pred_X[index_f_err[0:ntop-1,:]])
    
    #generate interior values (f(x,y))
    top_pred_val = np.squeeze(f_val[index_f_err[0:ntop-1,:]], axis=2)
    top_pred_X = np.concatenate((top_pred_xy, top_pred_val), axis=1)
    
