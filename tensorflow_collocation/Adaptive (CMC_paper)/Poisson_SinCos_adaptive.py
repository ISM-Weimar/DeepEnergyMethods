'''
Implement a Poisson 2D problem with Dirichlet and Neumann boundary conditions:
    - \Delta u(x,y) = f(x,y) for (x,y) \in \Omega:= (0,1)x(0,1)
    u(x,y) = 0, for x = 0
    du/dy = 0 for y = 0, y = 1
    du/dx = k*pi*cos(k*pi*x)*cos(k*pi*y) for x = 1
    
    Exact solution: u(x,y) = sin(k*pi*x)*cos(k*pi*y) corresponding to 
    f(x,y) = 2*k^2*pi^2*sin(k*pi*x)*cos(k*pi*y)
'''

import tensorflow as tf
import numpy as np
#import sys
#print(sys.path)
from utils.PoissonEqAdapt import PoissonEquationColl
from utils.Geometry import QuadrilateralGeom
import matplotlib.pyplot as plt
import time

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

print("Initializing domain...")
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

#problem parameters
alpha = 0
k = 2

#model paramaters
layers = [2, 10, 1] #number of neurons in each layer
num_train_its = 10000        #number of training iterations
data_type = tf.float64
pen_dir = 1
pen_neu = 1

numIter = 3

numPts = 21
numBndPts = 2*numPts
numIntPtsX = numPts
numIntPtsY = numPts

#generate points 
domainCorners = np.array([[0,0],[1,0],[1,1],[0,1]])
domainGeom = QuadrilateralGeom(domainCorners)
dirichlet_left_x, dirichlet_left_y, _, _ = domainGeom.getLeftPts(numBndPts)
neumann_bottom_x, neumann_bottom_y, normal_bottom_x, normal_bottom_y = domainGeom.getBottomPts(numBndPts)
neumann_top_x, neumann_top_y, normal_top_x, normal_top_y = domainGeom.getTopPts(numBndPts)
neumann_right_x, neumann_right_y, normal_right_x, normal_right_y = domainGeom.getRightPts(numBndPts)
interior_x, interior_y = domainGeom.getUnifIntPts(numIntPtsX, numIntPtsY, [0,0,0,0])
interior_x_flat = np.ndarray.flatten(interior_x)[np.newaxis]
interior_y_flat = np.ndarray.flatten(interior_y)[np.newaxis]

#generate boundary values
dirichlet_left_u = np.zeros((numBndPts,1))
neumann_bottom_flux = np.zeros((numBndPts,1))
neumann_top_flux = np.zeros((numBndPts,1))
neumann_right_flux = k*np.pi*np.cos(k*np.pi*neumann_right_x)*np.cos(k*np.pi*neumann_right_y)

#generate interior values (f(x,y))
f_val = 2*k**2*np.pi**2*np.sin(k*np.pi*interior_x_flat)*np.cos(k*np.pi*interior_y_flat)

#combine points
dirichlet_bnd = np.concatenate((dirichlet_left_x, dirichlet_left_y, dirichlet_left_u), axis=1)
neumann_bottom_bnd = np.concatenate((neumann_bottom_x, neumann_bottom_y,
                                     normal_bottom_x, normal_bottom_y, neumann_bottom_flux), axis=1)
neumann_top_bnd = np.concatenate((neumann_top_x, neumann_top_y,
                                  normal_top_x, normal_top_y, neumann_top_flux), axis=1)
neumann_right_bnd = np.concatenate((neumann_right_x, neumann_right_y,
                                    normal_right_x, normal_right_y, neumann_right_flux), axis=1)
neumann_bnd = np.concatenate((neumann_bottom_bnd, neumann_top_bnd, neumann_right_bnd), axis=0)
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
    
    #generate points for plotting the results
    print('Evaluating model...')
    numPredPtsX = 2*numIntPtsX
    numPredPtsY = 2*numIntPtsY
    pred_x, pred_y = domainGeom.getUnifIntPts(numPredPtsX, numPredPtsY, [1,1,1,1])
    pred_x_flat = np.ndarray.flatten(pred_x)[np.newaxis]
    pred_y_flat = np.ndarray.flatten(pred_y)[np.newaxis]
    pred_X = np.concatenate((pred_x_flat.T, pred_y_flat.T), axis=1)
    u_pred, f_pred = model.predict(pred_X)
    
    #define exact solution
    u_exact = np.sin(k*np.pi*pred_x_flat.T)*np.cos(k*np.pi*pred_y_flat.T)
    
    u_pred_err = u_exact-u_pred
    error_u = (np.linalg.norm(u_pred_err,2)/np.linalg.norm(u_exact,2))
    print('Relative error u: %e' % (error_u))       
    
    #    #plot the solution u_comp
    print('$u_{comp}$')
    u_pred = np.resize(u_pred, [numPredPtsY, numPredPtsX])
    CS = plt.contourf(pred_x, pred_y, u_pred, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    #plt.title('$u_{comp}$')
    plt.show()
    
      #plot the error u_ex 
    print('$u_{ex}$')
    u_exact = np.resize(u_exact, [numPredPtsY, numPredPtsX])
    plt.contourf(pred_x, pred_y, u_exact, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    #plt.title('$u_{ex}$')
    plt.show()
    
     #plot the error u_ex - u_comp
    print('u_ex - u_comp')
    u_pred_err = np.resize(u_pred_err, [numPredPtsY, numPredPtsX])
    plt.contourf(pred_x, pred_y, u_pred_err, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    #plt.title('$u_{ex}-u_{comp}$')
    plt.show()
    
    #
    print('Loss convergence')    
    range_adam = np.arange(1,num_train_its+1)
    range_lbfgs = np.arange(num_train_its+2, num_train_its+2+len(model.lbfgs_buffer))
    ax0, = plt.semilogy(range_adam, model.loss_adam_buff, label='Adam')
    ax1, = plt.semilogy(range_lbfgs,  model.lbfgs_buffer, label='L-BFGS')
    plt.legend(handles=[ax0,ax1])
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.show()
        
    #generate interior points for evaluating the model
    numPredPtsX = 2*numIntPtsX-1
    numPredPtsY = 2*numIntPtsY-1
    pred_int_x, pred_int_y = domainGeom.getUnifIntPts(numPredPtsX, numPredPtsY, [0,0,0,0])
    pred_int_x_flat = np.ndarray.flatten(pred_int_x)[np.newaxis]
    pred_int_y_flat = np.ndarray.flatten(pred_int_y)[np.newaxis]
    pred_X = np.concatenate((pred_int_x_flat.T, pred_int_y_flat.T), axis=1)
    u_pred, f_pred = model.predict(pred_X)
    
    f_val = 2*k**2*np.pi**2*np.sin(k*np.pi*pred_int_x_flat.T)*np.cos(k*np.pi*pred_int_y_flat.T)
    f_err = f_val - f_pred
    f_err_rel = (np.linalg.norm(f_err,2)/np.linalg.norm(f_val,2))
    
    rel_err[i] = error_u
    rel_est_err[i] = f_err_rel
    numPts[i] = len(X_int)
    
    print('Estimated relative error f_int-f_pred: %e' %  (f_err_rel))
    print('f_int - f_pred')
    f_err_plt = np.resize(f_err, [numPredPtsY-2, numPredPtsX-2])
    plt.contourf(pred_int_x, pred_int_y, f_err_plt, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    plt.show()
    
    #pick the top N percent interior points with highest error
    N = 30
    ntop =  np.int(np.round(len(f_err)*N/100))
    index_f_err = np.argsort(-np.abs(f_err),axis=0)
    top_pred_xy = np.squeeze(pred_X[index_f_err[0:ntop-1,:]])
    
    #generate interior values (f(x,y))
    top_pred_val = np.squeeze(f_val[index_f_err[0:ntop-1,:]], axis=2)
    top_pred_X = np.concatenate((top_pred_xy, top_pred_val), axis=1)
    print('Num new points:', len(top_pred_X))
    
    
    
    
print(rel_err)
print(rel_est_err)
print(numPts)
print('Num parameters: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
