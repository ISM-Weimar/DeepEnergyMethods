'''
Implement a Poisson-type 2D problem with pure Dirichlet boundary conditions on an 
Annulus domain:
    - \Delta u(x,y) + u(x,y) = f(x,y) for (x,y) \in \Omega quarter-annulus with inner 
    radius 1, and outer radius 4
    u(x,y) = 0, for (x,y) \in \partial \Omega
    Exact solution: u(x,y) = (x^2+y^2-1)*(x^2+y^2-16)*sin(x)*sin(y) corresponding to 
    f(x,y) = (3*x^4 - 67*x^2 - 67*y^2 + 3*y^4 + 6*x^2*y^2 + 116)*sin(x)*sin(y)+
    +(68*x - 8*x^3 - 8*x*y^2)* cos(x)*sin(y) + 
    +(68*y - 8*y^3 - 8*y*x^2)* cos(x)*sin(y)
    (defined in net_f_u function)
    See Example 4.4 in https://doi.org/10.1142/S0218202510004878
'''
import tensorflow as tf
import numpy as np
#import sys
#print(sys.path)
from utils.PoissonEqAdapt import PoissonEquationColl
from utils.Geometry import AnnulusGeom
import matplotlib.pyplot as plt
import time

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

print("Initializing domain...")
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

#problem parameters
alpha = 1

#model paramaters
layers = [2, 10, 10, 1] #number of neurons in each layer
num_train_its = 10000        #number of training iterations
data_type = tf.float32
pen_dir = 100
pen_neu = 0
numIter = 3

numPts = 21
numBndPts = 2*numPts
numIntPtsX = numPts
numIntPtsY = numPts

#generate points 
rad_int = 1
rad_ext = 4
domainGeom = AnnulusGeom(rad_int, rad_ext)
dirichlet_inner_x, dirichlet_inner_y, _, _ = domainGeom.getInnerPts(numBndPts)
dirichlet_outer_x, dirichlet_outer_y, _, _ = domainGeom.getOuterPts(numBndPts)
dirichlet_xax_x, dirichlet_xax_y, _, _ = domainGeom.getXAxPts(numBndPts)
dirichlet_yax_x, dirichlet_yax_y, _, _ = domainGeom.getYAxPts(numBndPts)

interior_x, interior_y = domainGeom.getUnifIntPts(numIntPtsX, numIntPtsY, [0,0,0,0])
int_x = np.ndarray.flatten(interior_x)[np.newaxis]
int_y = np.ndarray.flatten(interior_y)[np.newaxis]

#generate boundary values
dirichlet_inner_u = np.zeros((numBndPts,1))
dirichlet_outer_u = np.zeros((numBndPts,1))
dirichlet_xax_u = np.zeros((numBndPts,1))
dirichlet_yax_u = np.zeros((numBndPts,1))
#generate interior values (f(x,y))
f_val = (3*int_x**4 - 67*int_x**2 - 67*int_y**2 + 3*int_y**4 + 6*int_x**2*int_y**2 + 116)*np.sin(int_x)*np.sin(int_y) \
    +(68*int_x - 8*int_x**3 - 8*int_x*int_y**2) * np.cos(int_x)*np.sin(int_y)  \
    +(68*int_y - 8*int_y**3 - 8*int_y*int_x**2) * np.cos(int_y)*np.sin(int_x)


#combine points
dirichlet_inner_bnd = np.concatenate((dirichlet_inner_x, dirichlet_inner_y, dirichlet_inner_u), axis=1)
dirichlet_outer_bnd = np.concatenate((dirichlet_outer_x, dirichlet_outer_y, dirichlet_outer_u), axis=1)
dirichlet_xax_bnd = np.concatenate((dirichlet_xax_x, dirichlet_xax_y, dirichlet_xax_u), axis=1)
dirichlet_yax_bnd = np.concatenate((dirichlet_yax_x, dirichlet_yax_y, dirichlet_yax_u), axis=1)
dirichlet_bnd = np.concatenate((dirichlet_inner_bnd, dirichlet_outer_bnd, 
                               dirichlet_xax_bnd, dirichlet_yax_bnd), axis=0)
#neumann_bnd = np.array([[],[],[],[],[]])
neumann_bnd = np.zeros((1,5))
top_pred_X = np.zeros([0,3])

X_int = np.concatenate((int_x.T, int_y.T, f_val.T), axis=1)

print('Defining model...')
model = PoissonEquationColl(dirichlet_bnd, neumann_bnd, alpha, layers, data_type, pen_dir, pen_neu)


#adaptivity loop

rel_err = np.zeros(numIter)
rel_est_err = np.zeros(numIter)
numPts = np.zeros(numIter)

for i in range(numIter):

    X_int = np.concatenate((X_int, top_pred_X))
    
    
    print('Domain geometry')
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
    pred_interior_x, pred_interior_y = domainGeom.getUnifIntPts(numPredPtsX, numPredPtsY, [1,1,1,1])
    pred_x = np.ndarray.flatten(pred_interior_x)[np.newaxis]
    pred_y = np.ndarray.flatten(pred_interior_y)[np.newaxis]
    pred_X = np.concatenate((pred_x.T, pred_y.T), axis=1)
    u_pred, _ = model.predict(pred_X)
    
    
    u_exact =  (pred_x.T**2+pred_y.T**2-1)*(pred_x.T**2+pred_y.T**2-16)*np.sin(pred_x.T)*np.sin(pred_y.T)
    u_pred_err = u_exact-u_pred
    
    
    error_u = (np.linalg.norm(u_exact-u_pred,2)/np.linalg.norm(u_exact,2))
    print('Relative error u: %e' % (error_u))       
    
    #    #plot the solution u_comp
    print('$u_{comp}$')
    u_pred = np.resize(u_pred, [numPredPtsX, numPredPtsY])
    CS = plt.contourf(pred_interior_x, pred_interior_y, u_pred, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    #plt.title('$u_{comp}$')
    plt.show()
    
      #plot the error u_ex 
    print('$u_{ex}$')
    u_exact = np.resize(u_exact, [numPredPtsX, numPredPtsY])
    plt.contourf(pred_interior_x, pred_interior_y, u_exact, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    #plt.title('$u_{ex}$')
    plt.show()
    
     #plot the error u_ex - u_comp
    print('u_ex - u_comp')
    u_pred_err = np.resize(u_pred_err, [numPredPtsX, numPredPtsY])
    plt.contourf(pred_interior_x, pred_interior_y, u_pred_err, 255, cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    #plt.title('$u_{ex}-u_{comp}$')
    plt.show()
    #
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
    int_x = np.ndarray.flatten(pred_int_x)[np.newaxis]
    int_y = np.ndarray.flatten(pred_int_y)[np.newaxis]
    pred_X = np.concatenate((int_x.T, int_y.T), axis=1)
    u_pred, f_pred = model.predict(pred_X)
    
    f_val = (3*int_x.T**4 - 67*int_x.T**2 - 67*int_y.T**2 + 3*int_y.T**4 + 6*int_x.T**2*int_y.T**2 + 116)*np.sin(int_x.T)*np.sin(int_y.T) \
    +(68*int_x.T - 8*int_x.T**3 - 8*int_x.T*int_y.T**2) * np.cos(int_x.T)*np.sin(int_y.T)  \
    +(68*int_y.T - 8*int_y.T**3 - 8*int_y.T*int_x.T**2) * np.cos(int_y.T)*np.sin(int_x.T)
    
    
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

print(rel_err)
print(rel_est_err)
print(numPts)
    
