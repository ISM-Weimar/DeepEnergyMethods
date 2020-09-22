# -*- coding: utf-8 -*-
'''
Implements the an hollow sphere under pressure problem
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK 

import scipy.io


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

from utils.Geom import Geometry3D
from utils.PINN import Elasticity3D

class HollowSphere(Geometry3D):
    '''
     Class for definining a quarter-annulus domain centered at the orgin
         (the domain is in the first quadrant)
     Input: rad_int, rad_ext - internal and external radii of the annulus
    '''
    def __init__(self, a, b):            
                
        geomData = dict()
        
        
        
        # Set degrees
        geomData['degree_u'] = 2
        geomData['degree_v'] = 2
        geomData['degree_w'] = 1
        
        # Set control points
        geomData['ctrlpts_size_u'] = 3
        geomData['ctrlpts_size_v'] = 3
        geomData['ctrlpts_size_w'] = 2
        
        wgt = 1/np.sqrt(2)
                    
        geomData['ctrlpts'] =  [[a, 0, 0],
                 [wgt*a, 0, wgt*a],
                 [0, 0, a],
                 [wgt*a, wgt*a, 0],
                 [0.5*a, 0.5*a, 0.5*a],
                 [0, 0, wgt*a],
                 [0, a, 0],
                 [0, wgt*a, wgt*a],
                 [0, 0, a],
                 [b, 0, 0],
                 [wgt*b, 0, wgt*b],
                 [0, 0, b],
                 [wgt*b, wgt*b, 0],
                 [0.5*b, 0.5*b, 0.5*b],
                 [0, 0, wgt*b],
                 [0, b, 0],
                 [0, wgt*b, wgt*b],
                 [0, 0, b]]
        
        geomData['weights'] = [1., 
                   wgt, 
                   1, 
                   wgt, 
                   1/2, 
                   wgt, 
                   1, 
                   wgt, 
                   1, 
                   1, 
                   wgt, 
                   1, 
                   wgt, 
                   1/2, 
                   wgt, 
                   1, 
                   wgt, 
                   1]
        
        # Set knot vectors
        geomData['knotvector_u'] = [0., 0., 0., 1., 1., 1.]
        geomData['knotvector_v'] = [0., 0., 0., 1., 1., 1.]
        geomData['knotvector_w'] = [0., 0., 1., 1.]

        super().__init__(geomData)

class PINN_HS(Elasticity3D):
    '''
    Class including (symmetry) boundary conditions for the hollow sphere problem
    '''       
    def net_uvw(self, x, y, z):

        X = tf.concat([x, y, z], 1)      

        uvw = self.neural_net(X, self.weights, self.biases)

        u = x*uvw[:, 0:1]
        v = y*uvw[:, 1:2]
        w = z*uvw[:, 2:3]

        return u, v, w

def cart2sph(x, y, z):
    # From https://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

def sph2cart(azimuth, elevation, r):
    # From https://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z    

def getExactDisplacements(x, y, z, model):
    P = model['P']
    b = model['b']
    a = model['a']
    E = model['E']
    nu = model['nu']
    
    azimuth, elevation, r  = cart2sph(x, y, z)    

    u_r = P*a**3*r/(E*(b**3-a**3))*((1-2*nu)+(1+nu)*b**3/(2*r**3))    
    u_exact, v_exact, w_exact = sph2cart(azimuth, elevation, u_r)

    return u_exact, v_exact, w_exact

def getExactStresses(x, y, z, model):
    numPts = len(x)
    P = model['P']
    b = model['b']
    a = model['a']
    
    sigma_xx = np.zeros_like(x)
    sigma_yy = np.zeros_like(x)
    sigma_zz = np.zeros_like(x)
    sigma_xy = np.zeros_like(x)
    sigma_yz = np.zeros_like(x)
    sigma_zx = np.zeros_like(x)
    
    for i in range(numPts):
    
        azimuth, elevation, r  = cart2sph(x[i], y[i], z[i])    
        
        phi = azimuth
        theta = np.pi/2-elevation
        
        sigma_r = P*a**3*(b**3-r**3)/(r**3*(a**3-b**3))
        sigma_th = P*a**3*(b**3+2*r**3)/(2*r**3*(b**3-a**3))
        
        rot_mat = np.array( \
             [[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
              [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],
              [np.cos(theta), -np.sin(theta), 0.]])
        A = np.array( [[sigma_r, 0., 0.], [0., sigma_th, 0.], [0., 0., sigma_th]] )
        stress_cart = rot_mat@A@rot_mat.T
        
        sigma_xx[i] = stress_cart[0,0]
        sigma_yy[i] = stress_cart[1,1]
        sigma_zz[i] = stress_cart[2,2]
        sigma_xy[i] = stress_cart[0,1]
        sigma_zx[i] = stress_cart[0,2]
        sigma_yz[i] = stress_cart[1,2]

    return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx
    
model_data = dict()
model_data['E'] = 1e3
model_data['nu'] = 0.3

model = dict()
model['E'] = model_data['E']
model['nu'] = model_data['nu']
model['a'] = 1.
model['b'] = 4.
model['P'] = 1.

# Domain bounds
model['lb'] = np.array([0., 0., 0.]) #lower bound of the plate
model['ub'] = np.array([model['b'], model['b'], model['b']]) # Upper bound of the plate

NN_param = dict()
NN_param['layers'] = [3, 50, 50, 50, 3]
NN_param['data_type'] = tf.float32

# Generating points inside the domain using GeometryIGA
myDomain = HollowSphere(model['a'], model['b'])

#Load interior Gauss points
data = scipy.io.loadmat('gaussPtsHollowSphere.mat')
X_f = data['gaussPts']

#Load boundary Gauss points, normals and tractions
dataBound = scipy.io.loadmat('gaussPtsBoundHollowSphere.mat')
X_bnd = dataBound['gaussPtsBound']

fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')
ax.scatter(X_f[:,0], X_f[:,1], X_f[:,2], s = 0.75)
ax.scatter(X_bnd[:,0], X_bnd[:,1], X_bnd[:,2], s=0.75)
ax.set_xlabel('$x$',fontweight='bold',fontsize = 12)
ax.set_ylabel('$y$',fontweight='bold',fontsize = 12)
ax.set_zlabel('$z$',fontweight='bold',fontsize = 12)
ax.tick_params(axis='both', which='major', labelsize = 6)
ax.tick_params(axis='both', which='minor', labelsize = 6)
plt.show()

model_pts = dict()
model_pts['X_int'] = X_f
model_pts['X_bnd'] = X_bnd

modelNN = PINN_HS(model_data, model_pts, NN_param)

start_time = time.time()
num_train_its = 2000
modelNN.train(num_train_its)
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

nPred = 50

withSides = [1, 1, 1, 1, 1, 1]
xGrid, yGrid, zGrid = myDomain.getUnifIntPts(nPred, nPred, nPred, withSides)
Grid = np.concatenate((xGrid, yGrid, zGrid), axis=1)

u_pred, v_pred, w_pred, energy_pred, sigma_x_pred, sigma_y_pred, sigma_z_pred, \
     sigma_xy_pred, sigma_yz_pred, sigma_zx_pred = modelNN.predict(Grid)

# Computing exact displacements
u_exact, v_exact, w_exact = getExactDisplacements(xGrid, yGrid, zGrid, model)    

errU = u_exact - u_pred
errV = v_exact - v_pred
errW = w_exact - w_pred
err = errU**2 + errV**2 + errW**2
err_norm = err.sum()
ex = u_exact**2 + v_exact**2 + w_exact**2
ex_norm = ex.sum()
error_u = np.sqrt(err_norm/ex_norm)
print("Relative L2 error: ", error_u)

# Plot results        
oShapeX = np.resize(xGrid, [nPred, nPred, nPred])
oShapeY = np.resize(yGrid, [nPred, nPred, nPred])
oShapeZ = np.resize(zGrid, [nPred, nPred, nPred])

u = np.resize(u_pred, [nPred, nPred, nPred])
v = np.resize(v_pred, [nPred, nPred, nPred])
w = np.resize(w_pred, [nPred, nPred, nPred])
displacement = (u, v, w)

elas_energy = np.resize(energy_pred, [nPred, nPred, nPred])

gridToVTK("./HollowSphere", oShapeX, oShapeY, oShapeZ, pointData = 
              {"Displacement": displacement, "Elastic Energy": elas_energy})

err_u = np.resize(errU, [nPred, nPred, nPred])
err_v = np.resize(errV, [nPred, nPred, nPred])
err_w = np.resize(errW, [nPred, nPred, nPred])

disp_err = (err_u, err_v, err_w)
gridToVTK("./HollowSphereErr", oShapeX, oShapeY, oShapeZ, pointData = 
              {"Displacement": disp_err, "Elastic Energy": elas_energy})

print('Loss convergence')    
range_adam = np.arange(1,num_train_its+1)
range_lbfgs = np.arange(num_train_its+2, num_train_its+2+len(modelNN.lbfgs_buffer))
ax0, = plt.plot(range_adam, modelNN.loss_adam_buff, label='Adam')
ax1, = plt.plot(range_lbfgs,  modelNN.lbfgs_buffer, label='L-BFGS')
plt.legend(handles=[ax0,ax1])
plt.ylim((min(modelNN.lbfgs_buffer),1))
plt.xlabel('Iteration')
plt.ylabel('Loss value')
plt.show()

#Compute the L2 and energy norm errors using integration
u_pred, v_pred, w_pred, energy_pred, sigma_x_pred, sigma_y_pred, sigma_z_pred, \
     tau_xy_pred, tau_yz_pred, tau_zx_pred = modelNN.predict(X_f)
u_exact, v_exact, w_exact = getExactDisplacements(X_f[:,0], X_f[:,1], X_f[:,2], model)
err_l2 = np.sum(((u_exact-u_pred[:,0])**2 + (v_exact-v_pred[:,0])**2 + \
                 (w_exact-w_pred[:,0])**2)*X_f[:,3])
norm_l2 = np.sum((u_exact**2 + v_exact**2 + w_exact**2)*X_f[:,3])
error_u_l2 = np.sqrt(err_l2/norm_l2)
print("Relative L2 error (integration): ", error_u_l2)

sigma_xx_exact, sigma_yy_exact, sigma_zz_exact, sigma_xy_exact, sigma_yz_exact, \
    sigma_zx_exact = getExactStresses(X_f[:,0], X_f[:,1], X_f[:,2], model)
sigma_xx_err = sigma_xx_exact - sigma_x_pred[:,0]
sigma_yy_err = sigma_yy_exact - sigma_y_pred[:,0]
sigma_zz_err = sigma_zz_exact - sigma_z_pred[:,0]
sigma_xy_err = sigma_xy_exact - tau_xy_pred[:,0]
sigma_yz_err = sigma_yz_exact - tau_yz_pred[:,0]
sigma_zx_err = sigma_zx_exact - tau_zx_pred[:,0]

energy_err = 0
energy_norm = 0
numPts = X_f.shape[0]

C_mat = np.zeros((6,6))
C_mat[0,0] = model['E']*(1-model['nu'])/((1+model['nu'])*(1-2*model['nu']))
C_mat[1,1] = model['E']*(1-model['nu'])/((1+model['nu'])*(1-2*model['nu']))
C_mat[2,2] = model['E']*(1-model['nu'])/((1+model['nu'])*(1-2*model['nu']))
C_mat[0,1] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
C_mat[0,2] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
C_mat[1,0] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
C_mat[1,2] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
C_mat[2,0] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
C_mat[2,0] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
C_mat[3,3] = model['E']/(2*(1+model['nu']))
C_mat[4,4] = model['E']/(2*(1+model['nu']))
C_mat[5,5] = model['E']/(2*(1+model['nu']))

C_inv = np.linalg.inv(C_mat)
for i in range(numPts):
    err_pt = np.array([sigma_xx_err[i], sigma_yy_err[i], sigma_zz_err[i], 
                       sigma_xy_err[i], sigma_yz_err[i], sigma_zx_err[i]])
    norm_pt = np.array([sigma_xx_exact[i], sigma_yy_exact[i], sigma_zz_exact[i], \
                        sigma_xy_exact[i], sigma_yz_exact[i], sigma_zx_exact[i]])
    energy_err = energy_err + err_pt@C_inv@err_pt.T*X_f[i,3]
    energy_norm = energy_norm + norm_pt@C_inv@norm_pt.T*X_f[i,3]
    
print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
