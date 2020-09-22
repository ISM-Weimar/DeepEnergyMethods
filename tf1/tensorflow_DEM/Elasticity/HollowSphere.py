# Implements the an hollow sphere under pressure problem

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
import os
from utils.gridPlot import energyError
from utils.gridPlot import scatterPlot
from utils.gridPlot import createFolder
from utils.gridPlot import energyPlot
from utils.gridPlot import plotConvergence
from utils.gridPlot import cart2sph
from utils.gridPlot import sph2cart
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
                    
        geomData['ctrlpts'] =  [[a, 0, 0],[wgt*a, 0, wgt*a],[0, 0, a],
                 [wgt*a, wgt*a, 0],[0.5*a, 0.5*a, 0.5*a],[0, 0, wgt*a],
                 [0, a, 0],[0, wgt*a, wgt*a],[0, 0, a],[b, 0, 0],[wgt*b, 0, wgt*b],
                 [0, 0, b],[wgt*b, wgt*b, 0],[0.5*b, 0.5*b, 0.5*b],[0, 0, wgt*b],
                 [0, b, 0],[0, wgt*b, wgt*b],[0, 0, b]]
        
        geomData['weights'] = [1.0,wgt,1,wgt,1/2,wgt,1,wgt,1,1,wgt,1,wgt,1/2,wgt,1,wgt,1]
        
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



#Main program
originalDir = os.getcwd()
foldername = 'HollowSphere_results'    
createFolder('./'+ foldername + '/')
os.chdir(os.path.join(originalDir, './'+ foldername + '/'))

    
figHeight = 6
figWidth = 6
    
# Material parameters  
model_data = dict()
model_data['E'] = 1e3
model_data['nu'] = 0.3

model = dict()
model['E'] = model_data['E']
model['nu'] = model_data['nu']
model['a'] = 1.0 # Internal diameter
model['b'] = 4.0 # External diameter
model['P'] = 1.0 # Internal oressure

# Domain bounds
model['lb'] = np.array([0., 0., 0.]) #lower bound of the plate
model['ub'] = np.array([model['b'], model['b'], model['b']]) # Upper bound of the plate

NN_param = dict()
NN_param['layers'] = [3, 50, 50, 50, 3]
NN_param['data_type'] = tf.float32

# Generating points inside the domain using GeometryIGA
myDomain = HollowSphere(model['a'], model['b'])

numElemU = 15
numElemV = 15
numElemW = 15
numGauss = 2

vertex = myDomain.genElemList(numElemU, numElemV, numElemW)
xPhys, yPhys, zPhys, wgtsPhys = myDomain.getElemIntPts(vertex, numGauss)
X_f = np.concatenate((xPhys,yPhys,zPhys,wgtsPhys),axis=1)

numElemFace = [40, 40]
numGaussFace = 2
orientFace = 5
xFace, yFace, zFace, xNorm, yNorm, zNorm, wgtsFace = myDomain.getQuadFacePts(numElemFace,
                                                            numGaussFace, orientFace)
X_bnd = np.concatenate((xFace, yFace, zFace, wgtsFace, xNorm, yNorm, zNorm), axis=1)

model_pts = dict()
model_pts['X_int'] = X_f
model_pts['X_bnd'] = X_bnd

modelNN = PINN_HS(model_data, model_pts, NN_param)
filename = 'Training_scatter'
scatterPlot(X_f,X_bnd,figHeight,figWidth,filename)

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
filename = 'HollowSphere'
energyPlot(xGrid,yGrid,zGrid,nPred,u_pred,v_pred,w_pred,energy_pred,errU,errV,errW,filename)

print('Loss convergence')
adam_buff = modelNN.loss_adam_buff
lbfgs_buff = modelNN.lbfgs_buffer
plotConvergence(num_train_its,adam_buff,lbfgs_buff,figHeight,figWidth)

# Compute the L2 and energy norm errors using integration
u_pred, v_pred, w_pred, energy_pred, sigma_x_pred, sigma_y_pred, sigma_z_pred, \
     tau_xy_pred, tau_yz_pred, tau_zx_pred = modelNN.predict(X_f)
u_exact, v_exact, w_exact = getExactDisplacements(X_f[:,0], X_f[:,1], X_f[:,2], model)
err_l2 = np.sum(((u_exact-u_pred[:,0])**2 + (v_exact-v_pred[:,0])**2 + \
                 (w_exact-w_pred[:,0])**2)*X_f[:,3])
norm_l2 = np.sum((u_exact**2 + v_exact**2 + w_exact**2)*X_f[:,3])
error_u_l2 = np.sqrt(err_l2/norm_l2)
print("Relative L2 error (integration): ", error_u_l2)

energy_err, energy_norm = energyError(X_f,sigma_x_pred,sigma_y_pred,model,\
                                     sigma_z_pred,tau_xy_pred,tau_yz_pred,tau_zx_pred,getExactStresses)
print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
os.chdir(originalDir)

