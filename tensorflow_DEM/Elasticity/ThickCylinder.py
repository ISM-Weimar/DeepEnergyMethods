'''
Implements the 2D pressurized cylinder benchmark problem
See Example 4.1 in https://doi.org/10.1016/j.cma.2018.10.039
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

from utils.Geom import Geometry2D
from utils.PINN import Elasticity2D

class Annulus(Geometry2D):
    '''
     Class for definining a quarter-annulus domain centered at the orgin
         (the domain is in the first quadrant)
     Input: rad_int, rad_ext - internal and external radii of the annulus
    '''
    def __init__(self, radInt, radExt):            
        
        geomData = dict()
        # Set degrees
        geomData['degree_u'] = 1
        geomData['degree_v'] = 2
        
        # Set control points
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 3
                
        geomData['ctrlpts'] = [[radInt,0.,0.],
                    [radInt*np.sqrt(2)/2, radInt*np.sqrt(2)/2, 0.],
                    [0., radInt, 0.],
                    [radExt, 0., 0.],
                    [radExt*np.sqrt(2)/2, radExt*np.sqrt(2)/2, 0.],
                    [0., radExt, 0.]]
        
        geomData['weights'] = [1, np.sqrt(2)/2, 1, 1, np.sqrt(2)/2, 1]
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        super().__init__(geomData)

class PINN_TC(Elasticity2D):
    '''
    Class including (symmetry) boundary conditions for the thick cylinder problem
    '''       
    def net_uv(self, x, y):

        X = tf.concat([x, y], 1)      

        uv = self.neural_net(X,self.weights,self.biases)

        u = x*uv[:, 0:1]
        v = y*uv[:, 1:2]

        return u, v

def cart2pol(x, y):
    rho = np.sqrt(np.array(x)**2 + np.array(y)**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def getExactDisplacements(x,y,model):
    r, th = cart2pol(x,y)
    u_r = model['radInt']**2*model['P']*r/(model['E']* \
               (model['radExt']**2-model['radInt']**2)) * \
               (1-model['nu']+(model['radExt']/r)**2 * (1 + model['nu']))
    u_exact = u_r*np.cos(th)               
    v_exact = u_r*np.sin(th)
               
    return u_exact, v_exact

def getExactStresses(x, y, model):
    sigma_xx = np.zeros_like(x)
    sigma_yy = np.zeros_like(x)
    sigma_xy = np.zeros_like(x)
    numPts = len(x)
    for i in range(numPts):
        r, th = cart2pol(x[i],y[i])
        sigma_rr = model['radInt']**2*model['P']/(model['radExt']**2-model['radInt']**2) \
                    *(1-model['radExt']**2/r**2)
        sigma_tt = model['radInt']**2*model['P']/(model['radExt']**2-model['radInt']**2) \
                    *(1+model['radExt']**2/r**2)
        sigma_rt = 0
        
        A = np.array([[np.cos(th)**2, np.sin(th)**2, 2*np.sin(th)*np.cos(th)],
                       [np.sin(th)**2, np.cos(th)**2, -2*np.sin(th)*np.cos(th)],
                       [-np.sin(th)*np.cos(th), np.sin(th)*np.cos(th), np.cos(th)**2-np.sin(th)**2]])

        stress_vec = np.linalg.solve(A, np.array([sigma_rr, sigma_tt, sigma_rt]))
        sigma_xx[i] = stress_vec[0]
        sigma_yy[i] = stress_vec[1]
        sigma_xy[i] = stress_vec[2]
    return sigma_xx, sigma_yy, sigma_xy
    
    
model_data = dict()
model_data['E'] = 1e5
model_data['nu'] = 0.3

model = dict()
model['E'] = 1e5
model['nu'] = 0.3
model['radInt'] = 1.0
model['radExt'] = 4.0
model['P'] = 10.0

# Domain bounds
model['lb'] = np.array([0.0,0.0]) #lower bound of the plate
model['ub'] = np.array([model['radExt'],model['radExt']]) # Upper bound of the plate

NN_param = dict()
NN_param['layers'] = [2, 30, 30, 30, 2]
NN_param['data_type'] = tf.float32

# Generating points inside the domain using GeometryIGA
myAnnulus = Annulus(model['radInt'], model['radExt'])

numElemU = 80
numElemV = 80
numGauss = 1

xPhys, yPhys, wgtsPhys = myAnnulus.getQuadIntPts(numElemU, numElemV, numGauss)
myAnnulus.plotKntSurf()
plt.scatter(xPhys, yPhys, s=0.5)
X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)

# Generate the boundary points using Geometry class
numElemEdge = 80
numGaussEdge = 1
xEdge, yEdge, xNormEdge, yNormEdge, wgtsEdge = myAnnulus.getQuadEdgePts(numElemEdge,
                                                            numGaussEdge, 4)
plt.scatter(xEdge, yEdge, s=1, c='red', zorder=10 )
#plt.quiver(xEdge, yEdge, xNormEdge, yNormEdge)
plt.show()

trac_x = -model['P'] * xNormEdge
trac_y = -model['P'] * yNormEdge
xEdgePts = np.concatenate((xEdge, yEdge, wgtsEdge, trac_x, trac_y), axis=1)

model_pts = dict()
model_pts['X_int'] = X_f
model_pts['X_bnd'] = xEdgePts

modelNN = PINN_TC(model_data, model_pts, NN_param)

start_time = time.time()
num_train_its = 1000
modelNN.train(num_train_its)
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))


nPred = 40

withEdges = [1, 1, 1, 1]
xGrid, yGrid = myAnnulus.getUnifIntPts(nPred, nPred, withEdges)
Grid = np.concatenate((xGrid,yGrid),axis=1)

u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(Grid)

# Computing exact displacements
u_exact, v_exact = getExactDisplacements(xGrid, yGrid, model)    

errU = u_exact - u_pred
errV = v_exact - v_pred
err = errU**2 + errV**2
err_norm = err.sum()
ex = u_exact**2 + v_exact**2
ex_norm = ex.sum()
error_u = np.sqrt(err_norm/ex_norm)
print("Relative L2 error: ", error_u)


# Plot results

# Magnification factors for plotting the deformed shape
x_fac = 2
y_fac = 2

# Compute the approximate displacements at plot points     
oShapeX = np.resize(xGrid, [nPred, nPred])
oShapeY = np.resize(yGrid, [nPred, nPred])
surfaceUx = np.resize(u_pred, [nPred, nPred])
surfaceUy = np.resize(v_pred, [nPred, nPred])
surfaceExUx = np.resize(u_exact, [nPred, nPred])
surfaceExUy = np.resize(v_exact, [nPred, nPred])

defShapeX = oShapeX + surfaceUx * x_fac
defShapeY = oShapeY + surfaceUy * y_fac
surfaceErrUx = surfaceExUx - surfaceUx
surfaceErrUy = surfaceExUy - surfaceUy
      
def plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY):
    fig, axes = plt.subplots(ncols=2)
    cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=plt.cm.jet)    
    cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=plt.cm.jet)
    #plot equal colorbars as in 
    #https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(cs1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(cs2, ax=axes[1], fraction=0.046, pad=0.04)

    axes[0].set_title("Displacement in x")
    axes[1].set_title("Displacement in y")
    fig.tight_layout()
    for tax in axes:
        tax.set_xlabel('$x$')
        tax.set_ylabel('$y$')
        tax.set_aspect('equal')    
    plt.show()

print("Deformation plots")
plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY)

print("Exact plots")
plotDeformedDisp(surfaceExUx, surfaceExUy, defShapeX, defShapeY)
print("Error plots")
plotDeformedDisp(surfaceErrUx, surfaceErrUy, oShapeX, oShapeY)

# Plotting the strain energy densities        
sEnergy = np.resize(energy_pred, [nPred, nPred])
fig = plt.contourf(defShapeX, defShapeY, sEnergy, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Strain Energy Density")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.show()

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
u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(X_f)
u_exact, v_exact = getExactDisplacements(X_f[:,0], X_f[:,1], model)
err_l2 = np.sum(((u_exact-u_pred[:,0])**2 + (v_exact-v_pred[:,0])**2)*X_f[:,2])
norm_l2 = np.sum((u_exact**2 + v_exact**2)*X_f[:,2])
error_u_l2 = np.sqrt(err_l2/norm_l2)
print("Relative L2 error (integration): ", error_u_l2)

sigma_xx_exact, sigma_yy_exact, sigma_xy_exact = getExactStresses(X_f[:,0], X_f[:,1], model)
sigma_xx_err = sigma_xx_exact - sigma_x_pred[:,0]
sigma_yy_err = sigma_yy_exact - sigma_y_pred[:,0]
sigma_xy_err = sigma_xy_exact - tau_xy_pred[:,0]

energy_err = 0
energy_norm = 0
numPts = numElemU*numElemV*numGauss

C_mat = np.zeros((3,3))
C_mat[0,0] = model['E']/(1-model['nu']**2)
C_mat[1,1] = model['E']/(1-model['nu']**2)
C_mat[0,1] = model['E']*model['nu']/(1-model['nu']**2)
C_mat[1,0] = model['E']*model['nu']/(1-model['nu']**2)
C_mat[2,2] = model['E']/(2*(1+model['nu']))
C_inv = np.linalg.inv(C_mat)
for i in range(numPts):
    err_pt = np.array([sigma_xx_err[i],sigma_yy_err[i],sigma_xy_err[i]])
    norm_pt = np.array([sigma_xx_exact[i],sigma_yy_exact[i],sigma_xy_exact[i]])
    energy_err = energy_err + err_pt@C_inv@err_pt.T*X_f[i,2]
    energy_norm = energy_norm + norm_pt@C_inv@norm_pt.T*X_f[i,2]
    
print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
