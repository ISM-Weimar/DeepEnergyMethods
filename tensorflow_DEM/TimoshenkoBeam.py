'''
Implements the 2D Timonshenko beam benchmark problem
Cantilever beam subject to parabolic force
Exact solution from Augarde - The use of Timoshenko’s exact solution for a 
cantilever beam in adaptive analysis ( http://dx.doi.org/10.1016/j.finel.2008.01.010 ) 
Use IGA classes
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.Geom import Geometry2D

mpl.rcParams['figure.dpi'] = 200
import time
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

from utils.PINN import Elasticity2D

class Quadrilateral(Geometry2D):
    '''
     Class for definining a quadrilateral domain
     Input: quadDom - array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                containing the domain corners (control-points)
    '''
    def __init__(self, quadDom):
      
        # Domain bounds
        self.quadDom = quadDom
        
        self.x1, self.y1 = self.quadDom[0,:]
        self.x2, self.y2 = self.quadDom[1,:]
        self.x3, self.y3 = self.quadDom[2,:]
        self.x4, self.y4 = self.quadDom[3,:]
        
        geomData = dict()
        
        # Set degrees
        geomData['degree_u'] = 1
        geomData['degree_v'] = 1
        
        # Set control points
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 2
                
        geomData['ctrlpts'] = [[self.x1, self.y1, 0], [self.x2, self.y2, 0],
                        [self.x3, self.y3, 0], [self.x4, self.y4, 0]]
        
        geomData['weights'] = [1., 1., 1., 1.]
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 1.0, 1.0]
        super().__init__(geomData)

class PINN_TB(Elasticity2D):
    '''
    Class including the non-homogeneous Dirichlet boundary conditions for 
    the Timoshenko beam problem
    '''
    def net_uv(self,x,y):

        X = tf.concat([x,y],1)
        self.W = 2.0
        self.L = 8.0
        self.I = self.W**3/12
        self.P = 2.0
        self.pei = self.P/(6*self.E*self.I)

        uv = self.neural_net(X,self.weights,self.biases)
        y_temp = y - self.W/2    
        u_left = self.pei*y_temp*((2+self.nu)*(y_temp**2-self.W**2/4));
        v_left =-self.pei*(3*self.nu*y_temp**2*self.L);

        u = x*uv[:,0:1] + u_left
        v = x*uv[:,1:2] + v_left

        return u, v

    
def getExactDisplacements(x,y,model):
    y_temp = y - model['W']/2     
    u_exact = model['pei']*y_temp*((6*model['L']-3*x)*x+(2+model['nu'])*\
                   (y_temp**2-model['W']**2/4))
    v_exact =-model['pei']*(3*model['nu']*y_temp**2*(model['L']-x)+ \
                   (4+5*model['nu'])*model['W']**2*x/4+(3*model['L']-x)*x**2)
    return u_exact, v_exact

def getExactStresses(x, y, model):
    y_temp = y - model['W']/2   
    inert = model['W']**3/12
    sigma_xx = model['P']*(model['L']-x)*y_temp/inert
    sigma_yy = np.zeros_like(x)
    sigma_xy = model['P']*(y_temp**2-model['W']**2/4)/(2*inert)
    return sigma_xx, sigma_yy, sigma_xy
    
    
model_data = dict()
model_data['E'] = 1000.0
model_data['nu'] = 0.25


model = dict()
model['E'] = 1000.0
model['nu'] = 0.25
model['L'] = 8.0
model['W'] = 2.0
model['P'] = 2.0
model['I'] = model['W']**3/12
model['pei'] = model['P']/(6*model_data['E']*model['I'])

# Domain bounds
model['lb'] = np.array([0.0,0.0]) #lower bound of the plate
model['ub'] = np.array([model['L'],model['W']]) # Upper bound of the plate

NN_param = dict()
NN_param['layers'] = [2, 30, 30, 30, 2]
NN_param['data_type'] = tf.float32

# Generating points inside the domain using Geometry class
domainCorners = np.array([[0,0],[0, model['W']],[ model['L'],0],[ model['L'],model['W']]])
myQuad = Quadrilateral(domainCorners)

numElemU = 160
numElemV = 40
numGauss = 1

xPhys, yPhys, wgtsPhys = myQuad.getQuadIntPts(numElemU, numElemV, numGauss)
myQuad.plotKntSurf()
plt.scatter(xPhys, yPhys, s=0.5)
X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)

# Generate the boundary points using Geometry class
numElemEdge = 80
numGaussEdge = 1
xRight, yRight, xNormRight, yNormRight, wgtsRight = myQuad.getQuadEdgePts(numElemEdge,
                                                            numGaussEdge, 2)
plt.scatter(xRight, yRight, s=1, c='red', zorder=10 )
plt.show()

trac_x = np.zeros_like(xRight)
yTemp = yRight - model['W']/2 
trac_y = model['P']*(yTemp**2-model['W']**2/4)/2/model['I'];
xPtsRight = np.concatenate((xRight, yRight, wgtsRight, trac_x, trac_y), axis=1)

model_pts = dict()
model_pts['X_int'] = X_f
model_pts['X_bnd'] = xPtsRight

modelNN = PINN_TB(model_data, model_pts, NN_param)

nPred = 40
xPred = np.linspace(0.0, model['L'], nPred)
yPred = np.linspace(0.0, model['W'], nPred)
withEdges = [1, 1, 1, 1]
xGrid, yGrid = myQuad.getUnifIntPts(nPred, nPred, withEdges)
Grid = np.concatenate((xGrid,yGrid),axis=1)

start_time = time.time()
num_train_its = 1000
modelNN.train(num_train_its)
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(Grid)
u_exact, v_exact = getExactDisplacements(xGrid, yGrid, model)  
  

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
    fig, axes = plt.subplots(nrows=2)
    cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=plt.cm.jet)
    cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=plt.cm.jet)
    fig.colorbar(cs1, ax=axes[0])
    fig.colorbar(cs2, ax=axes[1])
    axes[0].set_title("Displacement in x")
    axes[1].set_title("Displacement in y")
    fig.tight_layout()
    for tax in axes:
        tax.set_xlabel('$x$')
        tax.set_ylabel('$y$')
    plt.show()

err = surfaceErrUx**2 + surfaceErrUy**2
err_norm = err.sum()
ex = surfaceExUx**2 + surfaceExUy**2
ex_norm = ex.sum()
error_u = np.sqrt(err_norm/ex_norm)
print("Relative L2 error: ", error_u)

print("Deformation plots")
plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY)
print("Error plots")
plotDeformedDisp(surfaceErrUx, surfaceErrUy, oShapeX, oShapeY)

# Plot the errors on the left boundary
yPred = yPred[np.newaxis]
plt.plot(yPred.T, surfaceUx[:,0], yPred.T, surfaceExUx[:,0])
plt.show()
plt.plot(yPred.T, surfaceErrUx[:,0])
plt.show()


plt.plot(yPred.T, surfaceUy[:,0], yPred.T, surfaceExUy[:,0])
plt.show()
plt.plot(yPred.T, surfaceErrUy[:,0])
plt.show()

# Plotting the strain energy densities        
sEnergy = np.resize(energy_pred, [nPred, nPred])
fig = plt.contourf(defShapeX, defShapeY, sEnergy, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Strain Energy Density")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
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

    
    
    



