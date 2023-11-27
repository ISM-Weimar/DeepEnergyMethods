#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D linear elasticity example
Solve the equilibrium equation -\nabla \cdot \sigma(x) = f(x) for x\in\Omega 
with the strain-displacement equation:
    \epsilon = 1/2(\nabla u + \nabla u^T)
and the constitutive law:
    \sigma = 2*\mu*\epsilon + \lambda*(\nabla\cdot u)I,
where \mu and \lambda are Lame constants, I is the identity tensor.
Dirichlet boundary conditions: u(x)=\hat{u} for x\in\Gamma_D
Neumann boundary conditions: \sigma n = \hat{t} for x\in \Gamma_N,
where n is the normal vector.
For this example:
    \Omega is a rectangle with corners at  (0,0) and (8,2)
   Dirichlet boundary conditions for x=0:
           u(x,y) = P/(6*E*I)*y*((2+nu)*(y^2-W^2/4))
           v(x,y) = -P/(6*E*I)*(3*nu*y^2*L)
    and parabolic traction at x=8
           p(x,y) = P*(y^2 - y*W)/(2*I)
    where P=2 is the maxmimum traction
          E = 1e3 is Young's modulus
          nu = 0.25 is the Poisson ratio
          I = W^3/12 is second moment of area of the cross-section
        
Use Deep Energy Method
"""
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Elasticity2D_DEM_dist
from utils.Plotting import plot_pts, plot_convergence_dem

np.random.seed(42)
tf.random.set_seed(42)

class Elast_TimoshenkoBeam(Elasticity2D_DEM_dist):
    '''
    Class including the boundary conditions for the Timoshenko beam problem
    '''       
    def __init__(self, layers, train_op, num_epoch, print_epoch, model_data, data_type):        
        super().__init__(layers, train_op, num_epoch, print_epoch, model_data, data_type)
       
    @tf.function
    def dirichletBound(self, X, xPhys, yPhys):    
        # multiply by x,y for strong imposition of boundary conditions
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        self.W = 2.0
        self.L = 8.0
        self.I = self.W**3/12
        self.P = 2.0
        self.pei = self.P/(6*self.Emod*self.I)

        y_temp = yPhys - self.W/2    
        u_left = self.pei*y_temp*((2+self.nu)*(y_temp**2-self.W**2/4));
        v_left = -self.pei*(3*self.nu*y_temp**2*self.L);
        
        u_val = xPhys*u_val + u_left
        v_val = xPhys*v_val + v_left
        
        return u_val, v_val

        
#define the input and output data set
beam_length = 8.
beam_width = 2.
pressure = 2.
domainCorners = np.array([[0., 0.], [0, beam_width], [beam_length, 0.], [beam_length, beam_width]])
geomDomain = Quadrilateral(domainCorners)

model_data = dict()
model_data["E"] = 1e3
model_data["nu"] = 0.25
model_data["state"] = "plane stress"

numElemU = 20
numElemV = 10
numGauss = 4
#xPhys, yPhys = myQuad.getRandomIntPts(numPtsU*numPtsV)
xPhys, yPhys, Wint = geomDomain.getQuadIntPts(numElemU, numElemV, numGauss)
data_type = "float64"

Xint = np.concatenate((xPhys,yPhys),axis=1).astype(data_type)
Wint = np.array(Wint).astype(data_type)

# prepare boundary points in the fromat Xbnd = [Xcoord, Ycoord, norm_x, norm_y] and
# Wbnd for boundary integration weights and
# Ybnd = [trac_x, trac_y], where Xcoord, Ycoord are the x and y coordinates of the point,
# norm_x, norm_y are the x and y components of the unit normals
# trac_x, trac_y are the x and y components of the traction vector at each point
               
#boundary for x=beam_length, include both the x and y directions
xPhysBnd, yPhysBnd, xNorm, yNorm, Wbnd = geomDomain.getQuadEdgePts(numElemV, numGauss, 2)
Xbnd = np.concatenate((xPhysBnd, yPhysBnd, xNorm, yNorm), axis=1).astype(data_type)
Wbnd = np.array(Wbnd).astype(data_type)
inert = beam_width**3/12
Ybnd_x = np.zeros_like(yPhysBnd).astype(data_type)
Ybnd_y = (pressure*(yPhysBnd**2 - yPhysBnd*beam_width)/(2*inert)).astype(data_type)
Ybnd = np.concatenate((Ybnd_x, Ybnd_y), axis=1)

plt.scatter(xPhys, yPhys, s=0.1)
plt.scatter(xPhysBnd, yPhysBnd, s=1, c='red')
plt.title("Boundary and interior integration points")
plt.show()
    
#define the model 
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(20, "swish")         
l2 = tf.keras.layers.Dense(20, "swish")          
l3 = tf.keras.layers.Dense(20, "swish")           
l4 = tf.keras.layers.Dense(2, None)
train_op = tf.keras.optimizers.Adam()
train_op2 = "TFP-BFGS"
num_epoch = 5000
print_epoch = 100
pred_model = Elast_TimoshenkoBeam([l1, l2, l3, l4], train_op, num_epoch, 
                                    print_epoch, model_data, data_type)

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Wint_tf = tf.convert_to_tensor(Wint)
Xbnd_tf = tf.convert_to_tensor(Xbnd)
Wbnd_tf = tf.convert_to_tensor(Wbnd)
Ybnd_tf = tf.convert_to_tensor(Ybnd)

#training
t0 = time.time()
print("Training (ADAM)...")

pred_model.network_learn(Xint_tf, Wint_tf, Xbnd_tf, Wbnd_tf, Ybnd_tf)
t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")


print("Training (TFP-BFGS)...")

loss_func = tfp_function_factory(pred_model, Xint_tf, Wint_tf, Xbnd_tf, Wbnd_tf, Ybnd_tf)
# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)
# train the model with L-BFGS solver
results = tfp.optimizer.bfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
          max_iterations=1000, tolerance=1e-14)
# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
loss_func.assign_new_model_parameters(results.position)
t2 = time.time()
print("Time taken (BFGS)", t2-t1, "seconds")
print("Time taken (all)", t2-t0, "seconds")

#define the exact displacements
def exact_disp(x,y):
    E = model_data["E"]
    nu = model_data["nu"]
    inert=beam_width**3/12;
    pei=pressure/(6*E*inert)
    y_temp = y - beam_width/2  #move (0,0) to below left corner     
    x_disp = pei*y_temp*((6*beam_length-3*x)*x+(2+nu)*(y_temp**2-beam_width**2/4))
    y_disp = -pei*(3*nu*y_temp**2*(beam_length-x)+(4+5*nu)*beam_width**2*x/4+(3*beam_length-x)*x**2)
    return x_disp, y_disp

print("Testing...")
numPtsUTest = 2*numElemU*numGauss
numPtsVTest = 2*numElemV*numGauss
xPhysTest, yPhysTest = geomDomain.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
XTest = np.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
XTest_tf = tf.convert_to_tensor(XTest)
YTest = pred_model(XTest_tf).numpy()        

xPhysTest2D = np.resize(XTest[:,0], [numPtsVTest, numPtsUTest])
yPhysTest2D = np.resize(XTest[:,1], [numPtsVTest, numPtsUTest])
YTest2D_x = np.resize(YTest[:,0], [numPtsVTest, numPtsUTest])
YTest2D_y = np.resize(YTest[:,1], [numPtsVTest, numPtsUTest])

plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D_x, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed x-displacement")
plt.axis('equal')
plt.show()

plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D_y, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed y-displacement")
plt.axis('equal')
plt.show()    


# comparison with exact solution    
ux_exact, uy_exact = exact_disp(xPhysTest, yPhysTest)
ux_test = YTest[:,0:1]    
uy_test = YTest[:,1:2]
err_norm = np.sqrt(np.sum((ux_exact-ux_test)**2+(uy_exact-uy_test)**2))
ex_norm = np.sqrt(np.sum(ux_exact**2 + uy_exact**2))
rel_err_l2 = err_norm/ex_norm
print("Relative L2 error: ", rel_err_l2)

YExact2D_x = np.resize(ux_exact, [numPtsVTest, numPtsUTest])
YExact2D_y = np.resize(uy_exact, [numPtsVTest, numPtsUTest])

plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_x, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Exact x-displacement")
plt.axis('equal')
plt.show()

plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_y, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Exact y-displacement")
plt.axis('equal')
plt.show()

plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_x-YTest2D_x, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Error for x-displacement")
plt.axis('equal')
plt.show()

plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_y-YTest2D_y, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Error for y-displacement")
plt.axis('equal')
plt.show()           
   
# plot the loss convergence
plot_convergence_dem(pred_model.adam_loss_hist, loss_func.history, percentile=95.)