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
    \Omega is a quarter annulus in the 1st quadrant, centered at origin
    with inner radius 1, outer radius 4
    Symmetry (Dirichlet) boundary conditions on the bottom and left 
    u_x(x,y) = 0 for x=0
    u_y(x,y) = 0 for y=0
    and pressure boundary conditions for the curved boundaries:
        \sigma n = P_int n on the interior boundary with P_int = 10 MPa
        \sigma n = P_ext n on the exterior boundary with P_ext = 0 MPa.
Use DEM        
"""
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from utils.tfp_loss import tfp_function_factory
from utils.Geom_examples import QuarterAnnulus
from utils.Solvers import Elasticity2D_DEM_dist
from utils.Plotting import plot_field_2d, plot_convergence_dem

np.random.seed(42)
tf.random.set_seed(42)


class Elast_ThickCylinder(Elasticity2D_DEM_dist):
    '''
    Class including the symmetry boundary conditions for the thick cylinder problem
    '''       
    def __init__(self, layers, train_op, num_epoch, print_epoch, model_data, data_type):        
        super().__init__(layers, train_op, num_epoch, print_epoch, model_data, data_type)
       
    @tf.function
    def dirichletBound(self, X, xPhys, yPhys):    
        # multiply by x,y for strong imposition of boundary conditions
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        
        u_val = xPhys*u_val
        v_val = yPhys*v_val
        
        return u_val, v_val
        
#define the model properties
model_data = dict()
model_data["radius_int"] = 1.
model_data["radius_ext"] = 4.
model_data["E"] = 1e2
model_data["nu"] = 0.3
model_data["state"] = "plane strain"
model_data["inner_pressure"] = 10.
model_data["outer_pressure"] = 0.

# generate the model geometry
geomDomain = QuarterAnnulus(model_data["radius_int"], model_data["radius_ext"])

# define the input and output data set
numElemU = 10
numElemV = 10
numGauss = 5
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

# inner curved boundary, include both x and y directions
xPhysBnd, yPhysBnd , xNorm, yNorm, Wbnd = geomDomain.getQuadEdgePts(numElemV, numGauss, 4)
Xbnd = np.concatenate((xPhysBnd, yPhysBnd), axis=1).astype(data_type)
Wbnd = np.array(Wbnd).astype(data_type)

plt.scatter(xPhys, yPhys, s=0.1)
plt.scatter(xPhysBnd, yPhysBnd, s=1, c='red')
plt.title("Boundary and interior integration points")
plt.show()

# define loading
Ybnd_x = -model_data["inner_pressure"]*xNorm
Ybnd_y = -model_data["inner_pressure"]*yNorm
Ybnd = np.concatenate((Ybnd_x, Ybnd_y), axis=1).astype(data_type)

#define the model 
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(20, "swish")
l2 = tf.keras.layers.Dense(20, "swish")
l3 = tf.keras.layers.Dense(20, "swish")
l4 = tf.keras.layers.Dense(2, None)
train_op = tf.keras.optimizers.Adam()
train_op2 = "TFP-BFGS"
num_epoch = 1000
print_epoch = 100
pred_model = Elast_ThickCylinder([l1, l2, l3, l4], train_op, num_epoch, 
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
# train the model with BFGS solver
results = tfp.optimizer.bfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
          max_iterations=1000, tolerance=1e-14)  
# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
loss_func.assign_new_model_parameters(results.position)    
t2 = time.time()
print("Time taken (BFGS)", t2-t1, "seconds")
print("Time taken (all)", t2-t0, "seconds")


def cart2pol(x, y):
    rho = np.sqrt(np.array(x)**2 + np.array(y)**2)
    phi = np.arctan2(y, x)
    return rho, phi

# define the exact displacements
def exact_disp(x,y,model):
    nu = model["nu"]
    r = np.hypot(x,y)
    a = model["radius_int"]
    b = model["radius_ext"]
    mu = model["E"]/(2*(1+nu))
    p1 = model["inner_pressure"]
    p0 = model["outer_pressure"]
    dispxy = 1/(2*mu*(b**2-a**2))*((1-2*nu)*(p1*a**2-p0*b**2)+(p1-p0)*a**2*b**2/r**2)
    ux = x*dispxy
    uy = y*dispxy
    return ux, uy

#define the exact stresses
def exact_stresses(x,y,model):
    r = np.hypot(x,y)
    a = model["radius_int"]
    b = model["radius_ext"]
    p1 = model["inner_pressure"]
    p0 = model["outer_pressure"]
    term_fact = a**2*b**2/(b**2-a**2)
    term_one = p1/b**2 - p0/a**2 + (p1-p0)/r**2
    term_two = 2*(p1-p0)/r**4
    sigma_xx = term_fact*(term_one - term_two*x**2)
    sigma_yy = term_fact*(term_one - term_two*y**2)
    sigma_xy = term_fact*(-term_two*x*y)
    return sigma_xx, sigma_yy, sigma_xy

print("Testing...")
numPtsUTest = 2*numElemU*numGauss
numPtsVTest = 2*numElemV*numGauss
xPhysTest, yPhysTest = geomDomain.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
XTest = np.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
XTest_tf = tf.convert_to_tensor(XTest)
YTest = pred_model(XTest_tf).numpy()  
xPhysTest = xPhysTest.astype(data_type)
yPhysTest = yPhysTest.astype(data_type)
stress_xx_comp, stress_yy_comp, stress_xy_comp = pred_model.constitutiveEq(xPhysTest, yPhysTest)

stress_xx_comp = stress_xx_comp.numpy()
stress_yy_comp = stress_yy_comp.numpy()
stress_xy_comp = stress_xy_comp.numpy()

# plot the displacement
plot_field_2d(XTest, YTest[:,0], numPtsUTest, numPtsVTest, title="Computed x-displacement")
plot_field_2d(XTest, YTest[:,1], numPtsUTest, numPtsVTest, title="Computed y-displacement")

# comparison with exact solution    
ux_exact, uy_exact = exact_disp(xPhysTest, yPhysTest, model_data)
ux_test = YTest[:,0:1]    
uy_test = YTest[:,1:2]
err_norm = np.sqrt(np.sum((ux_exact-ux_test)**2+(uy_exact-uy_test)**2))
ex_norm = np.sqrt(np.sum(ux_exact**2 + uy_exact**2))
rel_err_l2 = err_norm/ex_norm
print("Relative L2 error: ", rel_err_l2)

stress_xx_exact, stress_yy_exact, stress_xy_exact = exact_stresses(xPhysTest,  
                                                        yPhysTest, model_data)

stress_xx_err = stress_xx_exact - stress_xx_comp
stress_yy_err = stress_yy_exact - stress_yy_comp
stress_xy_err = stress_xx_exact - stress_xx_comp

C_inv = np.linalg.inv(pred_model.Emat.numpy())
energy_err = 0.
energy_norm = 0.
numPts = len(xPhysTest)
for i in range(numPts):
    err_pt = np.array([stress_xx_err[i,0],stress_yy_err[i,0],stress_xy_err[i,0]])
    norm_pt = np.array([stress_xx_exact[i,0],stress_yy_exact[i,0],stress_xy_exact[i,0]])
    energy_err = energy_err + err_pt@C_inv@err_pt.T
    energy_norm = energy_norm + norm_pt@C_inv@norm_pt.T

print("Relative energy error: ", np.sqrt(energy_err/energy_norm))


plot_field_2d(XTest, ux_exact-YTest[:,0:1], numPtsUTest, numPtsVTest, title="Error for x-displacement")
plot_field_2d(XTest, uy_exact-YTest[:,1:2], numPtsUTest, numPtsVTest, title="Error for y-displacement")

# plot the stresses
plot_field_2d(XTest, stress_xx_comp, numPtsUTest, numPtsVTest, title="Computed sigma_xx")
plot_field_2d(XTest, stress_yy_comp, numPtsUTest, numPtsVTest, title="Computed sigma_yy")
plot_field_2d(XTest, stress_xy_comp, numPtsUTest, numPtsVTest, title="Computed sigma_xy")

plot_field_2d(XTest, stress_xx_err, numPtsUTest, numPtsVTest, title="Error for sigma_xx")
plot_field_2d(XTest, stress_yy_err, numPtsUTest, numPtsVTest, title="Error for sigma_yy")
plot_field_2d(XTest, stress_xy_err, numPtsUTest, numPtsVTest, title="Error for sigma_xy")

# plot the loss convergence
plot_convergence_dem(pred_model.adam_loss_hist, loss_func.history, percentile=95.)
