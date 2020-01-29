import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK 

def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)

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

def scatterPlot(X_f,X_bnd,figHeight,figWidth,filename):
    fig = plt.figure(figsize=(figWidth,figHeight))
    ax = fig.gca(projection='3d')
    ax.scatter(X_f[:,0], X_f[:,1], X_f[:,2], s = 0.75)
    ax.scatter(X_bnd[:,0], X_bnd[:,1], X_bnd[:,2], s = 0.75, c='red')
    ax.set_xlabel('$x$',fontweight='bold',fontsize = 12)
    ax.set_ylabel('$y$',fontweight='bold',fontsize = 12)
    ax.set_zlabel('$z$',fontweight='bold',fontsize = 12)
    ax.tick_params(axis='both', which='major', labelsize = 6)
    ax.tick_params(axis='both', which='minor', labelsize = 6)
    plt.savefig(filename+'.png', dpi=300, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()

def energyPlot(xGrid,yGrid,zGrid,nPred,u_pred,v_pred,w_pred,energy_pred,errU,errV,errW,filename):    
    # Plot results        
    oShapeX = np.resize(xGrid, [nPred, nPred, nPred])
    oShapeY = np.resize(yGrid, [nPred, nPred, nPred])
    oShapeZ = np.resize(zGrid, [nPred, nPred, nPred])
    
    u = np.resize(u_pred, [nPred, nPred, nPred])
    v = np.resize(v_pred, [nPred, nPred, nPred])
    w = np.resize(w_pred, [nPred, nPred, nPred])
    displacement = (u, v, w)
    
    elas_energy = np.resize(energy_pred, [nPred, nPred, nPred])
    
    gridToVTK(filename, oShapeX, oShapeY, oShapeZ, pointData = 
                  {"Displacement": displacement, "Elastic Energy": elas_energy})
    
    err_u = np.resize(errU, [nPred, nPred, nPred])
    err_v = np.resize(errV, [nPred, nPred, nPred])
    err_w = np.resize(errW, [nPred, nPred, nPred])
    
    disp_err = (err_u, err_v, err_w)
    gridToVTK(filename+'Err', oShapeX, oShapeY, oShapeZ, pointData = 
                  {"Displacement": disp_err, "Elastic Energy": elas_energy})
    
def plotConvergence(iter,adam_buff,lbfgs_buff,figHeight,figWidth):   
    filename = "convergence"
    plt.figure(figsize=(figWidth, figHeight))        
    range_adam = np.arange(1,iter+1)
    range_lbfgs = np.arange(iter+2, iter+2+len(lbfgs_buff))
    ax0, = plt.semilogy(range_adam, adam_buff, c='b', label='Adam',linewidth=2.0)
    ax1, = plt.semilogy(range_lbfgs, lbfgs_buff, c='r', label='L-BFGS',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Iteration',fontweight='bold',fontsize=14)
    plt.ylabel('Loss value',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename+".pdf", dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()    
    
def energyError(X_f,sigma_x_pred,sigma_y_pred,model,sigma_z_pred,tau_xy_pred,tau_yz_pred,tau_zx_pred,getExactStresses):   
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
        
    return energy_err, energy_norm