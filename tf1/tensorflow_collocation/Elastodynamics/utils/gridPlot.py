import os
import matplotlib.pyplot as plt
import numpy as np


def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)
        
def plot1d(u_pred_err,X_star,u_pred,u_exact,v_pred,v_exact,v_pred_err,figHeight,figWidth):
      
    filename = 'disp_err'
    plt.figure(figsize=(figWidth, figHeight)) 
    plt.plot(X_star, 100*u_pred_err, c='b', linewidth=2.0)
    #plt.title('$\phi_{comp}$ and $\phi_{exact}$')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('Relative $\%$ error in displacement',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    filename = 'disp'
    plt.figure(figsize=(figWidth, figHeight)) 
    ax0, = plt.plot(X_star,u_pred, label='$u_{comp}$', c='b', linewidth=2.0)
    ax1, = plt.plot(X_star,u_exact, label='$u_{exact}$', c='r',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    #plt.title('$\phi_{comp}$ and $\phi_{exact}$')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('Displacement',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()

    filename = 'velocity'
    plt.figure(figsize=(figWidth, figHeight)) 
    ax0, = plt.plot(X_star,v_pred, label='$v_{comp}$', c='b', linewidth=2.0)
    ax1, = plt.plot(X_star,v_exact, label='$v_{exact}$', c='r',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    #plt.title('$\phi_{comp}$ and $\phi_{exact}$')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('Velocity',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()

    filename = 'vel_err'
    plt.figure(figsize=(figWidth, figHeight)) 
    plt.plot(X_star, 100*v_pred_err, c='b', linewidth=2.0)
    #plt.title('$\phi_{comp}$ and $\phi_{exact}$')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('Relative $\%$ error in velocity',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()

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
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()

