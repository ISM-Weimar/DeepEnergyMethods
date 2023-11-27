#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:39:44 2020

@author: cosmin
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_pts(Xint, Xbnd):
    '''
    Plots the collcation points from the interior and boundary

    Parameters
    ----------
    Xint : TYPE
        DESCRIPTION.
    Xbnd : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
     #plot the boundary and interior points
    plt.scatter(Xint[:,0], Xint[:,1], s=0.5)
    plt.scatter(Xbnd[:,0], Xbnd[:,1], s=1, c='red')
    plt.title("Boundary and interior collocation points")
    plt.show()
    
    
def plot_solution(numPtsUTest, numPtsVTest, domain, pred_model, data_type):
    xPhysTest, yPhysTest = domain.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
    XTest = np.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
    XTest_tf = tf.convert_to_tensor(XTest)
    YTest = pred_model(XTest_tf).numpy()    
   # YExact = exact_sol(XTest[:,[0]], XTest[:,[1]])

    xPhysTest2D = np.resize(XTest[:,0], [numPtsUTest, numPtsVTest])
    yPhysTest2D = np.resize(XTest[:,1], [numPtsUTest, numPtsVTest])
    #YExact2D = np.resize(YExact, [numPtsUTest, numPtsVTest])
    YTest2D = np.resize(YTest, [numPtsUTest, numPtsVTest])
    # plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D, 255, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.title("Exact solution")
    # plt.show()
    plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title("Computed solution")
    plt.show()


def plot_solution_2d_elast(numPtsUTest, numPtsVTest, domain, pred_model, data_type):
    xPhysTest, yPhysTest = domain.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
    XTest = np.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
    XTest_tf = tf.convert_to_tensor(XTest)
    YTest = pred_model(XTest_tf).numpy()    
   # YExact = exact_sol(XTest[:,[0]], XTest[:,[1]])

    xPhysTest2D = np.resize(XTest[:,0], [numPtsUTest, numPtsVTest])
    yPhysTest2D = np.resize(XTest[:,1], [numPtsUTest, numPtsVTest])
    #YExact2D = np.resize(YExact, [numPtsUTest, numPtsVTest])
    YTest2D = np.resize(YTest, [numPtsUTest, numPtsVTest])
    # plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D, 255, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.title("Exact solution")
    # plt.show()
    plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title("Computed solution")
    plt.show()
    
def plot_field_2d(Xpts, Ypts, numPtsU, numPtsV, title=""):
    '''
    Plots a field in 2D

    Parameters
    ----------
    Xpts : nx2 array containing the x and y coordinates of the points on a grid
    YPts : nx1 array containing the value to be plotted at each point
    numPtsU : scalar
                number of points in the U-parametric direction of the grid
    numPtsV : scalar
                number of points in the V-parametric direction of the grid
    title : string, optional
        The title of the plot. The default is "".

    Returns
    -------
    None.

    '''
    xPtsPlt = np.resize(Xpts[:,0], [numPtsV, numPtsU])
    yPtsPlt = np.resize(Xpts[:,1], [numPtsV, numPtsU])
    fieldPtsPlt = np.resize(Ypts, [numPtsV, numPtsU])
    plt.contourf(xPtsPlt, yPtsPlt, fieldPtsPlt, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title)
    plt.show()
    
def plot_convergence_semilog(adam_loss_hist, bfgs_loss_hist):
    '''
    Plots the convergence of the loss function for the collocation method. The 
    plot uses log scale for the y-axis.

    Parameters
    ----------
    adam_loss_hist : (list of floats)
        the loss at each iteration of the Adam optimizer
    bfgs_loss_hist : (list of floats)
        the loss at each iteration of the BFGS optimizer

    Returns
    -------
    None.

    '''
    num_epoch = len(adam_loss_hist)
    num_iter_bfgs = len(bfgs_loss_hist)
    plt.semilogy(range(num_epoch), adam_loss_hist, label='Adam')
    plt.semilogy(range(num_epoch, num_epoch+num_iter_bfgs), bfgs_loss_hist, 
                 label = 'BFGS')
    plt.legend()
    plt.title('Loss convergence')
    plt.show()
    
def plot_convergence_dem(adam_loss_hist, bfgs_loss_hist, percentile=99., 
                         folder = None, file = None):
    '''
    Plots the convergence of the loss function for the Deep Energy Method. The
    values higher than the "percentile" parameter are ignored.

    Parameters
    ----------
    adam_loss_hist : (list of floats)
        the loss at each iteration of the Adam optimizer
    bfgs_loss_hist : (list of floats)
        the loss at each iteration of the BFGS optimizer
    percentile : (float)
        . The default is 99.

    Returns
    -------
    None.

    '''
    num_epoch = len(adam_loss_hist)
    num_iter_bfgs = len(bfgs_loss_hist)
    loss_hist_all = adam_loss_hist + bfgs_loss_hist
    y_max = np.percentile(loss_hist_all, percentile)
    y_min = np.min(loss_hist_all)
    plt.ylim(y_min, y_max)
    plt.plot(range(num_epoch), np.minimum(adam_loss_hist, y_max) , label='Adam')
    plt.plot(range(num_epoch, num_epoch+num_iter_bfgs), np.minimum(bfgs_loss_hist, y_max), 
                 label = 'BFGS')
    plt.legend()
    plt.title('Loss convergence')
    if folder != None:
        full_name = folder + '/' + file
        plt.savefig(full_name)
    plt.show()

