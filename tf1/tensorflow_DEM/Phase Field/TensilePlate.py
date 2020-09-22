# Implements the second-order phase field to study the growth of fracture in a two dimensional plate
# The plate has initial crack and is under tensile loading

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
import os
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
tf.logging.set_verbosity(tf.logging.ERROR)
from utils.gridPlot import scatterPlot
from utils.gridPlot import genGrid
from utils.gridPlot import plotDispStrainEnerg
from utils.gridPlot import plotPhiStrainEnerg
from utils.gridPlot import plotConvergence
from utils.gridPlot import createFolder
from utils.gridPlot import plot1dPhi
from utils.BezExtr import Geometry2D
from utils.gridPlot import plotForceDisp
from utils.BezExtr import refineElemRegionY2D
from utils.PINN_2ndPF import CalculateUPhi

np.random.seed(1234)
tf.set_random_seed(1234)

class Quadrilateral(Geometry2D):
    '''
    Class for definining a quadrilateral domain
    Input: quadDom: array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
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
        
        geomData['ctrlpts'] = np.array([[self.x1, self.y1, 0], [self.x2, self.y2, 0],
                        [self.x3, self.y3, 0], [self.x4, self.y4, 0]])

        geomData['weights'] = np.array([[1.0], [1.0], [1.0], [1.0]])
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 1.0, 1.0]

        super().__init__(geomData)

class PINN_PF(CalculateUPhi):
    '''
    Class including (symmetry) boundary conditions for the tension plate
    '''
    def __init__(self, model, NN_param):
        #PhysicsInformedNN.__init__(self, model_data, model_pts, NN_param)
        super().__init__(model, NN_param)
        
    def net_uv(self,x,y,vdelta):

        X = tf.concat([x,y],1)

        uvphi = self.neural_net(X,self.weights,self.biases)
        uNN = uvphi[:,0:1]
        vNN = uvphi[:,1:2]
        
        u = (1-x)*x*uNN
        v = y*(y-1)*vNN + y*vdelta

        return u, v
if __name__ == "__main__":
    
    originalDir = os.getcwd()
    foldername = 'TensionPlate_results'    
    createFolder('./'+ foldername + '/')
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    
    figHeight = 5
    figWidth = 5
    nSteps = 8 # Total number of steps to observe the growth of crack 
    deltaV = 1e-3 #displacement increment per step 
    
    model = dict()
    model['E'] = 210.0*1e3 # Modulus of elasticity
    model['nu'] = 0.3 # Poission's ratio
    model['L'] = 1.0 # Length of the plate
    model['W'] = 1.0 # Breadth of the plate
    model['l'] = 0.0125 # length scale parameter    
    # Domain bounds
    model['lb'] = np.array([0.0,0.0]) #lower bound of the plate
    model['ub'] = np.array([model['L'],model['W']]) # Upper bound of the plate

    NN_param = dict() # neural network parameters
    NN_param['layers'] = [2, 20, 20, 20, 3] # Layers and neurons for the neural network
    NN_param['data_type'] = tf.float32 # Data type of the variables for the analysis 
    
    # Generating points inside the domain using Geometry class
    domainCorners = np.array([[0,0],[model['W'], 0.],[0, model['L']],[ model['W'],model['L']]])
    myQuad = Quadrilateral(domainCorners)
    numElemU = 40
    numElemV = 40
    numGauss = 1  
    vertex = myQuad.genElemList(numElemU,numElemV)
    
    # Refine betwenn 0.3 < y < 0.7
    refYmax = 0.7
    refYmin = 0.3
    vertex = refineElemRegionY2D(vertex,refYmin,refYmax)    
    
    # Refine betwenn 0.4 < y < 0.6
    refYmax = 0.6
    refYmin = 0.4
    vertex = refineElemRegionY2D(vertex,refYmin,refYmax)    
    
    # Refine betwenn 0.45 < y < 0.55
    refYmax = 0.55
    refYmin = 0.45
    vertex = refineElemRegionY2D(vertex,refYmin,refYmax) 
    
    # Refine betwenn 0.475 < y < 0.525
    refYmax = 0.525
    refYmin = 0.475
    vertex = refineElemRegionY2D(vertex,refYmin,refYmax)
   
    
    xPhys, yPhys, wgtsPhys = myQuad.getElemIntPts(vertex, numGauss)
    X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)
    hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
    filename = 'Training_scatter'
    scatterPlot(X_f,figHeight,figWidth,filename)
    
    # Boundary data
    N_b = 800 # Number of boundary points for the fracture analysis (Left, Right, Top, Bottom)
    x_bottomEdge = np.array([np.linspace(0.0,model['L'],int(N_b/4), dtype = np.float32)])
    y_bottomEdge = np.zeros((int(N_b/4),1),dtype = np.float32)
    xBottomEdge = np.concatenate((x_bottomEdge.T,y_bottomEdge), axis=1)
                                 
    # Generating the prediction mesh
    nPred = np.array([[135,45],[135,45],[135,45]])
    offset = 2*model['l']    
    secBound = np.array([[0.0, 0.5*model['L']-offset],[0.5*model['L']-offset, 
                          0.5*model['L']+offset],[0.5*model['L']+offset, model['L']]], dtype = np.float32)
    Grid, xGrid, yGrid, hist_grid = genGrid(nPred,model['L'],secBound)
    filename = 'Prediction_scatter'
    scatterPlot(Grid,figHeight,figWidth,filename)

    fdGraph = np.zeros((nSteps,2),dtype = np.float32)
    phi_pred_old = hist_grid # Initializing phi_pred_old to zero
    
    modelNN = PINN_PF(model, NN_param)
    num_train_its = 10000
    
    for iStep in range(0,nSteps):
        
        v_delta = deltaV*iStep       
        
        start_time = time.time()    
        if iStep==0:
            num_lbfgs_its = 10000
        else:
            num_lbfgs_its = 2000
        
        modelNN.train(X_f, v_delta, hist_f, num_train_its, num_lbfgs_its)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        
        _, _, _, _, _, hist_f = modelNN.predict(X_f[:,0:2], hist_f, v_delta) # Computing the history function for the next step
        u_pred, v_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = modelNN.predict(Grid,hist_grid,v_delta)        
        
        traction_pred = modelNN.predict_traction(xBottomEdge,v_delta)
        fdGraph[iStep,0] = v_delta
        fdGraph[iStep,1] = 4*np.sum(traction_pred,axis=0)/N_b       
        
        phi_pred = np.maximum(phi_pred, phi_pred_old)
        phi_pred_old = phi_pred
        
        plotPhiStrainEnerg(nPred,xGrid,yGrid,phi_pred,frac_energy_pred,iStep,figHeight,figWidth)
        plotDispStrainEnerg(nPred,xGrid,yGrid,u_pred,v_pred,elas_energy_pred,iStep,figHeight,figWidth)

        adam_buff = modelNN.loss_adam_buff
        lbfgs_buff = modelNN.lbfgs_buffer
        plotConvergence(num_train_its,adam_buff,lbfgs_buff,iStep,figHeight,figWidth)
                
        # 1D plot of phase field
        xVal = 0.25
        nPredY = 2000
        xPred = xVal*np.ones((nPredY,1))
        yPred = np.linspace(0,model['W'],nPredY)[np.newaxis]
        xyPred = np.concatenate((xPred,yPred.T),axis=1)
        phi_pred_1d = modelNN.predict_phi(xyPred)
        phi_exact = np.exp(-np.absolute(yPred-0.5)/model['l'])
        plot1dPhi(yPred,phi_pred_1d,phi_exact,iStep,figHeight,figWidth)
        
        error_phi = (np.linalg.norm(phi_exact-phi_pred_1d.T,2)/np.linalg.norm(phi_exact,2))
        print('Relative error phi: %e' % (error_phi))
        
        print('Completed '+ str(iStep+1) +' of '+str(nSteps)+'.')    
        
    plotForceDisp(fdGraph,figHeight,figWidth)
    os.chdir(originalDir)
    