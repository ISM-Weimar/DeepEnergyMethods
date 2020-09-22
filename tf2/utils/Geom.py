# -*- coding: utf-8 -*-
"""
File for base geometry class built using the Geomdl class
"""

import numpy as np
from geomdl import NURBS
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

class Geometry2D:
    '''
     Base class for 2D domains
     Input: geomData - dictionary containing the geomety information
     Keys: degree_u, degree_v: polynomial degree in the u and v directions
       ctrlpts_size_u, ctrlpts_size_v: number of control points in u,v directions
       ctrlpts: weighted control points (in a list with 
            ctrlpts_size_u*ctrlpts_size_v rows and 3 columns for x,y,z coordinates)
       weights: correspond weights (list with ctrlpts_size_u*ctrlpts_size_v entries)
       knotvector_u, knotvector_v: knot vectors in the u and v directions
    '''
    def __init__(self, geomData):
        self.surf = NURBS.Surface()
        self.surf.degree_u = geomData['degree_u']
        self.surf.degree_v = geomData['degree_v']
        self.surf.ctrlpts_size_u = geomData['ctrlpts_size_u']
        self.surf.ctrlpts_size_v = geomData['ctrlpts_size_v']
        self.surf.ctrlpts = self.getUnweightedCpts(geomData['ctrlpts'], 
                                             geomData['weights'])
        self.surf.weights = geomData['weights']
        self.surf.knotvector_u = geomData['knotvector_u']
        self.surf.knotvector_v = geomData['knotvector_v']                
        
    def getUnweightedCpts(self, ctrlpts, weights):
        numCtrlPts = np.shape(ctrlpts)[0]
        PctrlPts = np.zeros_like(ctrlpts)
        for i in range(3):
            for j in range(numCtrlPts):
                PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
        PctrlPts = PctrlPts.tolist()
        return PctrlPts
        
    def mapPoints(self, uPar, vPar):
        '''
        Map points from the parameter domain [0,1]x[0,1] to the quadrilater domain
        Input:  uPar - array containing the u-coordinates in the parameter space
                vPar - array containing the v-coordinates in the parameter space
                Note: the arrays uPar and vPar must be of the same size
        Output: xPhys - array containing the x-coordinates in the physical space
                yPhys - array containing the y-coordinates in the physical space
        '''        
        gpParamUV = np.array([uPar, vPar])
        evalList = tuple(map(tuple, gpParamUV.transpose()))
        res = np.array(self.surf.evaluate_list(evalList))
                
        return res
    
    def getUnifIntPts(self, numPtsU, numPtsV, withEdges):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU, numPtsV - number of points (including edges) in the u and v
                   directions in the parameter space
               withEdges - 1x4 array of zeros or ones specifying whether the boundary points
                           should be included. The boundary order is [bottom, right,
                           top, left] for the unit square.
        Output: xPhys, yPhys - flattened array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0, 1, numPtsU)
        vEdge = np.linspace(0, 1, numPtsV)
        
        #remove endpoints depending on values of withEdges
        if withEdges[0]==0:
            vEdge = vEdge[1:]
        if withEdges[1]==0:
            uEdge = uEdge[:-1]
        if withEdges[2]==0:
            vEdge = vEdge[:-1]
        if withEdges[3]==0:
            uEdge = uEdge[1:]
            
        #create meshgrid
        uPar, vPar = np.meshgrid(uEdge, vEdge)        
                        
        uPar = uPar.flatten()
        vPar = vPar.flatten()     
        #map points
        res = self.mapPoints(uPar.T, vPar.T)
        
        xPhys = res[:, 0:1]
        yPhys = res[:, 1:2]
        
        return xPhys, yPhys
    
    def getUnifEdgePts(self, numPtsU, numPtsV, edgeIndex):
        '''
        Generate uniformly spaced points on the edge boundaries
        Input: numPtsU, numPtsV - number of points (including edges) in the u and v
                   directions in the parameter space
               edgeIndex - 1x4 array of zeros or ones specifying whether the boundary points
                           should be included. The boundary order is [bottom, right,
                           top, left] for the unit square.
        Output: xPhys, yPhys - flattened array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0, 1, numPtsU)
        vEdge = np.linspace(0, 1, numPtsV)
        
        uPts = np.zeros(0)
        vPts = np.zeros(0)
        
        #remove endpoints depending on values of withEdges
        if edgeIndex[0]==1:
            uPts = np.concatenate((uPts, uEdge))
            vPts = np.concatenate((vPts, np.zeros((numPtsU))))
        if edgeIndex[1]==1:
            uPts = np.concatenate((uPts, np.ones((numPtsV)))) 
            vPts = np.concatenate((vPts, vEdge))
        if edgeIndex[2]==1:
            uPts = np.concatenate((uPts, uEdge))
            vPts = np.concatenate((vPts, np.ones((numPtsU))))
        if edgeIndex[3]==1:
            uPts = np.concatenate((uPts, np.zeros((numPtsV))))
            vPts = np.concatenate((vPts, vEdge))
            
        #map points
        res = self.mapPoints(uPts, vPts)
        
        xPhys = res[:, 0:1]
        yPhys = res[:, 1:2]
        
        return xPhys, yPhys
    
    def getQuadIntPts(self, numElemU, numElemV, numGauss):
        '''
        Generate quadrature points inside the domain
        Input: numElemU, numElemV - number of subdivisions in the u and v
                   directions in the parameter space
               numGauss - number of Gauss quadrature points for each subdivision
        Output: xPhys, yPhys, wgtPhy - arrays containing the x and y coordinates
                                    of the points and the corresponding weights
        '''
        #allocate quadPts array
        quadPts = np.zeros((numElemU*numElemV*numGauss**2, 3))
        
        #get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)
        
        #get the Gauss weights on the reference element [-1, 1]x[-1,1]
        gpWeightU, gpWeightV = np.meshgrid(gw, gw)
        gpWeightUV = np.array(gpWeightU.flatten()*gpWeightV.flatten())
        
        #generate the knots on the interval [0,1]
        uEdge = np.linspace(0, 1, numElemU+1)
        vEdge = np.linspace(0, 1, numElemV+1)            

        #create meshgrid
        uPar, vPar = np.meshgrid(uEdge, vEdge)              
                        
        #generate points for each element
        indexPt = 0
        for iV in range(numElemV):
            for iU in range(numElemU):
                uMin = uPar[iV, iU]
                uMax = uPar[iV, iU+1]
                vMin = vPar[iV, iU]
                vMax = vPar[iV+1, iU]
                gpParamU = (uMax-uMin)/2*gp+(uMax+uMin)/2
                gpParamV = (vMax-vMin)/2*gp+(vMax+vMin)/2
                gpParamUg, gpParamVg = np.meshgrid(gpParamU, gpParamV)
                gpParamUV = np.array([gpParamUg.flatten(), gpParamVg.flatten()])
                #Jacobian of the transformation from the reference element [-1,1]x[-1,1]
                scaleFac = (uMax-uMin)*(vMax-vMin)/4
                
                #map the points to the physical space
                for iPt in range(numGauss**2):
                    curPtU = gpParamUV[0, iPt]
                    curPtV = gpParamUV[1, iPt]
                    derivMat = self.surf.derivatives(curPtU, curPtV, order=1)
                    physPtX = derivMat[0][0][0]
                    physPtY = derivMat[0][0][1]
                    derivU = derivMat[1][0][0:2]
                    derivV = derivMat[0][1][0:2]
                    JacobMat = np.array([derivU,derivV])
                    detJac = np.linalg.det(JacobMat)
                    quadPts[indexPt, 0] = physPtX
                    quadPts[indexPt, 1] = physPtY
                    quadPts[indexPt, 2] = scaleFac * detJac * gpWeightUV[iPt]
                    indexPt = indexPt + 1
                            
        xPhys = quadPts[:, 0:1]
        yPhys = quadPts[:, 1:2]
        wgtPhys = quadPts[:, 2:3]
        
        return xPhys, yPhys, wgtPhys
    
    def getUnweightedCpts2d(self, ctrlpts2d, weights):
        numCtrlPtsU = np.shape(ctrlpts2d)[0]
        numCtrlPtsV = np.shape(ctrlpts2d)[1]
        PctrlPts = np.zeros([numCtrlPtsU,numCtrlPtsV,3])
        counter = 0    
        for j in range(numCtrlPtsU):
            for k in range(numCtrlPtsV):
                for i in range(3):
                    PctrlPts[j,k,i]=ctrlpts2d[j][k][i]/weights[counter]
                counter = counter + 1
        PctrlPts = PctrlPts.tolist()
        return PctrlPts
    
    
    def plotSurf(self):
        #plots the NURBS/B-Spline surface and the control points in 2D
        fig, ax = plt.subplots()
        patches = []
            
        #get the number of points in the u and v directions
        numPtsU = np.int(1/self.surf.delta[0])
        numPtsV = np.int(1/self.surf.delta[1])
        
        for j in range(numPtsV):
            for i in range(numPtsU):
                #get the index of point in the lower left corner of the visualization element
                indexPtSW = j*(numPtsU+1) + i
                indexPtSE = indexPtSW + 1
                indexPtNE = indexPtSW + numPtsU + 2
                indexPtNW = indexPtSW + numPtsU + 1
                XYPts = np.array(self.surf.evalpts)[[indexPtSW, indexPtSE, 
                                indexPtNE, indexPtNW],0:2]
                poly = mpatches.Polygon(XYPts)
                patches.append(poly)
                
                
        collection = PatchCollection(patches, color="lightgreen", cmap=plt.cm.hsv, alpha=1)
        ax.add_collection(collection)
        
        numCtrlPtsU = self.surf._control_points_size[0]
        numCtrlPtsV = self.surf._control_points_size[1]
        ctrlpts = self.getUnweightedCpts2d(self.surf.ctrlpts2d, self.surf.weights)
        #plot the horizontal lines
        for j in range(numCtrlPtsU):
            plt.plot(np.array(ctrlpts)[j,:,0],np.array(ctrlpts)[j,:,1],ls='--',color='black')
        #plot the vertical lines
        for i in range(numCtrlPtsV):
            plt.plot(np.array(ctrlpts)[:,i,0],np.array(ctrlpts)[:,i,1],ls='--',color='black')
        #plot the control points
        plt.scatter(np.array(self.surf.ctrlpts)[:,0],np.array(self.surf.ctrlpts)[:,1],color='red',zorder=10)
        plt.axis('equal')
        
    def plotKntSurf(self):
        #plots the NURBS/B-Spline surface and the knot lines in 2D
        fig, ax = plt.subplots()
        patches = []
        
        #get the number of points in the u and v directions
        self.surf.delta = 0.02
        self.surf.evaluate()
        numPtsU = np.int(1/self.surf.delta[0])
        numPtsV = np.int(1/self.surf.delta[1])
        
        for j in range(numPtsV):
            for i in range(numPtsU):
                #get the index of point in the lower left corner of the visualization element
                indexPtSW = j*(numPtsU+1) + i
                indexPtSE = indexPtSW + 1
                indexPtNE = indexPtSW + numPtsU + 2
                indexPtNW = indexPtSW + numPtsU + 1
                XYPts = np.array(self.surf.evalpts)[[indexPtSW, indexPtSE, indexPtNE, indexPtNW],0:2]
                poly = mpatches.Polygon(XYPts)
                patches.append(poly)
                
        collection = PatchCollection(patches, color="lightgreen", cmap=plt.cm.hsv, alpha=1)
        ax.add_collection(collection)
        
        #plot the horizontal knot lines
        for j in np.unique(self.surf.knotvector_u):
            vVal = np.linspace(0, 1, numPtsV)
            uVal = np.ones(numPtsV)*j    
            uvVal = np.array([uVal, vVal])
            
            evalList=tuple(map(tuple, uvVal.transpose()))
            res=np.array(self.surf.evaluate_list(evalList))        
            plt.plot(res[:,0],res[:,1], ls='-', linewidth=1, color='black')
            
        #plot the vertical lines
        for i in np.unique(self.surf.knotvector_v):
            uVal = np.linspace(0, 1, numPtsU)
            vVal = np.ones(numPtsU)*i    
            uvVal = np.array([uVal, vVal])
            
            evalList=tuple(map(tuple, uvVal.transpose()))
            res=np.array(self.surf.evaluate_list(evalList))        
            plt.plot(res[:,0],res[:,1], ls='-', linewidth=1, color='black')
       
        plt.axis('equal')        
    
    def getQuadEdgePts(self, numElem, numGauss, orient):
        '''
        Generate points on the boundary edge given by orient
        Input: numElem - number of number of subdivisions (in the v direction)
               numGauss - number of Gauss points per subdivision
               orient - edge orientation in parameter space: 1 is down (v=0), 
                        2 is left (u=1), 3 is top (v=1), 4 is right (u=0)
        Output: xBnd, yBnd, wgtBnd - coordinates of the boundary in the physical
                                     space and the corresponding weights
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
                #allocate quadPts array
        quadPts = np.zeros((numElem*numGauss, 5))
        
        #get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)        
        
        #generate the knots on the interval [0,1]
        edgePar = np.linspace(0, 1, numElem+1)            
                        
        #generate points for each element
        indexPt = 0
        for iE in range(numElem):                
                edgeMin = edgePar[iE]
                edgeMax = edgePar[iE+1]
                if orient==1:
                    gpParamU = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                    gpParamV = np.zeros_like(gp)                    
                elif orient==2:
                    gpParamU = np.ones_like(gp)
                    gpParamV = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                elif orient==3:
                    gpParamU = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                    gpParamV = np.ones_like(gp)   
                elif orient==4:
                    gpParamU = np.zeros_like(gp)
                    gpParamV = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                else:
                    raise Exception('Wrong orientation given')
                        
                gpParamUV = np.array([gpParamU.flatten(), gpParamV.flatten()])
                
                #Jacobian of the transformation from the reference element [-1,1]
                scaleFac = (edgeMax-edgeMin)/2
                
                #map the points to the physical space
                for iPt in range(numGauss):
                    curPtU = gpParamUV[0, iPt]
                    curPtV = gpParamUV[1, iPt]
                    derivMat = self.surf.derivatives(curPtU, curPtV, order=1)
                    physPtX = derivMat[0][0][0]
                    physPtY = derivMat[0][0][1]
                    derivU = derivMat[1][0][0:2]
                    derivV = derivMat[0][1][0:2]
                    JacobMat = np.array([derivU,derivV])
                    if orient==1:                                                
                        normX = JacobMat[0,1]
                        normY = -JacobMat[0,0]
                    elif orient==2:
                        normX = JacobMat[1,1]
                        normY = -JacobMat[1,0]
                    elif orient==3:
                        normX = -JacobMat[0,1]
                        normY = JacobMat[0,0]
                    elif orient==4:
                        normX = -JacobMat[1,1]
                        normY = JacobMat[1,0]
                    else:
                        raise Exception('Wrong orientation given')
                        
                    JacobEdge = np.sqrt(normX**2+normY**2)
                    normX = normX/JacobEdge
                    normY = normY/JacobEdge
        
                    quadPts[indexPt, 0] = physPtX
                    quadPts[indexPt, 1] = physPtY
                    quadPts[indexPt, 2] = normX
                    quadPts[indexPt, 3] = normY
                    quadPts[indexPt, 4] = scaleFac * JacobEdge * gw[iPt]
                    indexPt = indexPt + 1
                            
        xPhys = quadPts[:, 0:1]
        yPhys = quadPts[:, 1:2]
        xNorm = quadPts[:, 2:3]
        yNorm = quadPts[:, 3:4]
        wgtPhys = quadPts[:, 4:5]        
        
        return xPhys, yPhys, xNorm, yNorm, wgtPhys  
    

    
class Geometry3D:
    '''
     Base class for 3D domains
     Input: geomData - dictionary containing the geomety information
     Keys: degree_u, degree_v, degree_w: polynomial degree in the u, v, w directions
       ctrlpts_size_u, ctrlpts_size_v, ctrlpts_size_w: number of control points in u,v,w directions
       ctrlpts: weighted control points (in a list with 
            ctrlpts_size_u*ctrlpts_size_v*ctrlpts_size_w rows and 3 columns for x,y,z coordinates)
       weights: correspond weights (list with ctrlpts_size_u*ctrlpts_size_v*ctrlpts_size_w entries)
       knotvector_u, knotvector_v, knotvector_w: knot vectors in the u, v, w directions
    '''
    def __init__(self, geomData):
        self.vol = NURBS.Volume()
        self.vol.degree_u = geomData['degree_u']
        self.vol.degree_v = geomData['degree_v']
        self.vol.degree_w = geomData['degree_w']
        self.vol.ctrlpts_size_u = geomData['ctrlpts_size_u']
        self.vol.ctrlpts_size_v = geomData['ctrlpts_size_v']
        self.vol.ctrlpts_size_w = geomData['ctrlpts_size_w']
        self.vol.ctrlpts = self.getUnweightedCpts(geomData['ctrlpts'], 
                                             geomData['weights'])
        self.vol.weights = geomData['weights']
        self.vol.knotvector_u = geomData['knotvector_u']
        self.vol.knotvector_v = geomData['knotvector_v']
        self.vol.knotvector_w = geomData['knotvector_w']
        
    def getUnweightedCpts(self, ctrlpts, weights):
        numCtrlPts = np.shape(ctrlpts)[0]
        PctrlPts = np.zeros_like(ctrlpts)
        for i in range(3):
            for j in range(numCtrlPts):
                PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
        PctrlPts = PctrlPts.tolist()
        return PctrlPts
        
    def mapPoints(self, uPar, vPar, wPar):
        '''
        Map points from the parameter domain [0,1]x[0,1] to the quadrilater domain
        Input:  uPar - array containing the u-coordinates in the parameter space
                vPar - array containing the v-coordinates in the parameter space
                wPar - array containing the w-coordinates in the parameter space
                Note: the arrays uPar, vPar and wPar must be of the same size
        Output: xPhys - array containing the x-coordinates in the physical space
                yPhys - array containing the y-coordinates in the physical space
                zPhys - array containing the z-coordinates in the physical space
        '''        
        gpParamUVW = np.array([uPar, vPar, wPar])
        evalList = tuple(map(tuple, gpParamUVW.transpose()))
        res = np.array(self.vol.evaluate_list(evalList))
                
        return res
    
    def getUnifIntPts(self, numPtsU, numPtsV, numPtsW, withSides):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU, numPtsV, numPtsW - number of points (including edges) in the u, v, w
                   directions in the parameter space
               withSides - 1x6 array of zeros or ones specifying whether the boundary points
                           should be included. The boundary order is [front, right,
                           back, left, bottom, top] for the unit square.
        Output: xM, yM, zM - flattened array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0, 1, numPtsU)
        vEdge = np.linspace(0, 1, numPtsV)
        wEdge = np.linspace(0, 1, numPtsW)
        
        #remove endpoints depending on values of withEdges
        if withSides[0]==0:
            vEdge = vEdge[1:]
        if withSides[1]==0:
            uEdge = uEdge[:-1]
        if withSides[2]==0:
            vEdge = vEdge[:-1]
        if withSides[3]==0:
            uEdge = uEdge[1:]
        if withSides[4]==0:
            wEdge = wEdge[1:]
        if withSides[5]==0:
            wEdge = wEdge[:-1]
            
        #create meshgrid
        uPar, vPar, wPar = np.meshgrid(uEdge, vEdge, wEdge, indexing='ij')
                        
        uPar = uPar.flatten()
        vPar = vPar.flatten()
        wPar = wPar.flatten()
        
        #map points
        res = self.mapPoints(uPar.T, vPar.T, wPar.T)
        
        xPhys = res[:, 0:1]
        yPhys = res[:, 1:2]
        zPhys = res[:, 2:3]
        
        return xPhys, yPhys, zPhys    
