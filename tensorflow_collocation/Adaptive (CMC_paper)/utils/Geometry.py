# -*- coding: utf-8 -*-
"""
File for rectangular geometry
"""

import numpy as np


class QuadrilateralGeom:
    '''
     Class for definining a quadrilateral domain
     Input: quadDom - array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                containing the domain corners in counter-clockwise order
    '''
    def __init__(self, quadDom):
      
        # Domain bounds
        self.quadDom = quadDom
        
        self.x1, self.y1 = self.quadDom[0,:]
        self.x2, self.y2 = self.quadDom[1,:]
        self.x3, self.y3 = self.quadDom[2,:]
        self.x4, self.y4 = self.quadDom[3,:]
        
    def mapPoints(self, uPar, vPar):
        '''
        Map points from the parameter domain [0,1]x[0,1] to the quadrilater domain
        Input:  uPar - array containing the u-coordinates in the parameter space
                vPar - array containing the v-coordinates in the parameter space
                Note: the arrays uPar and vPar must be of the same size
        Output: xPhys - array containing the x-coordinates in the physical space
                yPhys - array containing the y-coordinates in the physical space
        '''        

                
        xPhys = (1-uPar)*(1-vPar)*self.x1 + uPar*(1-vPar)*self.x2 + \
                    uPar*vPar*self.x3 + (1-uPar)*vPar*self.x4
        yPhys = (1-uPar)*(1-vPar)*self.y1 + uPar*(1-vPar)*self.y2 + \
                    uPar*vPar*self.y3 + (1-uPar)*vPar*self.y4
        
        return xPhys, yPhys
    
    def getUnifIntPts(self, numPtsU, numPtsV, withEdges):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU, numPtsV - number of points (including edges) in the u and v
                   directions in the parameter space
               withEdges - 1x4 array of zeros specifying whether the boundary points
                           should be included. The boundary order is [bottom, right,
                           top, left] for the unit square.
        Output: xM, yM - meshgrid array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0,1,numPtsU)
        vEdge = np.linspace(0,1,numPtsV)
        
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
        
        #map points
        xPhys, yPhys = self.mapPoints(uPar, vPar)
        
        return xPhys, yPhys
    
    
    def getLeftPts(self, numPts):
        '''
        Generate points on the left boundary mapped from edge {0}x[0,1]
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = np.zeros((numPts,1))
        vEdge = np.transpose(np.linspace(0,1,numPts)[np.newaxis])
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = (self.y1 - self.y4)*np.ones((numPts,1))
        yNorm = (self.x4 - self.x1)*np.ones((numPts,1))
        normMag = np.sqrt(xNorm**2 + yNorm**2)
        xNorm = xNorm/normMag
        yNorm = yNorm/normMag
        
        return xPhys, yPhys, xNorm, yNorm
    
    def getRightPts(self, numPts):
        '''
        Generate points on the right boundary mapped from edge {1}x[0,1]
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = np.ones((numPts,1))
        vEdge = np.transpose(np.linspace(0,1,numPts)[np.newaxis])
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = (self.y3 - self.y2)*np.ones((numPts,1))
        yNorm = (self.x2 - self.x3)*np.ones((numPts,1))
        normMag = np.sqrt(xNorm**2 + yNorm**2)
        xNorm = xNorm/normMag
        yNorm = yNorm/normMag
        return xPhys, yPhys, xNorm, yNorm
        
    def getBottomPts(self, numPts):
        '''
        Generate points on the right boundary mapped from edge [0,1]x{0}
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = np.transpose(np.linspace(0,1,numPts)[np.newaxis])
        vEdge = np.zeros((numPts,1))
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = (self.y2 - self.y1)*np.ones((numPts,1))
        yNorm = (self.x1 - self.x2)*np.ones((numPts,1))
        normMag = np.sqrt(xNorm**2 + yNorm**2)
        xNorm = xNorm/normMag
        yNorm = yNorm/normMag
        return xPhys, yPhys, xNorm, yNorm
    
    def getTopPts(self, numPts):
        '''
        Generate points on the right boundary mapped from edge [0,1]x{1}
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = np.transpose(np.linspace(0,1,numPts)[np.newaxis])
        vEdge = np.ones((numPts,1))
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = (self.y4-self.y3)*np.ones((numPts,1))
        yNorm = (self.x3-self.x4)*np.ones((numPts,1))
        normMag = np.sqrt(xNorm**2 + yNorm**2)
        xNorm = xNorm/normMag
        yNorm = yNorm/normMag
        return xPhys, yPhys, xNorm, yNorm
    
class AnnulusGeom:
    '''
     Class for definining a quarter-annulus domain centered at the orgin 
     Input: rad_int, rad_ext
    '''
    def __init__(self, rad_int, rad_ext):
      
        # Domain bounds
        self.rad_int = rad_int
        self.rad_ext = rad_ext
        
    def mapPoints(self, uPar, vPar):
        '''
        Map points from the polar domain [rad_int,rad_ext]x[0,pi/2] to the 
            cartesian quarter-annulus domain
            
        Input:  uPar - array containing the u-coordinates in the parameter space
                vPar - array containing the v-coordinates in the parameter space
                Note: the arrays uPar and vPar must be of the same size
        Output: xPhys - array containing the x-coordinates in the physical space
                yPhys - array containing the y-coordinates in the physical space
        '''                        
        xPhys = uPar*np.cos(vPar)
        yPhys = uPar*np.sin(vPar)
        
        return xPhys, yPhys
    
    def getUnifIntPts(self, numPtsU, numPtsV, withEdges):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU, numPtsV - number of points (including edges) in the u and v
                   directions in the parameter space
               withEdges - 1x4 array of zeros specifying whether the boundary points
                           should be included. The boundary order is [bottom, right,
                           top, left] for the unit square.
        Output: xM, yM - meshgrid array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(self.rad_int,self.rad_ext,numPtsU)
        vEdge = np.linspace(0,np.pi/2,numPtsV)
        
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
        
        #map points
        xPhys, yPhys = self.mapPoints(uPar, vPar)
        
        return xPhys, yPhys
    
    
    def getInnerPts(self, numPts):
        '''
        Generate points on the inner boundary of the annulus 
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = self.rad_int*np.ones((numPts,1))
        vEdge = np.transpose(np.linspace(0,np.pi/2,numPts)[np.newaxis])
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = -xPhys
        yNorm = -yPhys
        
        return xPhys, yPhys, xNorm, yNorm
    
    def getOuterPts(self, numPts):
        '''
        Generate points on the outer boundary of the annulus
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = self.rad_ext*np.ones((numPts,1))
        vEdge = np.transpose(np.linspace(0,np.pi/2,numPts)[np.newaxis])
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = xPhys
        yNorm = yPhys
        return xPhys, yPhys, xNorm, yNorm
        
    def getXAxPts(self, numPts):
        '''
        Generate points on the x-axis boundary of the annulus
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = np.transpose(np.linspace(self.rad_int,self.rad_ext,numPts)[np.newaxis])
        vEdge = np.zeros((numPts,1))
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = np.zeros((numPts,1))
        yNorm = -1*np.ones((numPts,1))
        return xPhys, yPhys, xNorm, yNorm
    
    def getYAxPts(self, numPts):
        '''
        Generate points on the y-axis boundary of the annulus
        Input: numPts - number of points to generated
        Output: xBnd, yBnd - coordinates of the boundary in the physical coordinates
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
        uEdge = np.transpose(np.linspace(self.rad_int,self.rad_ext,numPts)[np.newaxis])
        vEdge = np.pi/2*np.ones((numPts,1))
        xPhys, yPhys = self.mapPoints(uEdge, vEdge)
        xNorm = -1*np.ones((numPts,1))
        yNorm = np.zeros((numPts,1))
        return xPhys, yPhys, xNorm, yNorm
        
        
        
        
    
        
        