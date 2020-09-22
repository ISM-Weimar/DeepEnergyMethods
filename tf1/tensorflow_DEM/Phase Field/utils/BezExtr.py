# -*- coding: utf-8 -*-
"""File for base geometry class built using the Geomdl class"""
import numpy as np

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
        
        self.degree_u = geomData['degree_u']
        self.degree_v = geomData['degree_v']
        self.ctrlpts_size_u = geomData['ctrlpts_size_u']
        self.ctrlpts_size_v = geomData['ctrlpts_size_v']
        self.ctrlpts = self.getUnweightedCpts(geomData['ctrlpts'], geomData['weights'])
        self.weights = geomData['weights']
        self.knotvector_u = geomData['knotvector_u']
        self.knotvector_v = geomData['knotvector_v']
             
    def getUnweightedCpts(self, ctrlpts, weights):
        numCtrlPts = np.shape(ctrlpts)[0]
        PctrlPts = np.zeros_like(ctrlpts)
        for i in range(2):
            for j in range(numCtrlPts):
                PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
#        PctrlPts = PctrlPts.tolist()
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
    
    def bezierExtraction(self, knot, deg):
        '''
        Bezier extraction
        Based on Algorithm 1, from Borden - Isogeometric finite element data
        structures based on Bezier extraction
        '''
        m = len(knot)-deg-1
        a = deg + 1
        b = a + 1
        # Initialize C with the number of non-zero knotspans in the 3rd dimension
        nb_final = len(np.unique(knot))-1
        C = np.zeros((deg+1,deg+1,nb_final))
        nb = 1
        C[:,:,0] = np.eye(deg + 1)
        while b <= m:        
            C[:,:,nb] = np.eye(deg + 1)
            i = b        
            while (b <= m) and (knot[b] == knot[b-1]):
                b = b+1            
            multiplicity = b-i+1    
            alphas = np.zeros(deg-multiplicity)        
            if (multiplicity < deg):    
                numerator = knot[b-1] - knot[a-1]            
                for j in range(deg,multiplicity,-1):
                    alphas[j-multiplicity-1] = numerator/(knot[a+j-1]-knot[a-1])            
                r = deg - multiplicity
                for j in range(1,r+1):
                    save = r-j+1
                    s = multiplicity + j                          
                    for k in range(deg+1,s,-1):                                
                        alpha = alphas[k-s-1]
                        C[:,k-1,nb-1] = alpha*C[:,k-1,nb-1] + (1-alpha)*C[:,k-2,nb-1]  
                    if b <= m:                
                        C[save-1:save+j,save-1,nb] = C[deg-j:deg+1,deg,nb-1]  
                nb=nb+1
                if b <= m:
                    a=b
                    b=b+1    
            elif multiplicity==deg:
                if b <= m:
                    nb = nb + 1
                    a = b
                    b = b + 1                
        assert(nb==nb_final)
        
        return C, nb

    def computeC(self):
        
        knotU = self.knotvector_u
        knotV = self.knotvector_v
        degU = self.degree_u
        degV = self.degree_v
        C_u, nb = self.bezierExtraction(knotU, degU)
        C_v, nb = self.bezierExtraction(knotV, degV)
        
        numElemU = len(np.unique(knotU)) - 1
        numElemV = len(np.unique(knotV)) - 1
        
        basisU = len(knotU) - degU - 1
        nument = (degU+1)*(degV+1)
        elemInfo = dict()
        elemInfo['vertex'] = []
        elemInfo['nodes'] = []
        elemInfo['C'] = []
        
        for j in range (0, len(knotV)-1):
            for i in range (0, len(knotU)-1):
                if ((knotU[i+1] > knotU[i]) and (knotV[j+1] > knotV[j])):
                    vertices = np.array([knotU[i], knotV[j], knotU[i+1], knotV[j+1]])
                    elemInfo['vertex'].append(vertices)
                    currow = np.array([np.zeros(nument)])
                    tcount = 0
                    for t2 in range(j+1-degV,j+2):
                        for t1 in range(i+1-degU,i+2):

                            currow[0,tcount] = t1 + (t2-1)*basisU 
                            tcount = tcount + 1
                    elemInfo['nodes'].append(currow)

        for j in range (0, numElemV):
            for i in range (0, numElemU):
                cElem = np.kron(C_v[:,:,j],C_u[:,:,i])
                elemInfo['C'].append(cElem)
                    
        return elemInfo
    
    def bernsteinBasis(self,xi, deg):
        ''' 
        Algorithm A1.3 in Piegl & Tiller
        xi is a 1D array
        '''        
        B = np.zeros((len(xi),deg+1))
        B[:,0] = 1.0
        u1 = 1-xi
        u2 = 1+xi    
        
        for j in range(1,deg+1):
            saved = 0.0
            for k in range(0,j):
                temp = B[:,k].copy()
                B[:,k] = saved + u1*temp        
                saved = u2*temp
            B[:,j] = saved
        B = B/np.power(2,deg)
        
        dB = np.zeros((len(xi),deg))
        dB[:,0] = 1.0
        for j in range(1,deg):
            saved = 0.0
            for k in range(0,j):
                temp = dB[:,k].copy()
                dB[:,k] = saved + u1*temp
                saved = u2*temp
            dB[:,j] = saved
        dB = dB/np.power(2,deg)
        dB0 = np.transpose(np.array([np.zeros(len(xi))]))
        dB = np.concatenate((dB0, dB, dB0), axis=1)
        dB = (dB[:,0:-1] - dB[:,1:])*deg
    
        return B, dB    

    def findspan(self, uCoord, vCoord):
        '''Generates the element number on which the co-ordinate is located'''
        knotU = self.knotvector_u
        knotV = self.knotvector_v        
        
        counter = 0
        for j in range (0, len(knotV)-1):
            for i in range (0, len(knotU)-1):
                if ((knotU[i+1] > knotU[i]) and (knotV[j+1] > knotV[j])):
                    if ((uCoord > knotU[i]) and (uCoord < knotU[i+1]) and (vCoord > knotV[j]) and (vCoord < knotV[j+1])):
                        elmtNum = counter
                        break
                    counter = counter + 1
        
        return elmtNum

    def getDerivatives(self, uCoord, vCoord, elmtNo):
        '''
        Generate physical points and jacobians for parameter points inside the domain
        Assume there is one element in the parameter space
        Input: uCoord, vCoord: Inputs the co-odinates of the Gauss points in the parameter space.
        Output: xPhys, yPhys, ptJac - Generates the co-ordinates in the physical space and the jacobian
        '''
        curVertex = self.vertex[elmtNo]
        cElem = self.C[elmtNo]
        curNodes = np.int32(self.nodes[elmtNo])-1 # Python indexing starts from 0
        curPts = np.squeeze(self.ctrlpts[curNodes,0:2])
        wgts = np.transpose(np.array([np.squeeze(self.weights[curNodes,0:1])]))

        # Get the Gauss points on the reference interval [-1,1]
        uMax = curVertex[2]
        uMin = curVertex[0]
        vMax = curVertex[3]
        vMin = curVertex[1]
                
        uHatCoord = (2*uCoord - (uMax+uMin))/(uMax-uMin)
        vHatCoord = (2*vCoord - (vMax+vMin))/(vMax-vMin)
        
        degU = self.degree_u
        degV = self.degree_v
        
        B_u, dB_u = self.bernsteinBasis(uHatCoord,degU)
        B_v, dB_v = self.bernsteinBasis(vHatCoord,degV)
        numGauss = len(uCoord)

        B_u, dB_u = self.bernsteinBasis(uHatCoord,degU)
        B_v, dB_v = self.bernsteinBasis(vHatCoord,degV)

        # Computing the Bernstein polynomials in 2D
        dBdu = np.zeros((numGauss, numGauss, (degU+1)*(degV+1)))
        dBdv = np.zeros((numGauss, numGauss, (degU+1)*(degV+1)))
        R = np.zeros((numGauss, numGauss, (degU+1)*(degV+1)))

        counter = 0
        for j in range(0,degV+1):
            for i in range(0,degU+1):         
                R[:,:,counter] = np.outer(B_u[:,i], B_v[:,j])
                dBdu[:,:,counter] = np.outer(dB_u[:,i],B_v[:,j])
                dBdv[:,:,counter] = np.outer(B_u[:,i],dB_v[:,j])
                counter = counter + 1
                
        quadPts = np.zeros((3))

        # Map the points to the physical space
        for jPt in range(0,numGauss):
            for iPt in range(0,numGauss):
                dRdx = np.matmul(cElem,np.transpose(np.array([dBdu[iPt,jPt,:]])))*2/(uMax-uMin)
                dRdy = np.matmul(cElem,np.transpose(np.array([dBdv[iPt,jPt,:]])))*2/(vMax-vMin)

                RR = np.matmul(cElem,np.transpose(np.array([R[iPt,jPt,:]])))

                RR = RR*wgts
                dRdx = dRdx*wgts
                dRdy = dRdy*wgts
                w_sum = np.sum(RR, axis=0)
                dw_xi = np.sum(dRdx, axis=0)
                dw_eta = np.sum(dRdy, axis=0)
                
                dRdx = dRdx/w_sum  - RR*dw_xi/np.power(w_sum,2)
                dRdy = dRdy/w_sum - RR*dw_eta/np.power(w_sum,2)
                RR = RR/w_sum;
                
                dR  = np.concatenate((dRdx.T,dRdy.T),axis=0)
                dxdxi = np.matmul(dR,curPts)

                coord = np.matmul(np.array([R[iPt,jPt,:]]),curPts)
                detJac = np.absolute(np.linalg.det(dxdxi))
                quadPts[0] = coord[0,0]
                quadPts[1] = coord[0,1]
                quadPts[2] = detJac
                
        xPhys = quadPts[0]
        yPhys = quadPts[1]
        ptJac = quadPts[2]
        
        return xPhys, yPhys, ptJac
    
    def genElemList(self, numElemU, numElemV):
        '''
        Generate the element (vertex) list for an initial (uniform)
        subdivision mesh
        Input: numElemU, numElemV - number of subdivisions in the u and v
                   directions in the parameter space               
        Output: vertex - arrays containing the element vertices + initial level (=0)
        '''
        vertex = np.zeros((numElemU*numElemV, 5))
                        
        #generate the knots on the interval [0,1]
        uEdge = np.linspace(0, 1, numElemU+1)
        vEdge = np.linspace(0, 1, numElemV+1)            

        uPar, vPar = np.meshgrid(uEdge, vEdge)              
        counterElem = 0                
        initalLevel = 0
        
        # Generate points for each element
        for iV in range(numElemV):
            for iU in range(numElemU):
                uMin = uPar[iV, iU]
                uMax = uPar[iV, iU+1]
                vMin = vPar[iV, iU]
                vMax = vPar[iV+1, iU]                
                vertex[counterElem, 0] = uMin
                vertex[counterElem, 1] = vMin
                vertex[counterElem, 2] = uMax
                vertex[counterElem, 3] = vMax
                vertex[counterElem, 4] = initalLevel
                counterElem = counterElem + 1
                                        
        return vertex
    
    def getElemIntPts(self, elemList, numGauss):
        '''
        Generate quadrature points inside the domain
        Input: elemList - contains the vertices of the elements the refined elements
               numGauss - number of Gauss quadrature points for each subdivision
        Output: xPhys, yPhys, wgtPhy - arrays containing the x and y coordinates
                                    of the points and the corresponding weights
        '''
        # Allocate quadPts array        
        quadPts = np.zeros((elemList.shape[0]*numGauss**2, 3))     
        
        # Get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)
        
        # Get the Gauss weights on the reference element [-1, 1]x[-1,1]
        gpWeightU, gpWeightV = np.meshgrid(gw, gw)
        gpWeightUV = np.array(gpWeightU.flatten()*gpWeightV.flatten())
        
        elemInfo = self.computeC()
        self.C = elemInfo['C']
        self.nodes = elemInfo['nodes']
        self.vertex = elemInfo['vertex']
               
        # Generate points for each element
        indexPt = 0
        for iElem in range(elemList.shape[0]):
            
            uMin = elemList[iElem,0]
            uMax = elemList[iElem,2]
            vMin = elemList[iElem,1]
            vMax = elemList[iElem,3]
            
            gpParamU = (uMax-uMin)/2*gp+(uMax+uMin)/2
            gpParamV = (vMax-vMin)/2*gp+(vMax+vMin)/2
            gpParamUg, gpParamVg = np.meshgrid(gpParamU, gpParamV)
            gpParamUV = np.array([gpParamUg.flatten(), gpParamVg.flatten()])
            
            # Jacobian of the transformation from the reference element [-1,1]
            scaleFac = (uMax-uMin)*(vMax-vMin)/4
                
            # Map the points to the physical space
            for iPt in range(numGauss**2):
                curPtU = np.array([gpParamUV[0, iPt]])
                curPtV = np.array([gpParamUV[1, iPt]])
                elmtNo = self.findspan(curPtU, curPtV)
                physPtX, physPtY, ptJac = self.getDerivatives(curPtU, curPtV, elmtNo)
                quadPts[indexPt, 0] = physPtX
                quadPts[indexPt, 1] = physPtY
                quadPts[indexPt, 2] = scaleFac * ptJac * gpWeightUV[iPt]                
                indexPt = indexPt + 1
        
        xPhys = quadPts[:, 0:1]
        yPhys = quadPts[:, 1:2]
        wgtPhys = quadPts[:, 2:3]
        
        return xPhys, yPhys, wgtPhys
    
def refineElemVertex2D(vertex, refList):
    # Refines the elements in vertex with indices given by refList by splitting 
    # each element into 4 subdivisions
    # Input: vertex - array of vertices in format [umin, vmin, umax, vmax]
    #       refList - list of element indices to be refined
    # Output: newVertex - refined list of vertices
    
    numRef = len(refList)
    newVertex = np.zeros((4*numRef,5))
    for i in range(numRef):        
        elemIndex = refList[i]
        uMin = vertex[elemIndex, 0]
        vMin = vertex[elemIndex, 1]
        uMax = vertex[elemIndex, 2]
        vMax = vertex[elemIndex, 3]
        level = vertex[elemIndex, 4]
        uMid = (uMin+uMax)/2
        vMid = (vMin+vMax)/2
        newVertex[4*i, :] = [uMin, vMin, uMid, vMid, level+1]
        newVertex[4*i+1, :] = [uMid, vMin, uMax, vMid, level+1]
        newVertex[4*i+2, :] = [uMin, vMid, uMid, vMax, level+1]
        newVertex[4*i+3, :] = [uMid, vMid, uMax, vMax, level+1]
    vertex = np.delete(vertex, refList, axis=0)
    newVertex = np.concatenate((vertex, newVertex),axis=0)
    
    return newVertex

def refineElemRegionY2D(vertex, refYmin, refYmax):
    # Refines the region bounded by refYmin < y < refYmax
    # Input: vertex - array of vertices in format [umin, vmin, umax, vmax]
    #       refYmin - lower bound of the refinement region
    #       refYmax - upper bound of the refinement region
    # Output: newVertex - new list of vertices
    tol = 1e-4 #tolerance for equality
    refYmax = refYmax+tol
    refYmin = refYmin-tol
    index_ref = []
    for iVertex in range(0,vertex.shape[0]):
        if (vertex[iVertex,1] >= refYmin) and (vertex[iVertex,3] <= refYmax):
            index_ref.append(iVertex)
    newVertex = refineElemVertex2D(vertex, index_ref)
    return newVertex