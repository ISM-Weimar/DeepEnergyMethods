# -*- coding: utf-8 -*-
"""
Example geometries extending the base class
"""

from utils.Geom import Geometry2D
import numpy as np
class Quadrilateral(Geometry2D):
    '''
     Class for definining a quadrilateral domain
     Input: quadDom - array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                containing the domain corners (control-points)
    '''
    def __init__(self, quadDom):
      
        # Domain vertices
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
        
class Disk(Geometry2D):
    '''
    Class for defining a disk domain using 9 control points
    Input: center - array of the form [x,y] containing the disk center
           radius - disk radius
    '''
    def __init__(self, center, radius):
      
        #unweighted control points for the unit center
        cptsUnit = [[-1., 0., 0.], [-1., -1, 0.], [0., -1., 0.], 
                    [-1, 1., 0.], [0., 0., 0.], [1., -1., 0.], [0., 1., 0.],
                    [1., 1., 0.], [1., 0., 0.]]
        
        #scale and translate to a circle with given center and radius
        cptsDisk = np.array(cptsUnit)*radius+center
        
        #weigh the control points
        weights = [1., 1/np.sqrt(2), 1., 1/np.sqrt(2), 1., 1/np.sqrt(2), 
                   1., 1/np.sqrt(2), 1]
        for i in range(3):
            for j in range(9):
                cptsDisk[j,i]=cptsDisk[j,i]*weights[j]
        
        geomData = dict()
        
        # Set degrees
        geomData['degree_u'] = 2
        geomData['degree_v'] = 2
        
        # Set control points
        geomData['ctrlpts_size_u'] = 3
        geomData['ctrlpts_size_v'] = 3
                
        geomData['ctrlpts'] = cptsDisk.tolist()
        geomData['weights'] = weights
        
        # Set knot vectors
        geomData['knotvector_u'] = [0., 0., 0., 1., 1., 1.]
        geomData['knotvector_v'] = [0., 0., 0., 1., 1., 1.]
        super().__init__(geomData)
