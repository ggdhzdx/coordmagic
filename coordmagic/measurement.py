import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance_matrix

'''All methods that performs measurements are collected here'''

__all__ = [
    'distance',
    'angle',
    'torsion'
]
class Geometry:
    def __init__(self,coord):
        self.coord=np.array(coord)
        if len(self.coord.shape) == 1:
            self.coord = np.array([coord])
        self.coord = self.coord[:,:3]
        self.C = self.coord.mean(axis=0)
        self.coordR = self.coord - self.C
        if len(self.coord) > 1:
            u,sigma,v = np.linalg.svd(self.coordR)
            self.L = v[0]
            self.Lpercent = sigma[0]/np.sum(sigma)
        if len(self.coord) > 2:
            self.P = v[:2]
            self.Pnorm = v[2]
            self.Ppercent = (sigma[0]+sigma[1])/np.sum(sigma)

def __ang__(v1,v2):
    '''return angle between -PI/2 and PI/2'''
    cosang = np.dot(v1,v2)
    sinang = norm(np.cross(v1,v2))
    return np.rad2deg(np.arctan(sinang/cosang))

def __ang2__(v1,v2):
    '''return angle between -PI and PI'''
    cosang = np.dot(v1,v2)
    sinang = norm(np.cross(v1,v2))
    return np.rad2deg(np.arctan2(sinang,cosang))

def distance(A,B,mtype='cc'):
    '''
    A and B coords of two object
    aa: distance between all atoms, return 2D array
    cc: distance between two centroid
    ac: distance between all atoms and centroid, return 1D array
    cl: distance between centroid and line
    al: distance between all atoms and line, return 1D array
    cp: distance between centroid and plane
    ap: distance between all atoms and plane, return 1D array
    '''
    g1=Geometry(A)
    g2=Geometry(B)
    if mtype == 'cc':
        return norm(g1.C-g2.C)
    if mtype == 'aa':
        return distance_matrix(g1.coord,g2.coord)
    if mtype == 'ac':
        v = []
        for c in g1.coord:
            v.append(norm(c-g2.C))
        return np.array(v)
    if mtype == 'cl':
        # distance(x=a+tn,p) = ||(a-p)-((a-p).n)n||
        return norm(g2.C - g1.C - np.dot(g2.C - g1.C, g2.L) * g2.L)
    if mtype == 'al':
        v = []
        for c in g1.coord:
            v.append(norm(g2.C-c-np.dot(g2.C-c,g2.L)*g2.L))
        return np.array(v)
    if mtype == 'cp':
        # distance (x=a+tn,p) = ||(a-p).n||
        return norm(np.dot(g2.C-g1.C,g2.Pnorm))
    if mtype == 'ap':
        # distance (x=a+tn,p) = ||(a-p).n||
        v = []
        for c in g1.coord:
            v.append(norm(np.dot(g2.C - c, g2.Pnorm)))
        return np.array(v)


def angle(A,B,C=[],mtype='ccc'):
    '''
    ccc: angle between three centroid
    lp: angle between line and plane
    pp: angle bewteen two plane
    '''
    g1 = Geometry(A)
    g2 = Geometry(B)
    if mtype == 'ccc':
        g3 = Geometry(C)
        v1 = g3.C - g2.C
        v2 = g1.C - g2.C
        return __ang2__(v1, v2)
    if mtype == 'lp':
        return 90-abs(__ang__(g1.L,g2.Pnorm))
    if mtype == 'pp':
        return abs(__ang__(g1.Pnorm, g2.Pnorm))

def torsion(A,B,C,D):
    ''' The torsion angle between 4 atoms A-B-C-D is the angle by which the vector A-B
    must be rotated in order to eclipse the vector C-D when viewed along the vector B-C.
    Crystallographers usually express torsion angles in the range -180 to +180 degrees.
    According to convention a clockwise rotation is positive and an anti-clockwise rotation is negative.
    '''
    g1 = Geometry(A)
    g2 = Geometry(B)
    g3 = Geometry(C)
    g4 = Geometry(D)
    p0 = g1.C
    p1 = g2.C
    p2 = g3.C
    p3 = g4.C
    b0 = -1.0*(p1-p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))




