import copy
import numpy as np
import networkx as nx
from .measurement import torsion

'''note that different function requires different input and return different result 
   function contain st means it requires a structure object and will return a structure
   function contain mol mean it requires a graph object and will return a graph
   other functions will take coordinate as input are return transform matrix
'''


__all__ = [
    'shift_st_abc',
    'shift_st_xyz',
    'rotate_st_by',
    'rotate_mol_dihedral',
    'translate_to_align',
    'rotate_to_align',
    'rotate_cw_around',
    'rotate_dihedral_to',
    'translate_along',
]

def shift_st_abc(struct, abc):
    '''abc is a list of three numbers. Means shift of fraction coords in a
    b c cell axis'''
    s = copy.deepcopy(struct)
    s.setter('fcoord', (np.array(s.fcoord) + np.array(abc)).tolist())
    s.frac2cart()
    return s

def shift_st_xyz(struct, xyz, pbc=False):
    '''xyz is a list of three numbers. Means shift of cart coords in x y z'''
    s = copy.deepcopy(struct)
    s.setter('coord', (np.array(s.coord) + np.array(xyz)).tolist())
    if s.period_flag == 1:
        s.cart2frac()
        if pbc == True:
            s.wrap_in_fcoord()
            s.frac2cart()
    return s

def rotate_st_by(struct, abc, center='g'):
    '''abc is a list of three numbers in degrees.
    Means rotation by angle a b c about x y z axis, respectively
    if centor = g, the geometry centor will be used
    if center = m, the mass centor will be used
    '''
    mol = copy.deepcopy(struct)
    if center == 'g':
        shift_v = np.array([0, 0, 0]) - np.array(mol.geom_center)
    elif center == 'm':
        shift_v = np.array([0, 0, 0]) - np.array(mol.mass_center)
    else:
        shift_v = np.array([0, 0, 0])
    mol = mol.T.shift_xyz(shift_v)
    s = np.sin(np.deg2rad(np.array(abc)))
    c = np.cos(np.deg2rad(np.array(abc)))
    rx = np.array([[1, 0, 0], [0, c[0], -1*s[0]], [0, s[0], c[0]]])
    ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-1*s[1], 0, c[1]]])
    rz = np.array([[c[2], -1*s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
    coords = np.array(mol.coord) @ rx.T @ ry.T @ rz.T
    mol.setter('coord', coords.tolist())
    mol = mol.T.shift_xyz(-1*shift_v)
    return mol

def translate_to_align(A,B):
    '''
    A and B are two points in space, both are represented by [x, y, z, 1]
    translation matrix transform A to B
    the maxtrix M has the form
    1  0  0  0
    0  1  0  0
    0  0  1  0
    dx dy dz 1
    so that A@M = B
    '''
    if len(A) == 3:
        A=np.append(A,1)
    if len(B) == 3:
        B=np.append(B,1)
    delta = np.array(B) - np.array(A)
    m = np.eye(4)
    m[3][:3] = delta[:3]
    return m

def rotate_to_align(A,B,C=[0,0,0,1]):
    '''
    A and B are two vectors in space, represent by [x, y, z, 1]
    rotation matrix rotated A to align B (max their dot product)
    C is the point to rotate around
    the maxtrix M has the form
    0  0  0  0
    0  0  0  0
    0  0  0  0
    0  0  0  1
    so that A@M = A'  where cross(A',B)=0
    '''
    if len(A) == 3:
        A=np.append(A,1)
    if len(B) == 3:
        B=np.append(B,1)
    if len(C) == 3:
        C=np.append(C,1)
    A = np.array(A)-np.array(C)
    B = np.array(B)-np.array(C)
    m0 = translate_to_align(C,[0,0,0,1])
    m1 = translate_to_align([0,0,0,1],C)
    A=np.array(A[:3])/np.linalg.norm(A[:3])
    B=np.array(B[:3])/np.linalg.norm(B[:3])
    axis = np.cross(A,B)
    ax,ay,az=axis
    cosA = np.dot(A,B)
    k = 1.0 / (1.0+cosA+0.000001)
    m3 = np.array([[ax*ax*k+cosA,ay*ax*k-az,az*ax*k+ay],
                  [ax*ay*k+az,ay*ay*k+cosA,az*ay*k-ax],
                  [ax*az*k-ay,ay*az*k+ax,az*az*k+cosA]]).T
    m4 = np.eye(4)
    m4[:-1,:-1] = m3
    return m0@m4@m1


def rotate_cw_around(A,B,angle=0):
    '''rotate around axis defined by A and B by angle (in degree)
    where  A is near the observer and the rotation will be cw'''
    # fist move to A to origin
    if len(A) == 3:
        A=np.append(A,1)
    if len(B) == 3:
        B=np.append(B,1)
    m0 = translate_to_align(A,[0,0,0,1])
    m1 = translate_to_align([0,0,0,1],A)
    axis = np.array(B[:3]) - np.array(A[:3])
    axis = axis/np.linalg.norm(axis)
    ax,ay,az=axis
    s = np.sin(np.deg2rad(angle))
    c = np.cos(np.deg2rad(angle))
    d = 1 - c
    m = [[c+ax*ax*d,ax*ay*d-az*s,ax*az*d+ay*s,0],
         [ay*ax*d+az*s,c+ay*ay*d,ay*az*d-ax*s,0],
         [az*ax*d-ay*s,az*ay*d+ax*s,c+az*az*d,0],
         [0,0,0,1]]
    return m0@np.array(m).T@m1

def rotate_dihedral_to(A,B,C,D,angle=0):
    '''A B C D are 4 points in space
    return a rotate matrix to
    rotate A so that dihedral angle of  A-B-C-D is the specified angle
    the convention is that if target angle T > 0
    and observer view from B to C, the A need to rotate cw by T to eclipse with D
    '''
    # first calculate current angle
    current_angle = torsion(A,B,C,D)
    delta_angle = current_angle - angle
    return rotate_cw_around(B,C,angle=delta_angle)


def translate_along(A,B,dist=0):
    '''translate along direction defined by B-A by distance of dist'''
    if len(A) == 3:
        A=np.append(A,1)
    if len(B) == 3:
        B=np.append(B,1)
    v = (B-A)[:3]
    lenv = np.linalg.norm(v)
    v*(lenv+dist)/lenv
    m0 = translate_to_align(A,[0,0,0,1])
    m1 = translate_to_align([0,0,0,1],A)
    m4 = translate_to_align(v,v*(lenv+dist)/lenv)
    return m0@m4@m1

def rotate_mol_dihedral(graph,bond,angle):
    '''input is a graph
       return a new graph object with updated coords
       bond has the format 1-13, and the fragment contain atom sn 13
       will be near the observer and rotate around 1-13 cw by angle
    '''
    G = copy.deepcopy(graph.copy())
    a1,a2 = [int(i) for i in bond.split('-')]
    c2,c1 = [G.nodes[i]['coord'] for i in [a2,a1]]
    rot_mat = rotate_cw_around(c2,c1,angle=angle)
    try:
        G.remove_edge(a1,a2)
    except nx.NetworkXError:
        print('Warning! the bond {:s} to rotate around is not exits'.format(bond))
    if len(list(nx.connected_components(G))) == 2:
       for sn in  nx.connected_components(G):
           if a2 in sn:
               coords = np.array([G.nodes[i]['coord']+[1] for i in sn])
               new_coords = coords @ rot_mat
               new_sn2coord = {k:{'coord':list(v)} for k,v in zip(sn,new_coords[:,:3])}
               nx.set_node_attributes(G,new_sn2coord)

    return G





