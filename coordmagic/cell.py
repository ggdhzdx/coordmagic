import copy
import numpy as np
from . import structure

'''metheds that modify lattice parameters of structure object'''

__all__ = [
    "set_cell_param",
    "switch_edge",
    "delete_cell",
    "shift_cell_origin",
    "shift_cell_by",
    "super_cell",
    "add_image_atom",
]

def set_cell_param(struct, cell_param, keep='cart'):
    '''set new cell parameter. If keep is cart, the cartesian coord will be
    kept and if keep is frac, the fraction coord whill be kept'''
    s = copy.deepcopy(struct)
    s.cell_param = cell_param
    s.param2vect()
    if keep.startswith('c'):
        s.cart2frac()
    if keep.startswith('f'):
        s.frac2cart()
    return s

def switch_edge(struct, order=[1, 0, 2]):
    new_frac = []
    new_vect = []
    s = copy.deepcopy(struct)
    for i in range(3):
        new_vect.append(s.cell_vect[order[i]])
        new_frac.append(np.array(s.fcoord).T[order[i]])
    s.cell_vect = new_vect
    s.setter('fcoord', np.array(new_frac).T.tolist())
    s.vect2param()
    s.param2vect()
    s.frac2cart()
    return s

def delete_cell(struct,self):
    s = copy.deepcopy(struct)
    s.setter('fcoord', '')
    s.cell_vect = []
    s.cell_param = []
    s.period_flag = 0
    return s

def shift_cell_origin(struct, frac_origin, cart_origin=None):
    '''shift cell to new_origin, new_origin is a three integer in frac coord'''
    s = copy.deepcopy(struct)
    if not cart_origin:
        cart_origin = (np.matrix(frac_origin) * np.matrix(s.cell_vect)).tolist()
    s.cell_origin = cart_origin
    s.frac2cart()
    return s

def shift_cell_by(struct,cell_pos):
    '''shift coord to position defined by cell pos
    e.g. [0,0,1]'''
    if any(np.array(cell_pos) != 0):
        s = struct
        fcoord = np.array(s.fcoord) + np.array(cell_pos)
        s.setter('fcoord', fcoord.tolist())
        s.frac2cart()

def super_cell(struct, trans_mat):
    '''format of trans_mat:
    u = n1*a+n2*b+n3*c
    v = m1*a+m2*b+m3*c
    w = p1*a+p2*b+p3*c
    trans_mat = [[n1,n2,n3],[m1,m2,m3],[p1,p2,p3]]
    do not support negative value in trans_mat
    '''
    s = struct
    slat = structure.Structure()
    na = trans_mat[0][0]
    nb = trans_mat[1][1]
    nc = trans_mat[2][2]
    slat.basename = s.basename
    new_frac = []
    prim_frac = np.array(s.fcoord)
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                new_frac.append(prim_frac + np.array([i, j, k]))
    all_frac = np.concatenate(new_frac)
    #coord = np.matrix(all_frac) * np.matrix(s.cell_vect)
    #coord = (coord.A+np.array(s.cell_origin)).tolist()
    for i in range(abs(na*nb*nc)):
        slat.atoms = slat.atoms + [copy.deepcopy(a) for a in s.atoms]
    slat.cell_vect = (np.matrix(trans_mat)*np.matrix(s.cell_vect)).tolist()
    slat.setter('fcoord', (np.matrix(all_frac)*np.matrix(trans_mat).I).tolist())
    slat.reset_sn()
    slat.complete_self(reset_vect=False)
    slat.prim_cell_param = s.cell_param
    slat.prim_cell_vect = s.cell_vect
    slat.trans_mat = trans_mat
    return slat

def add_image_atom(struct, expand_length):
    '''return cart coord in expand cell
    expand_length is the distance between the face of expanded cell and
    the origin cell. It can be one number or a list of three numbers.
    '''
    s = copy.deepcopy(struct)
    s.wrap_in_fcoord()
    def angle(a, b, c):
        '''calculate angle between a and norm of bc plane'''
        d = np.cross(b, c)
        len_a = np.linalg.norm(a)
        len_d = np.linalg.norm(d)
        angle = np.arccos(np.dot(a, d)/(len_a*len_d))
        return angle
    v1, v2, v3 = np.array(s.cell_vect)
    angle_abc = np.array([angle(v1, v2, v3), angle(v2, v1, v3), angle(v3, v1, v2)])
    # angle a is a to norm of bc plane; angle b is b to norm of ac plane ...
    iparam = np.abs(np.array(expand_length)/np.cos(angle_abc)/np.array(s.cell_param[:3]))
    fa, fb, fc = iparam
    ifrac = []
    iatoms = []
    isn = []
    for i, f in enumerate(s.fcoord):
        for a in range(int(np.floor(-1*fa)), int(np.ceil(1+fa))):
            for b in range(int(np.floor(-1*fb)), int(np.ceil(1+fb))):
                for c in range(int(np.floor(-1*fc)), int(np.ceil(1+fc))):
                    new_f = np.array(f) + np.array([a, b, c])
                    if (all(new_f > iparam*-1) and all(new_f < iparam+1)  # within expaned cell
                            and (any(new_f < 0) or any(new_f >= 1))):  # not in origin cell
                        ifrac.append(new_f.tolist())
                        iatoms.append(s.atoms[i])
                        isn.append(s.atoms[i]['sn'])
    icart = np.matmul(np.array(ifrac),np.array(s.cell_vect))
    icart = (icart+np.array(s.cell_origin)).tolist()
    s.atoms = s.atoms + iatoms
    s.setter('fcoord', s.fcoord+ifrac)
    s.setter('sn', s.sn+isn)
    s.setter('coord', s.coord+icart)
    s.elem = s.getter('elem')
    s.atomnum = s.getter('atomnum')

    # for i,e in enumerate(s.atoms):
    #     if i+1!=e['sn']:
    #         print(i,e['sn'])
    return s

def set_plane(struct):
    '''set two axis as 2D lattice. These two axis will be reassigned as a and b
    and a aixs will be aligned to x '''
    pass

def shear_c(struct):
    '''shear c axis to z direction'''
    pass