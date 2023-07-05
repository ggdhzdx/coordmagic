import networkx as nx

from .layer import Layer
from .molgraph import MolGraph
from .labelatom import LabelAtom
from .cell import *
from .structurewriter import StructureWriter
from .parameter import Parameter
import copy
import os
import sys
import numpy as np
from collections import defaultdict



class Structure:

    def __init__(self, elem=[],atomnum=[],coord=[],fcoord=[],cell_vect=[],cell_param=[]):
        self.basename = ''
        self.atoms = []  # list of atom, each atom is a defaultdict object, this is a core property
        self.period_flag = 0  # weather periodic structure
        self.cell_vect = cell_vect
        self.cell_param = cell_param
        # following are properties extract from self.atoms in complete_self method
        self.coord = coord
        self.fcoord = fcoord
        self.elem = elem
        self.atomnum = atomnum
        self.sn = []  # serial number of atoms. Duplicates indicate image atoms
        if len(self.atoms) == 0:
            if len(self.elem) > 0:
                self.atoms = [defaultdict(str,{'elem':i}) for i in self.elem]
            elif len(self.atomnum) > 0:
                self.atoms = [defaultdict(str,{'atomnum':int(i)}) for i in self.atomnum]
        if len(self.coord) > 0:
            self.setter('coord',self.coord)
        if len(self.fcoord) > 0:
            self.setter('fcoord',self.fcoord)
        self.cell_origin = [0, 0, 0]
        # frames is a defaultdict will contain multiple atom lists
        self.frames = defaultdict(list,{1:self.atoms})
        self.frames_box = defaultdict(list,{1:self.cell_param})
        self.frame_sn = 1
        # hidden properties
        self._sn2atom = {} # dictionary map atom sn to atom dict
        # following are groups of tools to  analysis structure
        #self.T = Transformer(self)
        #self.L = Layer(self)
        self.P = Parameter() # store all constant like atom radius, vdw parameters...
        self.L = LabelAtom(self) # properties or descriptors that mapped to atom
        self.G = MolGraph(self)
        #self.MC = Cluster(self)
        #self.ML = label(self)
        #self.MM = Measurement(self)
        self.molecules = []
        self.graph = ''
        self.supergraph = ''
        self.mol_list = ''

    def choose_frame(self,n):
        '''the number in different frames may be different
        if n is not in self.frame then an empty structure
        the complete_self will store atomslist and cellparam to frames and frames_box'''
        if n < 0:
            current_max = max(self.frames.keys())
            n = current_max + 1 + n
        self.frame_sn = n
        self.atoms = self.frames[n]
        if len(self.frames_box[n]) == 6:
            self.cell_param = self.frames_box[n]
        self.complete_self(reset_vect=False)

    # def new_frame(self, n=0):
    #     '''set current frame to new frame
    #     the serial number of new frame is current max serial number plus one'''
    #     current_max = max(self.frames.keys())
    #     if n == 0:
    #         n = current_max + 1
    #     self.frames[self.frame_sn] = copy.deepcopy(self.frames[self.frame_sn])
    #     if len(self.frames_box[self.frame_sn]) == 6:
    #         self.frames_box[self.frame_sn] = copy.deepcopy(self.frames_box[self.frame_sn])
    #     self.choose_frame(n)

    def getter(self, prop_name):
        '''get prop_name from all atoms and form a list
        also set the structure attribute
        '''
        prop = [atom[prop_name] for atom in self.atoms]
        return prop

    def setter(self, prop_name, value):
        '''set prop_name for all atoms with value'''
        if len(value) < len(self.atoms) or isinstance(value, str):
            prop_value = len(self.atoms) * [value]
        elif len(value) == len(self.atoms):
            prop_value = value
        else:
            print("Set {:s} for {:s} Error! More values({:d}) than atoms({:d})"
                  .format(prop_name, self.basename, len(value), len(self.atoms)))
        setattr(self, prop_name, prop_value)
        for i, atom in enumerate(self.atoms):
            atom[prop_name] = prop_value[i]

    def __str__(self):
        if not self.formula:
            return ''
        if self.period_flag:
            return (self.formula +
            " in a={:.1f},b={:.1f},c={:.1f},alpha={:.1f},beta={:.1f},gamma={:.1f}".format(*self.cell_param))
        else:
            return (self.formula + ' in vacuum')

    def complete_self(self, wrap=False, reset_vect=True):
        '''re-Generate coords fcoords elems atomnums sn attribute from self.atoms
        generate atomnum from elem or elem from atomnum
        if both atomum or elem are missing, generate them from atomname
        generate cell_param from cell_vect or cell_vect from cell_param
        generate fcoord from coord or coord from fcoord
        reset_vect = True means reset a to x axis and b in xy plane
        wrap = True means move atoms outside cell inside. Cannot working when atoms with
        duplicate sn exist (image atom).
        '''
        self.coord = self.getter('coord')
        self.fcoord = self.getter('fcoord')
        self.elem = self.getter('elem')
        self.atomnum = self.getter('atomnum')
        self.sn = self.getter('sn')
        self.frames[self.frame_sn] = self.atoms
        self.frames_box[self.frame_sn] = self.cell_param
        self.frames = defaultdict(list,{k:v for k,v in self.frames.items() if len(v)>0})
        self.frames_box = defaultdict(list,{k:v for k,v in self.frames_box.items() if len(v)>0})

        if not all(self.sn):
            self.reset_sn()

        if all(self.elem) and not all(self.atomnum):
            self.elem2an()
        elif all(self.atomnum) and not all(self.elem):
            self.an2elem()
        elif not all(self.elem) and not all(self.atomnum):
            self.atomname = self.getter('atomname')
            if all(self.atomname):
                self.name2elem()
                self.elem2an()
            else:
                print('no atoms found for {:s}'.format(self.basename))

        if len(self.cell_param) == 0 and len(self.cell_vect) == 0:
            self.period_flag = 0
        else:
            self.period_flag = 1
            if len(self.cell_vect) == 3 and len(self.cell_param) < 6:
                self.vect2param()
                self.frames_box[self.frame_sn] = self.cell_param
            elif len(self.cell_param) == 6 and len(self.cell_vect) < 3:
                self.param2vect()
            elif len(self.cell_param) < 6 and len(self.cell_vect) < 3:
                print('Cell_vect or cell_param is not right, the contents are:')
                print(self.cell_param)
                print(self.cell_vect)
            if np.isnan(self.cell_param).any():
                self.cell_param = []
                self.cell_vect = []
                self.period_flag = 0
            else:
                # fill empty coord
                if reset_vect:
                    self.param2vect()
                if len(self.coord[0]) == 3 and len(self.fcoord[0]) == 0:
                    self.cart2frac()
                elif len(self.fcoord[0]) == 3 and len(self.coord[0]) == 0:
                    self.frac2cart()
                elif len(self.fcoord[0]) == 0 and len(self.coord[0]) == 0:
                    print('no coordinates found')
                if wrap and len(self.sn) == len(set(self.sn)):
                    self.wrap_in_fcoord()
                    self.frac2cart()

    def reset_sn(self):
        '''reset sn key in defaultdict by the order of atom in self.atoms list'''
        new_sn = list(range(1, len(self.atoms)+1))
        old_sn = self.getter('sn')
        self.setter('sn', new_sn)
        if self.graph:
            mapping = {k:v for k,v in zip(old_sn,new_sn)}
            nx.set_node_attributes(self.graph,mapping,'sn')
            nx.relabel_nodes(self.graph,mapping)

    def elem2an(self):
        atomnum = [self.P.elem2an[i[0].upper()+i[1:].lower()] for i in self.elem]
        self.setter('atomnum', atomnum)

    def name2elem(self):
        cleared_elems = []
        for a in self.atomname:
            elem = ''.join([i for i in a if not i.isdigit()])
            elem = elem[0].upper()+elem[1:].lower()
            if elem in self.P.elem2an:
                cleared_elems.append(elem)
            elif elem[0] in self.P.elem2an:
                cleared_elems.append(elem[0])
            else:
                print('Error!!! atom name {:s} could not convert to element'.format(a))
                sys.exit()
        self.setter('elem', cleared_elems)

    def an2elem(self):
        elem = [self.P.an2elem[int(i)] for i in self.atomnum]
        self.setter('elem', elem)

    def param2vect(self):
        a, b, c, alpha, beta, gamma = self.cell_param
        va = [a, 0.0, 0.0]
        vb = [b*np.cos(np.radians(gamma)), b*np.sin(np.radians(gamma)), 0.0]
        angle2Y = np.arccos(np.cos(np.radians(alpha))*np.cos(np.pi/2-np.radians(gamma)))
        cx = c*np.cos(np.radians(beta))
        cy = c*np.cos(angle2Y)
        cz = (c**2-cx**2-cy**2)**0.5
        vc = [cx, cy, cz]
        self.cell_vect = [va, vb, vc]

    def vect2param(self):
        va, vb, vc = self.cell_vect
        a = np.linalg.norm(va)
        b = np.linalg.norm(vb)
        c = np.linalg.norm(vc)
        if a == 0 or b == 0 or c == 0:
            print("Warning! One of the cell edge has zero length, the system will be considered as non periodic")
        alpha = np.rad2deg(np.arccos(np.dot(vc, vb)/(c*b)))
        beta = np.rad2deg(np.arccos(np.dot(va, vc)/(a*c)))
        gamma = np.rad2deg(np.arccos(np.dot(va, vb)/(a*b)))
        self.cell_param = [a, b, c, alpha, beta, gamma]

    def wrap_in_fcoord(self):
        for coord in self.fcoord:
            for i in range(len(coord)):
                if coord[i] < 0:
                    coord[i] += np.ceil(abs(coord[i]))
                if coord[i] >= 1:
                    coord[i] -= np.floor(coord[i])
        self.setter('fcoord', self.fcoord)

    def frac2cart(self, fcoord = []):
        if len(fcoord) > 0:
            coord = np.matmul(np.array(fcoord), np.array(self.cell_vect))
            coord = (coord +np.array(self.cell_origin)).tolist()
            return coord
        else:
            coord = np.matmul(np.array(self.fcoord), np.array(self.cell_vect))
            coord = (coord +np.array(self.cell_origin)).tolist()
            self.setter('coord', coord)

    def cart2frac(self,coord = []):
        if len(coord) > 0:
            coord = np.array(coord) - np.array(self.cell_origin)
            fcoord = np.matmul(np.array(coord), np.linalg.inv(np.array(self.cell_vect))).tolist()
            return fcoord
        else:
            coord = np.array(self.coord) - np.array(self.cell_origin)
            fcoord = np.matmul(np.array(coord), np.linalg.inv(np.array(self.cell_vect))).tolist()
            self.setter('fcoord', fcoord)

    def add_atom(self, atom):
        '''add a single atom
        atom is a default dict object'''
        s = copy.copy(self)
        s.atoms.append(atom)
        if atom['sn'] == '':
            atom['sn'] = max(s.sn) + 1
        s.complete_self()
        return s

    def remove_atom(self, sn, reset_sn=False):
        '''idx is a list of atom sn '''
        s = copy.copy(self)
        s.atoms = [i for i in s.atoms if i['sn'] not in list(sn)]
        s.complete_self()
        if reset_sn == True:
            s.reset_sn()
        return s

    def add_struc(self, struc, reset_sn=False):
        '''add structure object to the  structure
        the cell_param in structure object will be omitted'''
        s = copy.copy(self)
        s.atoms = s.atoms + struc.atoms
        s.complete_self()
        if reset_sn == True:
            s.reset_sn()
        return s

    def extract_struc(self, sn, reset_sn=False):
        '''sn is a list of atom sn '''
        s = Structure()
        s.atoms = [i for i in self.atoms if i['sn'] in sn]
        s.cell_vect = self.cell_vect
        s.cell_param = self.cell_param
        s.complete_self()
        if self.S.graph:
            s.graph = copy.deepcopy(nx.subgraph(self.S.graph,sn).copy())
        if reset_sn == True:
            s.reset_sn()
        return s

    def get_atom_by(self,**kwargs ):
        '''get a list of atom by key value pair
        where key is an atom property and value is a list'''
        atoms = []
        for key,value in kwargs.items():
            atoms += [i for i in self.atoms if i[key] in value]
        return atoms

    def get_atom(self,sn,reset_dict=False):
        '''get a list of atom by key value pair
        where key is an atom property and value is a list'''
        if reset_dict:
            self._sn2atom = {i['sn']:i for i in self.atoms}
        try:
            atoms = [self._sn2atom[i] for i in sn]
        except KeyError:
            self._sn2atom = {i['sn']:i for i in self.atoms}
            atoms = [self._sn2atom[i] for i in sn]
        return atoms


    def write_structure(self, basename=None, append=False, ext='', options=''):
        sw = StructureWriter()
        sw.st = self
        if not basename:
            basename = sw.st.basename
        if not ext:
            basename, ext = os.path.splitext(basename)
            if not ext:
                ext = 'pdb'
            else:
                ext = ext[1:]
        else:
            basename = basename
            ext = ext
        sw.basename = basename
        filename = basename + '.' + ext
        if append:
            print('Appending to file {:s}'.format(filename))
            sw.file = open(filename, 'a')
        else:
            print('Generating file {:s}'.format(filename))
            sw.file = open(filename, 'w')
        if ':' in options:
            sw.options = defaultdict(str, {v.split(':')[0]: v.split(':')[1] for v in options.split(';')})
        else:
            sw.options = defaultdict(str)
        if ext in sw.write_func:
            sw.write_func[ext]()
        else:
            print('Format {:s} is not supported yet'.format(ext))
            sys.exit()
        sw.file.close()

    @property
    def thick(self):
        return self.top-self.button

    @property
    def z(self):
        return np.mean(np.array(self.coord).T[2])

    @property
    def top(self):
        return np.max(np.array(self.coord).T[2])

    @property
    def geom_center(self):
        return np.average(np.array(self.coord),axis=0)

    @property
    def mass_center(self):
        weights = np.array([self.P.elem2mass[x] for x in self.elem])
        center_of_mass = np.average(np.array(self.coord), axis=0, weights=weights)
        return center_of_mass

    @property
    def inertia_tensor(self):
        coord = np.array(self.coord) - self.mass_center
        Ixx,Iyy,Izz,Ixy,Ixz,Iyz = [0,0,0,0,0,0]
        for i,c in enumerate(coord):
            m = self.P.elem2mass[self.elem[i]]
            x,y,z = c
            Ixx += m * (y**2 + z**2)
            Iyy += m * (x**2 + z**2)
            Izz += m * (x**2 + y**2)
            Ixy += -1 * m * x * y
            Ixz += -1 * m * x * z
            Iyz += -1 * m * y * z
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    @property
    def principal_inertia_axis(self):
        '''retrun  array of principle inertia axis
        ranged from large eigen_val (moment of inertia) to small eigen_val'''
        eigval, eigvec = np.linalg.eig(self.inertia_tensor)
        sort = np.argsort(eigval)[::-1]
        eigvec = eigvec[:,sort].T
        # right hand convention
        if np.dot(np.cross(eigvec[0], eigvec[1]), eigvec[2]) < 0:
            eigvec *= -1
        return eigvec
    @property
    def principal_geom_axis(self):
        '''retrun array of geometry axis from svd on coords
        ranged from long axis to short axis'''
        coord = np.array(self.coord) - self.geom_center
        u,s,vh  = np.linalg.svd(coord)
        return vh

    @property
    def button(self):
        return np.min(np.array(self.coord).T[2])

    @property
    def valence_electron(self):
        nele = 0
        for d in self.atomnum:
            if d > 86:
                nele = nele + d - 86
            elif d > 54:
                nele = nele + d - 54
            elif d > 36:
                nele = nele + d - 36
            elif d > 18:
                nele = nele + d - 18
            elif d > 10:
                nele = nele + d - 10
            else:
                nele = nele + d
        return nele

    @property
    def formula(self):
        # generate formula in cell
        c = [[c, self.elem.count(c)] for c in set(self.elem)]
        sc = sorted(c, key=lambda x: self.P.elem2an[x[0]], reverse=True)
        formula = ''.join([i[0]+str(i[1]) for i in sc])
        return formula

    def sort_atoms(self, by='z', reset_sn=False):
        s = copy.deepcopy(self)
        indexes = list(range(len(self.atoms)))
        if by == 'z':
            indexes.sort(key=lambda x: self.coord[x][2])
        if by == 'elem':
            indexes.sort(key=self.elem.__getitem__)
        if by == 'an':
            indexes.sort(key=self.atomnum.__getitem__)
        s.atoms = list(map(self.atoms.__getitem__, indexes))
        s.complete_self()
        if reset_sn == True:
            s.reset_sn()
        return s
