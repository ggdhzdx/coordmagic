from typing import List, Union, Any

import networkx as nx

from .layer import Layer
from .molgraph import MolGraph
from .labelatom import LabelAtom
from .cell import *
from .structurewriter import *
from .parameter import Parameter
import copy
import os
import sys
import numpy as np
from collections import defaultdict

### todo
# lazzy complete self


class Structure:


    def __init__(self, atoms: list=None, elem: list=None, atomnum: list=None, coord: list=None, fcoord: list=None, cell_vect: list=None, cell_param:list=None):
        if atoms is None:
            atoms = []
        if cell_param is None:
            cell_param = []
        if cell_vect is None:
            cell_vect = []
        if fcoord is None:
            fcoord = []
        if coord is None:
            coord = []
        if elem is None:
            elem = []
        if atomnum is None:
            atomnum = []
        self.basename = ''
        self.atoms = atoms  # list of atom, each atom is a defaultdict object, this is the core property
        self._is_periodic = 0  # whether periodic structure
        self._cell_vect = []
        self._cell_param = []
        self.cell_vect = cell_vect
        self.cell_param = cell_param
        # following are properties extract from self.atoms in complete_self method
        # variable record for state change, used for lazy property calculation
        self.__recalc_coord = 0
        self.__recalc_fcoord = 0
        if len(self.atoms) == 0:
            if len(elem) > 0:
                self.elem = elem
            elif len(self.atomnum) > 0:
                self.atomnum = atomnum
        if len(coord) > 0:
            self.coord = coord
        elif len(fcoord) > 0:
            self.fcoord = fcoord
        self.cell_origin = [0, 0, 0]
        # frames is a defaultdict will contain multiple structure object
        self.frames = [self]
        # hidden properties
        self._sn2atom = {} # dictionary map atom sn to atom dict
        # following are groups of tools to  analysis structure
        #self.T = Transformer(self)
        #self.L = Layer(self)
        self.param = Parameter() # store all constant like atom radius, vdw parameters...
        self.L = LabelAtom(self) # properties or descriptors that mapped to atom
        self.G = MolGraph(self)
        #self.MC = Cluster(self)
        #self.ML = label(self)
        #self.MM = Measurement(self)
        # list of molecules contain in the structure, only available after graph analysis
        self.molecules = None
        self.graph = None
        self.supergraph = None
        self.mol_list = None

    @property
    def frame_idx(self):
        return self.frames.index(self)

    def choose_frames(self, n: list):
        """return a list of st object from self.frames dict
        each st object has an updated frames attribute
        """
        new_frames = [self.frames[i] for i in n]
        for st in new_frames:
            st.frames=new_frames
        return new_frames

    def append_frame(self, st, frame_idx: list=None):
        """append another structure object st to the frame
        if frames is None, append all frames in st,
        else append frames specified by frame_idx such as [1,2,3,4] within st
        """
        if frame_idx is None:
            appended_frames = st.frames
        else:
            appended_frames = [st[i] for i in frame_idx]
        merged_frames = []
        for fs in [self.frames, appended_frames]:
            for f in fs:
                f.frames = merged_frames
                merged_frames.append(f)

    def new_frame(self,st=None, atoms: list=None, elem: list=None, atomnum: list=None, coord: list=None, fcoord: list=None, cell_vect: list=None, cell_param:list=None):
        if st is None:
            st=Structure(atoms=atoms, elem=elem, atomnum=atomnum, coord=coord, fcoord=fcoord, cell_vect=cell_vect, cell_param=cell_param)
        self.append_frame(st)
        return self.frames[-1]

    def del_frames(self, frame_idx=None):
        if frame_idx is None:
            frame_idx = []
        for idx in sorted(frame_idx,reverse=True):
            del self.frames[idx]

    def concat_frame(self, st_list, self_pos=0 ):
        """
        concat all frames from st in st_list
        and set update frames and frame_sn attribute in every st in st_list
        and the st_list will be returned
        if include_self is a number N, then the self will be put at N position in the st list
        """
        if self_pos in list(range(len(st_list))):
            st_list.insert(self_pos,self)
        merged_frames = []
        for st in st_list:
            for f in st.frames:
                f.frames = merged_frames
                merged_frames.append(f)
        return st_list

    def _getter(self, prop_name):
        """
        get prop_name from all atoms and form a list
        also set the structure attribute
        """
        prop = [atom[prop_name] for atom in self.atoms]
        return prop

    def _setter(self, prop_name, value: list):
        """
        set prop_name for all atoms in self.atoms with value
        """
        if len(value) == len(self.atoms):
            for i, atom in enumerate(self.atoms):
                atom[prop_name] = value[i]
        else:
            print("Set {:s} for {:s} Error! different number of values({:d}) from atoms({:d})"
                  .format(prop_name, self.basename, len(value), len(self.atoms)))
            sys.exit()

    @property
    def elem(self):
        return self._getter('elem')

    @elem.setter
    def elem(self, value):
        self._setter('elem', value)
        atomnum = [self.param.elem2an[i[0].upper() + i[1:].lower()] for i in value]
        self._setter('atomnum', atomnum)

    @property
    def atomnum(self):
        return self._getter('atomnum')

    @atomnum.setter
    def atomnum(self, value):
        self._setter('atomnum', value)
        elem = [self.param.an2elem[int(i)] for i in value]
        self._setter('elem', elem)

    def _frac2cart(self):
        """
        update coord from fcoord for all atom in self.atoms
        """
        coord = np.matmul(np.array(self.fcoord), np.array(self.cell_vect))
        coord = (coord +np.array(self.cell_origin)).tolist()
        self._setter("coord",coord)

    def _cart2frac(self):
        """
        update fcoord from coord for all atom in self.atoms
        """
        coord = np.array(self.coord) - np.array(self.cell_origin)
        fcoord = np.matmul(np.array(coord), np.linalg.inv(np.array(self.cell_vect))).tolist()
        self._setter('fcoord', fcoord)

    @property
    def coord(self):
        if self._is_periodic:
            if self.__recalc_coord == 1:
                self._frac2cart()
                self.__recalc_coord = 0
            elif any(not c for c in self._getter('coord')):
                self._frac2cart()
        return self._getter('coord')

    @coord.setter
    def coord(self, value):
        self._setter("coord",value)
        self.__recalc_fcoord = 1

    @property
    def fcoord(self):
        if self._is_periodic:
            if self.__recalc_fcoord == 1:
                self._cart2frac()
                self.__recalc_fcoord = 0
            elif any(not c for c in self._getter('fcoord')):
                self._cart2frac()
        return self._getter('fcoord')

    @fcoord.setter
    def fcoord(self, value):
        self._setter("fcoord",value)
        self.__recalc_coord = 1

    @property
    def sn(self):
        return self._getter('sn')

    @property
    def atomname(self):
        return self._getter('atomname')

    @atomname.setter
    def atomname(self,value):
        self._setter("atomname",value)

    def _param2vect(self):
        a, b, c, alpha, beta, gamma = self.cell_param
        va = [a, 0.0, 0.0]
        vb = [b*np.cos(np.radians(gamma)), b*np.sin(np.radians(gamma)), 0.0]
        cx = c*np.cos(np.radians(beta))
        cy = c*np.cos(np.arccos(np.cos(np.radians(alpha))*np.cos(np.pi/1-np.radians(gamma))))
        cz = (c**2-cx**2-cy**2)**0.5
        vc = [cx, cy, cz]
        self._cell_vect = [va, vb, vc]

    def _vect2param(self):
        va, vb, vc = self.cell_vect
        a = np.linalg.norm(va)
        b = np.linalg.norm(vb)
        c = np.linalg.norm(vc)
        if a == 0 or b == 0 or c == 0:
            print("Warning! One of the cell edge has zero length, the system will be considered as non periodic")
            self._cell_param = []
            return
        alpha = np.rad2deg(np.arccos(np.dot(vc, vb)/(c*b)))
        beta = np.rad2deg(np.arccos(np.dot(va, vc)/(a*c)))
        gamma = np.rad2deg(np.arccos(np.dot(va, vb)/(a*b)))
        self._cell_param = [a, b, c, alpha, beta, gamma]

    @property
    def cell_param(self):
        return self._cell_param

    @cell_param.setter
    def cell_param(self,value: list):
        if len(value) == 0:
            return
        elif len(value) != 6:
            print("Error! the format of cell_param should be [a,b,c,alpha,beta,gamma], but the input is:")
            print(value)
            sys.exit()
        else:
            self._cell_param = list(value)
            self._is_periodic = 1
            self._param2vect()

    @property
    def cell_vect(self):
        return self._cell_vect

    @cell_vect.setter
    def cell_vect(self,value: list):
        if len(value) == 0:
            return
        elif len(value) != 3:
            print("Error! the shape of cell_vect should be 3 x 3, but the input is:")
            print(value)
            sys.exit()
        else:
            for v in value:
                if len(v) != 3:
                    print("Error! the shape of cell_vect should be 3 x 3, but the input is:")
                    print(value)
                    sys.exit()
            self._cell_vect = list(value)
            self._is_periodic = 1
            self._vect2param()

    def _name2elem(self):
        cleared_elems = []
        a: str
        for a in self.atomname:
            elem = ''.join([i for i in a if not i.isdigit()])
            elem = elem[0].upper()+elem[1:].lower()
            if elem in self.param.elem2an:
                cleared_elems.append(elem)
            elif elem[0] in self.param.elem2an:
                cleared_elems.append(elem[0])
            else:
                print('Error!!! atom name {:s} could not convert to element'.format(a))
                sys.exit()
        self._setter('elem', cleared_elems)

    def gen_atomname(self):
        element_counts = dict()
        names = []
        element: str
        for element in self.elem:
            if element not in element_counts:
                element_counts[element] = 1
            else:
                element_counts[element] += 1
            names.append("{:s}{:d}".format(element, element_counts[element]))
        self.atomname = names

    def complete_self(self, reset_vect=True):
        """
        Generate sn from order of atom in self.atoms, if sn is not all avail
        if both atomnum or elem are missing, generate them from atomname
        reset_vect = True means reset a to x axis and b in xy plane
        """
        if not all(self.sn):
            self.reset_sn()
        if not all(self.elem) and not all(self.atomnum):
            if all(self.atomname):
                self._name2elem()
            else:
                print('no atoms found for {:s}'.format(self.basename))
                # fill empty coord
        if self._is_periodic == 1:
            if reset_vect:
                self._param2vect()
            allc = all(len(c) == 3 for c in self.coord)
            allf = all(len(c) == 3 for c in self.fcoord)
            if allc and not allf:
                self._cart2frac()
            elif allf and not allc:
                self._frac2cart()
            elif not allf and not allc:
                print("Warning!!! something wrong in coord or fcoord")

    def reset_sn(self):
        """reset sn key in defaultdict by the order of atom in self.atoms list
        @rtype: None
        """
        new_sn = list(range(1, len(self.atoms)+1))
        old_sn = self._getter('sn')
        self._setter('sn', new_sn)
        if self.graph:
            mapping = {k:v for k,v in zip(old_sn,new_sn)}
            nx.set_node_attributes(self.graph,mapping,'sn')
            nx.relabel_nodes(self.graph,mapping)

    def wrap_in(self):
        if len(self.sn) == len(set(self.sn)):
            print("Error! You should not wrap atoms while there are image atoms")
            sys.exit()
        fc = self.fcoord
        c: list
        for c in fc:
            for i in range(len(c)):
                if c[i] < 0:
                    c[i] += np.ceil(abs(c[i]))
                if c[i] >= 1:
                    c[i] -= np.floor(c[i])
        self.fcoord = fc

    def add_atom(self, atom):
        """add a single atom which
        is a default dict object"""
        s = copy.copy(self)
        s.atoms.append(atom)
        if atom['sn'] == '':
            atom['sn'] = max(s.sn) + 1
        s.complete_self()
        return s

    def remove_atom(self, sn, reset_sn=False):
        """idx is a list of atom sn """
        s = copy.copy(self)
        s.atoms = [i for i in s.atoms if i['sn'] not in list(sn)]
        s.complete_self()
        if reset_sn:
            s.reset_sn()
        return s

    def add_struc(self, struc, reset_sn=False):
        """add structure object to the  structure
        the cell_param in structure object will be omitted"""
        s = copy.copy(self)
        s.atoms = s.atoms + struc.atoms
        s.complete_self()
        if reset_sn:
            s.reset_sn()
        return s

    def extract_struc(self, sn, reset_sn=False):
        """sn is a list of atom sn """
        s = Structure()
        s.atoms = copy.deepcopy([i for i in self.atoms if i['sn'] in sn])
        s.cell_vect = self.cell_vect
        s.cell_param = self.cell_param
        s.complete_self()
        if self.graph:
            s.graph = copy.deepcopy(nx.subgraph(self.graph,sn).copy())
        if reset_sn:
            s.reset_sn()
        return s

    def get_atom_by(self,**kwargs ):
        """get a list of atom by key value pair
        where key is an atom property and value is a list"""
        atoms = []
        for key,value in kwargs.items():
            atoms += [i for i in self.atoms if i[key] in value]
        return atoms

    def get_atom(self,sn,reset_dict=False):
        """get a list of atoms by sn"""
        if reset_dict:
            self._sn2atom = {i['sn']:i for i in self.atoms}
        try:
            atoms = [self._sn2atom[i] for i in sn]
        except KeyError:
            self._sn2atom = {i['sn']:i for i in self.atoms}
            atoms = [self._sn2atom[i] for i in sn]
        return atoms

    def save(self, name=None, format=None, frame='all', frame_idx=None, pad0=None, **options):
        StructureWriter().write_st(self, name=name, format=format, frame=frame, frame_idx=frame_idx, pad0=pad0, **options)


    def __str__(self):
        if self.formula == "":
            formula = "empty structure"
        else:
            formula = self.formula
        if self._is_periodic:
            cell_info = " in a={:.1f},b={:.1f},c={:.1f},alpha={:.1f},beta={:.1f},gamma={:.1f}".format(*self.cell_param)
        else:
            cell_info = ' in vacuum'
        frame_info = "; frames: {:d}/{:d}".format(self.frame_idx+1,len(self.frames))
        return formula + cell_info + frame_info


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
        weights = np.array([self.param.elem2mass[x] for x in self.elem])
        center_of_mass = np.average(np.array(self.coord), axis=0, weights=weights)
        return center_of_mass

    @property
    def inertia_tensor(self):
        coord = np.array(self.coord) - self.mass_center
        Ixx,Iyy,Izz,Ixy,Ixz,Iyz = [0,0,0,0,0,0]
        for i,c in enumerate(coord):
            m = self.param.elem2mass[self.elem[i]]
            x,y,z = c
            Ixx += m * (y**2 + z**2)
            Iyy += m * (x**2 + z**2)
            Izz += m * (x**2 + y**2)
            Ixy += -1 * m * x * y
            Ixz += -1 * m * x * z
            Iyz += -1 * m * y * z
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    @property
    def principal_inertia_axis(self) -> np.ndarray:
        """retrun  array of principle inertia axis
        ranged from large eigen_val (moment of inertia) to small eigen_val"""
        eigval, eigvec = np.linalg.eig(self.inertia_tensor)
        sort = np.argsort(eigval)[::-1]
        eigvec = eigvec[:,sort].T
        # right hand convention
        if np.dot(np.cross(eigvec[0], eigvec[1]), eigvec[2]) < 0:
            eigvec *= -1
        return eigvec

    @property
    def principal_geom_axis(self):
        """return array of geometry axis from svd on coords
        ranged from long axis to short axis"""
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
        sc = sorted(c, key=lambda x: self.param.elem2an[x[0]], reverse=True)
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
        if reset_sn:
            s.reset_sn()
        return s
