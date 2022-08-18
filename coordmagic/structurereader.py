import os
import sys
import re
from collections import defaultdict
import numpy as np
from . import structure
import itertools


def read_structure(inputfile,filetype=''):
    sr = StructureReader()
    sr.st = structure.Structure()
    try:
        sr.file = open(inputfile, 'r')
        filename = os.path.basename(inputfile)
        sr.basename, ext = os.path.splitext(filename)
        sr.st.basename = sr.basename
    except TypeError:
        sys.exit('coordmagic: Error! The input {:s} is not a file'
                 .format(str(inputfile)))
    if not ext:
        if sr.basename == 'CONTCAR' or sr.basename == 'POSCAR':
            ext = '.vasp'
        else:
            print('coordmagic: Error! File {:s} do not have extension, exit now'.format(filename))
            sys.exit()
    if filetype:
        ext='.' + filetype
    if ext not in sr.read_func:
        print('coordmagic: Error! file format {:s} not supported'.format(ext))
        sys.exit()
    sr.read_func[ext]()
    if len(sr.st.atoms) == 0:
        print("No structure read from {:s}".format(inputfile))
        return False
    sr.st.complete_self(wrap=False)
    return sr.st

def conver_structure(struct_obj,obj_type,**kwargs):
    sr = StructureReader()
    sr.st = structure.Structure()
    sr.struct_obj = struct_obj
    sr.read_func[obj_type](**kwargs)
    sr.st.complete_self()
    return sr.st


class StructureReader:

    B2A = 0.529177249

    def __init__(self):
        self.read_func = {'.res': self._read_res,
                          '.cif': self._read_cif,
                          '.xsf': self._read_xsf,
                          '.STRUCT': self._read_STRUC,
                          '.vasp': self._read_vasp,
                          '.pdb': self._read_pdb,
                          '.mol2': self._read_mol2,
                          '.gro': self._read_gro,
                          '.log': self._read_gau_log,
                          '.fchk': self._read_gau_fchk,
                          '.gjf':self._read_gau_inp,
                          '.xyz':self._read_xyz,
                          'parmed': self._conver_parmed,
                          'mdanalysis': self._conver_mdanalysis,
                          'graph':self._conver_graph,
                          }


    def _read_res(self):
        for l in self.file:
            if 'CELL' not in l:
                pass
            else:
                param = [float(i) for i in l.split()[2:]]
            if 'SFAC' in l:
                ele = l.split()[1:]
            if re.search(r'\s+\d+\s+-?\d+.\d+\s+-?\d+.\d+\s+-?\d+.\d+\s+', l):
                atom = defaultdict(str)
                coord = [float(i) for i in (l.split()[2:5])]
                atom['elem'] = ele[int(l.split()[1])-1]
                atom['fcoord'] = coord
                self.st.atoms.append(atom)
        self.st.cell_param = param

    def _read_pdb(self):
        param=''
        for l in self.file:
            atom = defaultdict(str)
            if 'CRYST1' in l:
                param = [float(i) for i in l.split()[1:7]]
            if l.startswith('ATOM') or l.startswith('HETATM'):
                atom['resname'] = l[17:20].strip()
                atom['atomname'] = l[12:16].strip()
                atom['sn'] = int(l[6:11])
                try:
                    atom['resid'] = int(l[22:26].strip())
                except ValueError:
                    pass
                try:
                    atom['chainid'] = int(l[21])
                except ValueError:
                    pass
                try:
                    atom['bfactor'] = float(l[60:66].strip())
                except ValueError:
                    pass
                try:
                    atom['occupancy'] = float(l[54:60].strip())
                except ValueError:
                    pass

                x = float(l[30:38].strip())
                y = float(l[38:46].strip())
                z = float(l[46:54].strip())
                atom['elem'] = l[76:78].strip()
                atom['coord'] = [x, y, z]
                self.st.atoms.append(atom)
        self.st.cell_param = param

    def _read_cif(self):
        coord_flag = 0
        index = 0
        name_idx, ele_idx = [None, None]
        fx_idx, fy_idx, fz_idx = [None, None, None]
        # read coordinate and cell parameters from cif file
        for line in self.file:
            line = line.replace(')', '').replace('(', '')
            if re.search('_cell_length_a', line):
                a = float(line.split()[1])
            if re.search('_cell_length_b', line):
                b = float(line.split()[1])
            if re.search('_cell_length_c', line):
                c = float(line.split()[1])
            if re.search('_cell_angle_alpha', line):
                alpha = float(line.split()[1])
            if re.search('_cell_angle_beta', line):
                beta = float(line.split()[1])
            if re.search('_cell_angle_gamma', line):
                gamma = float(line.split()[1])
            if re.search('_symmetry_Int_Tables_number', line):
                self.st.sym_group = float(line.split()[1])
                if float(self.st.sym_group) != 1:
                    print('the symGroup in cif file is not 1, exit now')
                    sys.exit()
            if 'loop_' in line:
                coord_flag = 1
            if '_atom_site_' in line and coord_flag == 1:
                if 'label' in line:
                    name_idx = index
                if 'type_symbol' in line:
                    ele_idx = index
                if 'fract_x' in line:
                    fx_idx = index
                if 'fract_y' in line:
                    fy_idx = index
                if 'fract_z' in line:
                    fz_idx = index
                index = index + 1

            if re.search(r'-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+.\d+\s+', line) and coord_flag == 1:
                atom = defaultdict(str)
                fx = float(line.split()[fx_idx])
                fy = float(line.split()[fy_idx])
                fz = float(line.split()[fz_idx])
                coord = [fx, fy, fz]
                if ele_idx is not None:
                    atom['elem'] = line.split()[ele_idx]
                if name_idx is not None:
                    atom['atomname'] = line.split()[name_idx]
                atom['fcoord'] = coord
                self.st.atoms.append(atom)
        self.st.cell_param = [a, b, c, alpha, beta, gamma]

    def _read_xsf(self):
        read_vect = 0
        read_coord = 0
        for line in self.file:
            if 'PRIMVEC' in line:
                read_vect = 1
                continue
            if read_vect == 1 and len(line.split()) == 3:
                self.st.cell_vect.append([float(i) for i in line.split()])
                if len(self.st.cell_vect) == 3:
                    read_vect = 0
            if 'PRIMCOORD' in line:
                read_vect = 0
                read_coord = 1
                continue
            if read_coord == 1 and len(line.split()) == 4:
                atom = defaultdict(str)
                atom['atomnum'] = int(line.split()[0])
                atom['coord'] = [float(i) for i in line.split()[1:]]
                self.st.atoms.append(atom)
            if len(self.st.atoms) > 0 and len(line.split()) != 4:
                read_coord = 0

    def _read_gro(self):
        cell_vect = [0]
        #first two linea are omitted
        for l in itertools.islice(self.file, 2, None):
            if re.search('[a-zA-Z]',l):
                atom = defaultdict(str)
                atom['resid'] = int(l[:5].strip())
                atom['resname'] = l[5:10].strip()
                atom['atomname'] = l[10:15].strip()
                atom['sn'] = int(l[15:20].strip())
                x = float(l[20:28])*10
                y = float(l[28:36])*10
                z = float(l[36:44])*10
                atom['coord'] = [x, y, z]
                try:
                    vx = float(l[44:52])  # unit is nm/ps
                    vy = float(l[52:60])
                    vz = float(l[60:68])
                except ValueError:
                    vx = 0
                    vy = 0
                    vz = 0
                atom['velocity'] = [vx, vy, vz]
                self.st.atoms.append(atom)
            else:
                if len(l.split()) == 3:
                    x1, y2, z3 = [float(i)*10 for i in l.split()]
                    y1, z1, x2, z2, x3, y3 = [0, 0, 0, 0, 0, 0]
                elif len(l.split()) == 9:
                    x1, y2, z3, y1, z1, x2, z2, x3, y3 = \
                        [float(i)*10 for i in l.split()]
                else:
                    x1, y2, z3, y1, z1, x2, z2, x3, y3 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                cell_vect = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
        if np.sum(np.abs(cell_vect)) > 0:
            self.st.cell_vect = cell_vect
        

    def _read_STRUC(self):
        readCell = 0
        readCoord = 0
        # bohr convert to angstrom
        self.dimension = 3
        for line in self.file:
            if '$cell vectors' in line:
                readCell = 1
                continue
            if '$coordinates' in line:
                readCoord = 1
                readCell = 0
                continue
            if '$END' in line:
                readCoord = 0
                readCell = 0
                continue
            if readCell == 1:
                va = [float(i)*self.B2A for i in line.split()]
                readCell += 1
                continue
            if readCell == 2:
                vb = [float(i)*self.B2A for i in line.split()]
                if max(map(float, vb)) > 499:
                    self.dimension = self.dimension-1
                readCell += 1
                continue
            if readCell == 3:
                vc = [float(i)*self.B2A for i in line.split()]
                if max(map(float, vc)) > 499:
                    self.dimension = self.dimension-1
            if readCoord == 1:
                atom = int(line.split()[4])
                coord = np.array(float(line.split()[1:4]))*self.B2A
                self.st.cart_coord.append(coord)
                self.st.atoms.append(self.st._an2elem[atom])
        self.st.cell_vector = [va, vb, vc]

    def _read_mol2(self):
        read_flag = 0
        for l in self.file:
            if '@<TRIPOS>ATOM' in l:
                read_flag = 1
                continue
            if read_flag == 1 and '@<TRIPOS>' in l:
                read_flag = 0
            if read_flag == 1 and len(l.split()) >= 6:
                atom = defaultdict(str)
                sn, atomname, x, y, z, atom_type = l.split()[:6]
                atom['sn'] = int(sn)
                atom['atomname'] = atomname
                atom['coord'] = [float(i) for i in [x, y, z]]
                atom['atomtype'] = atom_type
                if len(l.split()) >= 7:
                    atom['resid'] = int(l.split()[6])
                if len(l.split()) >= 8:
                    atom['resname'] = l.split()[7]
                if len(l.split()) >= 9:
                    atom['charge'] = float(l.split()[8])
                self.st.atoms.append(atom)

    def _read_gau_log(self):
        read_coord = ""
        norm_flag = 0
        atoms = []
        energies = []
        ex_energies = []
        std_orient_atoms = []
        so_atoms = []
        input_orient_atoms = []
        io_atoms = []

        for l in self.file:
            if l.startswith(' Charge = '):
                charge = l.strip().split()[2]
                spin = l.strip().split()[5]
                charge_spin = [charge, spin]
            if 'Standard orientation' in l:
                if len(so_atoms) > 0:
                    std_orient_atoms.append(so_atoms)
                so_atoms = []
                read_coord = "so_1"
            if 'Input orientation' in l:
                if len(io_atoms) > 0:
                    input_orient_atoms.append(io_atoms)
                io_atoms = []
                read_coord = "io_1"
            if read_coord.endswith("1") and '----------' in l:
                read_coord += "2"
                continue
            if read_coord.endswith("2") and '----------' in l:
                read_coord += "3"
                continue
            if read_coord.endswith("3") and '----------' in l:
                read_coord += "4"
                continue
            if read_coord.endswith("3"):
                atom = defaultdict(str)
                atom['atomnum'] = l.split()[1]
                atom['coord'] = [float(i) for  i in l.split()[3:]]
                if read_coord.startswith('i'):
                    io_atoms.append(atom)
                if read_coord.startswith('s'):
                    so_atoms.append(atom)
            if 'SCF Done' in l:
                energy = float(l.split()[4])
                energies.append(energy)
            if 'Total Energy' in l:
                ex_energy = float(l.split()[4])
                ex_energies.append(ex_energy)
            if 'Normal termination' in l:
                norm_flag = 1
        if len(io_atoms)  > 0:
            input_orient_atoms.append(io_atoms)
        if len(so_atoms)  > 0:
            std_orient_atoms.append(so_atoms)
        if len(std_orient_atoms) > 0:
            all_atoms = std_orient_atoms
        elif len(input_orient_atoms) > 0:
            all_atoms = input_orient_atoms
        else:
            all_atoms = []
        if len(all_atoms) >= 1:
            for i in range(len(all_atoms)-1):
                self.st.atoms = atoms
                self.st.complete_self()
                self.st.choose_frame(self.st.frame_sn+1)
            self.st.atoms = all_atoms[-1]
        else:
            self.st.atoms = []
        self.st.energies = energies
        self.st.ex_energies = ex_energies

    def _read_gau_inp(self):
        cs_read = 0
        atoms=[]
        for l in self.file:
            if re.match('^\s*-?\d\s+\d\s?',l) and cs_read == 0:
                cs_read = 1
                continue
            if cs_read == 1 and re.match('^\s*$',l):
                cs_read = 2
                continue
            if cs_read == 1:
                atom = defaultdict(str)
                if len(l.split()) == 4:
                    astr = l.split()[0]
                    atom['coord'] = [float(i) for  i in l.split()[1:]]
                elif len(l.split()) == 5:
                    astr = l.split()[0]
                    atom['coord'] = [float(i) for  i in l.split()[2:]]
                else:
                    print('Error! only support gjf file with cartesian coordinates')
                    break
                at = astr.split('(')[0]
                if at.isdigit():
                    atom['atomnum'] = at
                elif at.isalpha():
                    atom['elem'] = at
                else:
                    print('Error! atom symbol illegal: {:s}'.format(astr))
                    break
                atoms.append(atom)
        self.st.atoms = atoms

    def _read_gau_fchk(self):
        atom_numbers = []
        read_numbers = 0
        list_coords = []
        read_coords = 0
        #Get atomic number from fchk
        for line in self.file:
            if 'Atomic numbers' in line:
                read_numbers = 1
                continue
            if read_numbers == 1:
                if line.strip()[:1].isalpha():
                    read_numbers = 0
                else:
                    atom_numbers = atom_numbers + line.split()
            #Gets coordinates
            if 'Current cartesian coordinates' in line:
                read_coords = 1
                continue
            if read_coords == 1:
                if line.strip()[:1].isalpha():
                    read_coords = 0
                else:
                    list_coords = list_coords + [float(i) for i in line.split()]
        list_coords = np.array(list_coords) * 0.52917721067121
        coords = np.reshape(list_coords, (-1, 3))
        atoms = []
        for i,n in enumerate(atom_numbers):
            atom = defaultdict(str)
            atom['atomnum'] = n
            atom['coord'] = coords[i]
            atoms.append(atom)
        self.st.atoms = atoms



    def _read_vasp(self):
        lines = self.file.readlines()
        scaling_factor = float(lines[1].strip())
        va = np.array([float(i) for i in lines[2].split()])*scaling_factor
        vb = np.array([float(i) for i in lines[3].split()])*scaling_factor
        vc = np.array([float(i) for i in lines[4].split()])*scaling_factor
        self.st.cell_vect = [list(va), list(vb), list(vc)]
        element = lines[5].strip().split()
        num_of_element = lines[6].strip().split()
        elem = []
        for i in range(len(element)):
            elem += [element[i]] * int(num_of_element[i])
        for i in elem:
            atom = defaultdict(str)
            atom['elem'] = i
            self.st.atoms.append(atom)
        if lines[7].strip().lower().startswith('c'):
            coord = []
            for n in lines[7:]:
                if re.search('-?\d+.\d+\s+-?\d+.\d+\s+-?\d+.\d+', n):
                    c = np.array([float(i) for i in n.strip().split()])*scaling_factor
                    coord.append(list(c))
            self.st.setter('coord', coord)
        else:
            fcoord = []
            for n in lines[7:]:
                if re.search('-?\d+.\d+\s+-?\d+.\d+\s+-?\d+.\d+', n):
                    fcoord.append([float(i) for i in  n.strip().split()[:3]])
            self.st.setter('fcoord', fcoord)

    def _read_xyz(self):
        read_coord = 0
        atoms = []
        comments = []
        for l in self.file:
            x = l.split()
            if read_coord != 1 and len(x) == 1:
                read_coord = 1
                if len(atoms) == int(l):
                    self.st.atoms = atoms
                    self.st.complete_self()
                    self.st.choose_frame(self.st.frame_sn+1)
                    atoms = []
                continue
            if read_coord == 1:
                comments.append(l)
                read_coord = 2
                continue
            if read_coord == 2:
                atom = defaultdict(str)
                atom['elem'] = x[0]
                atom['coord'] = [float(i) for i in x[1:]]
                atoms.append(atom)
        self.st.atoms = atoms
        self.st.complete_self()
        self.st.comments = comments

    def _conver_parmed(self):
        atoms = []
        for a in self.struct_obj.atoms:
            atom = defaultdict(str)
            atom['atomname'] = a.name
            atom['atomnum'] = a.atomic_number
            atom['sn'] = a.idx + 1
            atom['coord'] = [a.xx, a.xy, a.xz]
            atom['resname'] = a.residue.name
            atoms.append(atom)
        self.st.atoms = atoms
        # print(self.st.atoms)
        if self.struct_obj.box is not None:
            self.st.cell_param = self.struct_obj.box
        self.st.complete_self()


    def _conver_mdanalysis(self):
        '''conver mdanalysis univeral object to structure object'''
        for ts in self.struct_obj.trajectory:
            frame_sn = ts.frame + 1
            self.st.choose_frame(frame_sn)
            cell_param = ts.dimensions
            atoms = []
            for a in self.struc_obj.atoms:
                atom = defaultdict(str)
                atom['atomname'] = a.name
                atom['sn'] = a.ix + 1
                atom['coord'] = a.position
                atom['resname'] = a.resname
                atom['charge'] = a.charge
                atoms.append(atom)
            self.st.atoms = atoms
            self.st.cell_param = cell_param


    def _conver_graph(self, cell_param='',cell_vect=''):
        '''generate structure object from a graph
        you may not need this function if the graph is a subgraph
        of the self.S.graph. In this case you could use
        st=self.S.extract_struc(subgraph.nodes())
        to generate a structure
        The pbc option is auto indicate that if the self.S
        object has pbc, the newly generated structure will have same pbc param
        '''
        if cell_param:
            self.st.cell_param = cell_param
        elif cell_vect:
            self.st.cell_vect = cell_vect
        else:
            self.st.period_flag = 0
        for sn in sorted(list(self.struct_obj.nodes())):
            self.st.atoms.append(defaultdict(str,self.struct_obj.nodes[sn]))
            self.st.complete_self(wrap=False)








