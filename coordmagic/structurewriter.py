from collections import defaultdict
import sys
import os
import numpy as np
from . import structurereader as sr

__all__ = [
    'write_structure',
    'StructureWriter'
]

def write_structure(st, name='', format='', frame='all', frame_idx=None, pad0=None, **options):
    sw = StructureWriter()
    sw.write_st(st, name=name, format=format, frame=frame, frame_idx=frame_idx, pad0=pad0, **options)

class StructureWriter:
    def __init__(self):
        self.write_func = {'res': self._write_res,
                           'pdb': self._write_pdb,
                           'mol2': self._write_mol2,
                           'gjf': self._write_gjf,
                           'gro': self._write_gro,
                           'cif': self._write_cif,
                           'xyz': self._write_xyz
                           }
        self.multiframe_format=['xyz','pdb','mol2','gro']
        self.options=defaultdict(str)

    def write_st(self, st, name='', format='', frame='all', frame_idx=None, pad0=None, **options):
        """
        st is a structure or graph object
        name is the name of output file, if not defined will infer from the st.basename
        format is the output file type, if not defined will infer form the name
        if both name and format is specified the extension in name will not be considered as format
        frame could be all/each/last/first which will all set frame_idx if frame_idx is not set
        if frame is all but output format do not support traj, will auto switch to last and issue a warning

        """

        if type(st).__name__ == 'Structure':
            self.st = st
        elif type(st).__name__ == 'Graph':
            self.st = sr.conver_structure(st,"graph")
        else:
            print("Error! You should either specify a structure object or a graph object in write_structure()")
            sys.exit()
        # determine which frame to save
        self.frame=frame
        if frame_idx is None:
            if frame == 'all' or frame == 'each':
                self.frame_idx = list(range(len(st.frames)))
            elif frame == 'last':
                self.frame_idx = [len(st.frames) - 1]
            elif frame == 'first':
                self.frame_idx = [0]
            else:
                self.frame_idx = [len(st.frames) - 1]
        if not name:
            name = st.basename
        # determine output filename
        ext=format
        if not ext:
            name,ext = os.path.splitext(name)
            if ext == '':
                print("Error!!! output file format not set, exit")
            else:
                ext = ext[1:]
        if len(self.frame_idx) == 1:
            self.basename = [name]
        elif ext in self.multiframe_format and frame == 'all':
            self.basename = [name]
        elif ext not in self.multiframe_format and frame == 'all':
            print("Warning!!! output format {:s} do not support multiframe format. Only last frame will output".format(ext))
            self.basename = [name]
            self.frame_idx=[len(st.frames) - 1]
            self.frame = "last"
        elif frame == 'each':
            # multiframe
            if pad0 is None:
                pad0 = str(len(str(max(self.frame_idx))))
            elif str(pad0).isdecimal():
                pad0 = str(pad0)
            elif pad0.startswith('+') and pad0[1:].isdecimal():
                pad0 = str(len(str(max(self.frame_idx)))+int(pad0[1:]))
            else:
                print('Warning!!! Wrong format of pad0 parameter, will resort to default value')
                pad0 = str(len(str(max(self.frame_idx))))
            self.basename = []
            for i in self.frame_idx:
                self.basename.append(('{:s}_{:0'+pad0+'d}').format(name,i))
        self.filename = [name + '.' + ext for name in self.basename]
        self.options = defaultdict(str,options)
        if ext in self.write_func:
            if len(self.filename) == 1:
                print("Writing: {:s} using frame: {:d}-{:d}/{:d}"
                      .format(self.filename[0],self.frame_idx[0]+1,self.frame_idx[-1]+1,len(self.st.frames)))
            else:
                print("Writing: {:s} to {:s} using frame: {:d}-{:d}/{:d}"
                      .format(self.filename[0],self.filename[-1],self.frame_idx[0]+1,self.frame_idx[-1]+1,len(self.st.frames)))
            self.write_func[ext]()
        else:
            print('Format {:s} is not supported yet'.format(ext))
            sys.exit()

    def fd(self, input, f=float, d=0.0):
        """fill default
        f is format
        d is defatul value
        """
        if input == '':
            input = f(d)
        return f(input)

    def get_value(self, prop_name, fill=None):
        try:
            prop_name = getattr(self.st, prop_name)
        except AttributeError:
            if fill is None:
                prop_name = fill
            elif len(fill) == len(self.st.cart_coord):
                prop_name = fill
            elif len(fill) < len(self.st.cart_coord) or isinstance(fill, str):
                prop_name = len(self.st.cart_coord) * [fill]
        return prop_name

    def _write_res(self):
        self.file.write('TITL {:s}\n'.format(self.basename))
        self.file.write(('CELL'+7*'{:>11.6f}'+'\n').format(0.0000, *self.st.cell_param))
        self.file.write('LATT -1\n')
        elem_set = list(set(self.st.elem))
        elem_str = ' '.join(elem_set)
        self.file.write('SFAC '+elem_str+'\n')
        for i, c in enumerate(self.st.fcoord):
            elem_name = self.st.elem[i]
            atom_code = elem_set.index(self.st.elem[i])+1
            clist = [elem_name, atom_code] + list(c) + [1.0, 0.0]
            s = '{:<6s}{:<3d}{:<14.8f}{:<14.8f}{:<14.8f}{:<11.5f}{:<10.5f}\n'.format(*clist)
            self.file.write(s)
        self.file.write('END')

    def _write_pdb(self):
        outf = open(self.filename[0], "w")
        outf.write('REMARK Generated by coordmagic \n')
        for n,idx in enumerate(self.frame_idx):
            outf.write('{:6s}{:4s}{:4d}\n'.format('MODEL',"",n+1))
            st=self.st.frames[idx]
            if st._is_periodic == 1:
                outf.write('CRYST1{:>9.3f}{:>9.3f}{:>9.3f}{:>7.2f}{:>7.2f}{:>7.2f} {:<11s}\n'
                                .format(*(self.st.cell_param+['P1'])))
                scale = np.matrix(self.st.cell_vect).I.T.tolist()
                for i in range(1, 4):
                    outf.write('SCALE{:<4d}{:>10.6f}{:>10.6f}{:>10.6f}{:5s}{:>10.5f}\n'
                                .format(*([i]+scale[i-1]+[' ']+[0.0])))
            if not all(st.atomname):
                st.gen_atomname()
            for i, a in enumerate(st.atoms):
                str1 = '{:<6s}{:>5d} {:<4s} {:3s} {:1s}{:>4d}    '\
                       .format('ATOM', a['sn'], a['atomname'], a['resname'],
                               a['chainid'], self.fd(a['resid'], f=int, d=1))
                str2 = '{:>8.3f}{:>8.3f}{:>8.3f}'.format(*a['coord'])
                str3 = '{:>6.2f}{:>6.2f}{:10s}'\
                       .format(self.fd(a['occupancy'], d=1), self.fd(a['bfactor']), ' ')
                str4 = '{:>2s}{:2s}\n'.format(a['elem'], a['formal_charge'])
                outf.write(str1+str2+str3+str4)
            outf.write('{:6s}\n'.format('ENDMDL'))
            outf.close()
            if  len(self.filename) == len(self.frame_idx) and n < len(self.filename)-1:
                outf = open(self.filename[n+1],'w')
                outf.write('REMARK Generated by coordmagic \n')
            else:
                outf = open(self.filename[0],'a')
        outf.write('END\n')


    def _write_gro(self):
        atomname = self.st.getter('atomname')
        if not all(atomname):
            self.st.setter('atomname', self.st.elem)
        self.file.write('gro file generate by masagna, t= 0.0\n')
        self.file.write('{:d}\n'.format(len(self.st.atoms)))
        for i, a in enumerate(self.st.atoms):
            name_id = "{:5d}{:<5s}{:>5s}{:5d}"\
                      .format(self.fd(a['resid'], f=int, d=1),
                              self.fd(a['resname'], f=str, d='MOL'),
                              a['atomname'], a['sn'])
            coord = "{:8.3f}{:8.3f}{:8.3f}"\
                    .format(*[i/10 for i in self.st.coord[i]])
            if a['velocity'] == '':
                vel = '\n'
            else:
                vel = "{:8.4f}{:8.4f}{:8.4f}\n"\
                    .format(*a['velocity'])
            self.file.write(name_id+coord+vel)
        if len(self.st.cell_vect) == 3:
            v1, v2, v3 = self.st.cell_vect
            vlist = [v1[0], v2[1], v3[2], v1[1], v1[2], v2[0], v2[2], v3[0], v3[1]]
            vstr = ' '.join(['{:.5f}'.format(i/10) for i in vlist]) + '\n'
            self.file.write(vstr)

    def _write_mol2(self):
        atomname = self.st.getter('atomname')
        if not all(atomname):
            self.st.setter('atomname', self.st.elem)
        atomtype = self.st.getter('atomtype')
        if not all(atomtype):
            self.st.setter('atomtype', self.st.elem)
        self.file.write('@<TRIPOS>MOLECULE\n')
        self.file.write('{:s}\n'.format(self.st.basename))
        self.file.write('{:d} 0\n'.format(len(self.st.coord)))
        self.file.write('SMALL\n')
        self.file.write('NO_CHARGE\n')
        self.file.write('@<TRIPOS>ATOM\n')
        for i, a in enumerate(self.st.atoms):
            str1 = '{:<6d}{:<6s}'.format(a['sn'], a['atomname'])
            str2 = '{:<12.5f}{:<12.5f}{:<12.5f}'.format(*a['coord'])
            str3 = '{:<6s}'.format(a['atomtype'])
            if a['resid'] != '':
                str4 = '{:<6d}'.format(a['resid'])
                if a['resname'] != '':
                    str4 = str4 + '{:<6s}'.format(a['resname'])
                    if a['charge'] != '':
                        str4 = str4 + '{:<.6f}'.format(a['charge'])
            else:
                str4 = ''
            line = str1+str2+str3+str4+'\n'
            self.file.write(line)
        if self.options['connection']:
            if len(self.st.molecules) == 0:
                self.st.graph.gen_mol()
                self.st.graph.gen_internal_coords()
                self.st.graph.gen_bond_order()
            self.file.write('@<TRIPOS>BOND\n')
            for  i,b in enumerate(self.st.bonds.items()):
                bondtype = b[1]['bond_order']
                target_bo = np.array([1,2,3])
                bo = target_bo[np.argmin(np.abs(target_bo - bondtype))]
                self.file.write('{:d} {:d} {:d} {:d}\n'.format(i+1,b[0][0],b[0][1],bo))



    def _write_gjf(self):
        outf = open(self.filename[0], "w")
        def_param = {'nproc':'8','charge':'0','spin':'1','mem':'4GB','extra':'','cpu':'',
                     'oldchk':'','chk':'{:s}.chk'.format(self.basename[0]),'link':[],
                     'keywords':'pbe1pbe1 def2svp em(gd3bj)'}
        def_param.update({k:v for k,v in self.options.items() if k in def_param})
        outf.write('%chk={:s}\n'.format(def_param['chk']))
        if def_param['oldchk']:
            if def_param['oldchk'] != def_param['chk']:
                outf.write('%oldchk={:s}\n'.format(def_param['oldchk']))
        if def_param['cpu']:
            outf.write('%cpu={:s}\n'.format(def_param['cpu']))
        else:
            outf.write('%nprocshared={:s}\n'.format(def_param['nproc']))
        outf.write('%mem={:s}\n'.format(def_param['mem']))
        outf.write('#p {:s}\n'.format(def_param['keywords']))
        outf.write('\n')
        outf.write('{:s} generated by CoordMagic\n'.format(self.basename[0]))
        outf.write('\n')
        outf.write('{:s} {:s}\n'.format(def_param['charge'],def_param['spin']))
        for i, a in enumerate(self.st.atoms):
            str1 = '{:<6s}'.format(a['elem'])
            str2 = '{:<12.5f}{:<12.5f}{:<12.5f}'.format(*a['coord'])
            line = str1+str2+'\n'
            outf.write(line)
        if def_param['extra']:
            outf.write('\n')
            outf.write(def_param['extra'])
        if len(def_param['link']) > 0:
            for link in def_param['link']:
                if 'keywords' in link:
                    outf.write('\n')
                    outf.write('--Link1--\n')
                    outf.write('%nprocshared={:s}\n'.format(def_param['nproc']))
                    outf.write('%mem={:s}\n'.format(def_param['mem']))
                    if 'chk' in link:
                        if link['chk'] != def_param['chk']:
                            outf.write('%chk={:s}\n'.format(link['chk']))
                            outf.write('%oldchk={:s}\n'.format(def_param['chk']))
                    else:
                        outf.write('%chk={:s}\n'.format(def_param['chk']))
                    outf.write('#p {:s}\n'.format(link['keywords']))
                    if 'charge' in link and 'spin' in link:
                        outf.write('\n')
                        outf.write('{:s} generated by CoordMagic\n'.format(self.basename))
                        outf.write('\n')
                        outf.write('{:s} {:s}\n'.format(link['charge'], link['spin']))
        outf.write('\n')
        outf.write('\n')
        outf.write('\n')

    def _write_xyz(self):
        outf = open(self.filename[0], "w")
        for n,idx in enumerate(self.frame_idx):
            struct = self.st.frames[idx]
            outf.write('{:d}\n'.format(len(struct.atoms)))
            outf.write('{:s}\n'.format(struct.prop['comment']))
            for i, a in enumerate(struct.atoms):
                str1 = '{:<6s}'.format(a['elem'])
                str2 = '{:>12.6f}{:>12.6f}{:>12.6f}'.format(*a['coord'])
                line = str1 + str2 + '\n'
                outf.write(line)
            outf.close()
            if  len(self.filename) == len(self.frame_idx) and n < len(self.filename)-1:
                outf = open(self.filename[n+1],'w')
            else:
                outf = open(self.filename[0],'a')
        outf.close()

    def _write_cif(self):

        outf = open(self.filename[0], "w")
        outf.write('data_'+self.st.basename+'\n')
        for n,idx in enumerate(self.frame_idx):
            st=self.st.frames[idx]
            outf.write('_symmetry_space_group_name_H-M    \'P1\'\n')
            outf.write('_symmetry_Int_Tables_number       1\n')
            outf.write('_symmetry_equiv_pos_as_xyz\n')
            outf.write('  x,y,z\n')
            outf.write('{:35s}{:.6f}\n'.format('_cell_length_a', st.cell_param[0]))
            outf.write('{:35s}{:.6f}\n'.format('_cell_length_b', st.cell_param[1]))
            outf.write('{:35s}{:.6f}\n'.format('_cell_length_c', st.cell_param[2]))
            outf.write('{:35s}{:.6f}\n'.format('_cell_angle_alpha', st.cell_param[3]))
            outf.write('{:35s}{:.6f}\n'.format('_cell_angle_beta', st.cell_param[4]))
            outf.write('{:35s}{:.6f}\n'.format('_cell_angle_gamma', st.cell_param[5]))
            outf.write('loop_\n'
                          '_atom_site_label\n'
                          '_atom_site_type_symbol\n'
                          '_atom_site_fract_x\n'
                          '_atom_site_fract_y\n'
                          '_atom_site_fract_z\n')
            if not all(st.atomname) or st.atomname[0] == st.elem[0]:
                st.gen_atomname()
            for i, a in enumerate(st.atoms):
                str1 = '{:7s}{:7s}'.format(a['atomname'], a['elem'])
                str2 = '{:14.8f}{:14.8f}{:14.8f}'.format(*a['fcoord'])
                outf.write(str1+str2+'\n')
            outf.close()
            if  n < len(self.filename)-1:
                outf = open(self.filename[n+1],'w')
