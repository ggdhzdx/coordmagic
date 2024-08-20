'''todo list
dihedral from seminario method is too large
proper dihedral from hessian or just bond length
param option in dihedral to set empity k
improper param from hessian
'''
import sys
import os
import parmed as pmd
import timeit
import numpy as np
import pandas as pd
from collections import defaultdict,Counter
from networkx.algorithms import isomorphism
from .paramgen import ParamGen
sys.path.append('..')
import coordmagic as cm


class ParamRefine:
    '''update parameter for ParmED structure object from various sources '''
    def __init__(self,top_file = '',pmd_struct='', saveinfo=''):
        if top_file:
            self.load_file(top_file)
        if pmd_struct != '':
            self.convert2struct(pmd_struct)
        self.saveinfo = saveinfo

    def load_file(self,top_file):
        filename = os.path.basename(top_file)
        self.basename,ext = os.path.splitext(filename)
        if ext in ['.log','.mol2','.pdb']:
            pg = ParamGen(filename)
            top,crd,chg = pg.gen_param()
            pmd_struct = pmd.load_file(top)
            pmd_coord = pmd.load_file(crd)
            pmd_struct.coordinates = pmd_coord.coordinates
        else:
            pmd_struct = pmd.load_file(top_file)
        self.convert2struct(pmd_struct)
        # print(self.ps.rb_torsions[0].ignore_end)
        # self.ps = pmd_struct
        # self.ps = pmd.gromacs.GromacsTopologyFile.from_structure(pmd_struct)
        # self.complete_adjust()
        self.top_bug_fix()


    def top_bug_fix(self):
        '''fix a bug that chlorine was treated as carbon by antechamber'''
        for a in self.ps.atoms:
            if a.atom_type == 'cl':
                a.element = 17
                a.atomic_num = 17
    
    # def complete_adjust(self):
    #     """complete missing adjust
    #     adjust are usually 1-4 nonbonded interaction
    #     """
    #     #print(len(self.ps.adjusts))
    #     # print(self.ps.adjust_types)
    #     #print(len([d for d in self.ps.dihedrals if not d.improper]))
    #     # print(len([a for a in self.ps.adjusts if a.type.chgscale !=0 ]))
    #     exist_pair = [frozenset([a.atom1,a.atom2]) for a in self.ps.adjusts] 
    #     d_pair = [frozenset([d.atom1,d.atom4]) for d in self.ps.dihedrals]
    #     #print(len(exist_pair))
    #     #print(len(d_pair))
    #     for a in self.ps.adjusts:
    #         pair = frozenset([a.atom1,a.atom2])
    #         # if pair in d_pair:
    #             # print(a)
    #     # for d in self.ps.dihedrals:
    #     #     if not d.improper:
    #     #         pair = frozenset([d.atom1,d.atom4])
    #     #             pass
    #     #             #print("xxxxxxxxxxxxxxxx")
    #     #         else:
    #     #             print("aaaaaa")
    #     #             new_pair=pmd.NonbondedException(d.atom1,d.atom2)
    #     #             print(new_pair)
            

    def convert2struct(self,st):
        '''generate a new parmed structure object to remove 
        source specific feature of the parmed structure object '''
        self.ps = pmd.Structure()
        self.ps.atoms = st.atoms
        self.ps.residues = st.residues
        self.ps.bonds = st.bonds
        self.ps.angles = st.angles
        self.ps.dihedrals = st.dihedrals
        self.ps.rb_torsions = st.rb_torsions
        self.ps.rb_torsion_types = st.rb_torsion_types
        self.ps.bond_types = st.bond_types
        self.ps.angle_types = st.angle_types
        self.ps.dihedral_types = st.dihedral_types
        self.ps.combining_rule = st.combining_rule
        self.ps.adjusts = st.adjusts
        self.ps.adjust_types = st.adjust_types

    def refine_charge(self,charge_file):
        chg_list = []
        cf = open(charge_file,'r')
        for i,c in enumerate(cf):
            atom = self.ps.atoms[i]
            old_chg =atom.charge
            new_chg = float(c.split()[-1])
            chg_list.append([atom.element,old_chg,new_chg])
            atom.charge = new_chg
        df = pd.DataFrame(chg_list,columns=['elem','old_chg','new_chg'])
        df['delta_chg'] = df['new_chg'] - df['old_chg']
        report = self.do_statistics(df,['delta_chg'],groupby=None,transpose=True)
        print(report)

    def set_charge(self,target_charge = 0):
        '''set total charge of the system by distribution the charge evenly over all atoms'''
        charge = [a.charge for a in self.ps.atoms]
        total_q = sum(charge)
        delta_q = target_charge - total_q
        average_delta_q = delta_q / len(self.ps.atoms)
        print("The system have total charge of {:.6f}\n"
              "I will add charge {:.8f} to every atom to make system reach target charge {:.6f}"
              .format(total_q,average_delta_q,target_charge))
        for a in self.ps.atoms:
            a.charge += average_delta_q

    def load_coord(self,coord_file):
        '''load coordinates from file and set to prm_structure.positions
        note that the periodic cell info may not be read in parmed load_file function'''
        filename = os.path.basename(coord_file)
        basename,ext = os.path.splitext(filename)
        if ext in ['.pdb','.mol2','.pdbx','.inpcrd','.rst7','.restrt','.cif','.ncrst']:
            pmd_coord = pmd.load_file(coord_file)
            self.ps.coordinates = pmd_coord.coordinates
        elif ext in ['.res','.xyz','.log','.fchk','.xsf','.gro']:
            cm_st = cm.read_structure(coord_file)
            self.ps.coordinates = cm_st.coord
            if len(cm_st.cell_param) == 6:
                self.ps.box = cm_st.cell_param
                self.ps.box_vectors = cm_st.cell_vect
        else:
            print('format {:s} not suported'.format(ext))
            sys.exit()

    def refine_badi(self,refine_source, b='',a='',d='',i=''):
        # set default config
        self.bond_config = {'action':'replace','ek':'ek'}
        self.angle_config = {'action':'replace','ek':'ek','seminario':'ms'}
        self.dihedral_config = {'action':'add','ek':'e'}
        self.improper_config = {'action':'omit','ek':'ek','threshold':5}
        self.sn2atom = {a.idx+1:a for a in self.ps.atoms}
        # parse parameters to config dict
        def update_param(init_dict,config):
            for i in config:
                if i in ['replace','omit','add']:
                    init_dict['action'] = i
                if i in ['e','k','ek']:
                    init_dict['ek'] = i
                if i in ['ms','origin']:
                    init_dict['seminario'] = i
                if i.isdigit():
                    init_dict['threshold'] = i
        if b:
            update_param(self.bond_config,b.split(','))
        if a:
            update_param(self.angle_config,a.split(','))
        if d:
            update_param(self.dihedral_config,d.split(','))
        if i:
            update_param(self.improper_config,i.split(','))
        if self.bond_config['action'] != 'omit':
            self.__transfer_bond(param_dict = refine_source.bonds,action = self.bond_config['action'], 
                                 param=self.bond_config['ek'])
        if self.angle_config['action'] != 'omit':
            self.__transfer_angle(param_dict = refine_source.angles, action = self.angle_config['action'], 
                                  param=self.angle_config['ek'])
        if self.dihedral_config['action'] != 'omit':
            self.__transfer_dihedral(param_dict = refine_source.dihedrals, action = self.dihedral_config['action'], 
                                     param=self.dihedral_config['ek'])
        # if self.improper_config['action'] != 'omit':
            # improper_source = {k:pmd.ImproperType(v['k'],180) for k,v in refine_source.impropers}
            # self.__transfer_improper(improper_source,action = self.improper_config['action'], param=self.improper_config['ek'])

    def transfer_param_from(self,res_obj,m='',a='',p=''):
        '''m is atom sn for matching
        a is atom sn for transfer
        p is what parameter to transfer'''
        match_sn = []
        atom_sn = []
        # parse m and a
        for i in m.split(','):
            if '-' in i:
                a1 = int(i.split('-')[0])
                a2 = int(i.split('-')[1]) + 1
                match_sn += list(range(a1,a2))
            else:
                match_sn.append(int(i))
        for i in a.split(','):
            if '-' in i:
                a = int(i.split('-')[0])
                b = int(i.split('-')[1]) + 1
                atom_sn += list(range(a,b))
            else:
                atom_sn.append(int(i))
        G2 = res_obj['cm_frag']['graph'].subgraph(match_sn)
        st = cm.conver_structure(self.ps,'parmed')
        print('Generating target graph ... ',end="",flush=True)
        start = timeit.default_timer()
        st.G.gen_mol()
        end = timeit.default_timer()
        print('Graph generation took {:.3f} seconds'.format(end-start))
        # compute graph match time
        nm = isomorphism.categorical_node_match("elem","")
        mapping = []
        mapped_atom = []
        for id,m in st.molecules.items():
            print('Begin graph match of molecule {:d}: {:s} ... '.format(id,m['formula']), end='', flush=True)
            G1 = m['graph']
            GM = isomorphism.GraphMatcher(G1,G2,node_match=nm)
            start = timeit.default_timer()
            map_list = list(GM.subgraph_isomorphisms_iter())
            end = timeit.default_timer()
            print('took {:.3f} seconds'.format(end-start))
            for m in map_list:
                G1_atoms = ','.join(map(str,sorted([i for i in m.keys()])))
                if G1_atoms not in mapped_atom:
                    mapped_atom.append(G1_atoms)
                    mapping.append(m)
            print('Total {:d} match founded'.format(len(mapped_atom)))
        if len(mapped_atom) == 0:
            sys.exit()
        atom_map = []
        for m in mapping:
            atom_map.append({v:k for k,v in m.items() if v in atom_sn})
        self.adj_dihedral(contract='all')
        self.adj_dihedral(contract='all',param_struct=res_obj['pmd_struct'])
        self.sn2atom = {a.idx+1:a for a in self.ps.atoms}
        transfer_func = {'b':self.__transfer_bond,
                         'a':self.__transfer_angle,
                         'd':self.__transfer_dihedral,
                         'i':self.__transfer_improper,
                         'c':self.__transfer_charge,
                         't':self.__transfer_type,
                         'v':self.__transfer_vdw}
        for t in p.split(','):
            if t in ['b','a','d','i']:
                t = t + 'ek'
            transfer_func[t[0]](parmed_struct=res_obj['pmd_struct'],param=t[1:],atom_map = atom_map)
            # transfer bond

    def __map_sn(self,source_sn,atom_map=[]):
        # map to target sn and reorder, also have the filter function
        # use try and except to filter atoms that not in map
        # atom_map is a list of dict, so 5 atom may be mapped to 25 atoms if the length of list is 5
        # return a 2d list 
        if len(atom_map) == 0:
            return [source_sn]
        else:
            sn = []
            for m in atom_map: 
                if any(type(i) is frozenset for i in source_sn):
                    mapped_sn = []
                    try:
                        for ss in source_sn:
                            if type(ss) is frozenset:
                                j = frozenset([m[i] for i in ss])
                            else:
                                j=m[ss]
                            mapped_sn.append(j)
                        sn.append(tuple(mapped_sn))
                    except KeyError:
                        pass
                else:
                    try:
                        sn.append(self.__reorder_sn([m[i] for i in source_sn]))
                    except  KeyError:
                        pass
            return sn

    def __reorder_sn(self,sn):
        # reorder sn, which is a iterable of int
        if sn[0] > sn[-1]:
            return tuple(sn[::-1])
        else:
            return tuple(sn)

    def __check_source_target(self,source_param, target_param, atom_map, improper=False):
        '''source param are dict where keys are tuples of atom sn and values are parmed 
        bond/angle/dihedral/improper (BADI) type
        target_param are parmed bonds/angles/dihedrals/impropers TrackedList
        this function will return:
        target -- a dict where keys are sn and values are BADI types in targe param
        target_lack -- a dict of BADI param that in source but not in target 
        overlap -- a dict of BADI param that both in source and in target
        lack_note -- a string with the form (n X m) where n is number of match and m is params in each match
        overlap_note -- similar to lack_note, but depict the match in overlap part
        '''
        # first sort keys in source and target and re-format keys to tuple of int
        source = {}
        target = {}
        for k,v in source_param.items():
            if not improper:
                k = [int(i) for i in k]
                source[self.__reorder_sn(k)] = v
            else:
                k = [int(i) for i in k]
                source[(k[0],frozenset(k[1:]))] = v
        if len(source) == 0:
            return {},{},{},"",""
        sn_len = len(next(iter(source.keys())))
        if improper:
            ptype = "improper"
            for i in target_param:
                target[(i.atom1.idx+1,frozenset([i.atom2.idx+1,i.atom3.idx+1,i.atom4.idx+1]))] = i
        elif sn_len == 2:
            ptype = "bond"
            for b in target_param:
                target[self.__reorder_sn([b.atom1.idx+1,b.atom2.idx+1])] = b
        elif sn_len == 3:
            ptype = "angle"
            for a in target_param:
                target[self.__reorder_sn([a.atom1.idx+1,a.atom2.idx+1,a.atom3.idx+1])] = a
        elif sn_len == 4 and not improper:
            ptype = "dihedral"
            for d in target_param:
                target[self.__reorder_sn([d.atom1.idx+1,d.atom2.idx+1,d.atom3.idx+1,d.atom4.idx+1])] = d
        else:
            ptype = "unknown"
        # found overlap of source and target
        target_lack = {}
        for k,v in source.items():
            # print(len(self.__map_sn(k,atom_map=atom_map)))
            target_lack.update({sn:v for sn in self.__map_sn(k,atom_map = atom_map) if sn not in target})
        overlap = {}
        for k,v in source.items():
            overlap.update({sn:v for sn in self.__map_sn(k,atom_map = atom_map) if sn in target})
        source_lack = {}
        source_sn = []
        for k,v in target.items():
            source_sn += self.__map_sn(k,atom_map=atom_map)
        source_lack.update({sn:v for sn,v in target.items() if sn not in source_sn})
        # print the infor
        if len(atom_map) == 0:
            nmap = 1
        else:
            nmap = len(atom_map)
        if nmap > 1:
            lack_note = ' ({:d} X {:d})'.format(nmap,int(len(target_lack)/nmap))
            overlap_note = ' ({:d} X {:d})'.format(nmap,int(len(overlap)/nmap))
        else:
            lack_note = ''
            overlap_note = ''
        if len(source_lack) > 0:
            print("NOTE: {:d} {:s} parameters in top file do not have countpart in source"
                  .format(len(source_lack),ptype))
        if len(target_lack) > 0:
            print("NOTE: target topfile lack {:d} {:s} parameters"
                  .format(len(target_lack),ptype))
        if len(overlap) > 0:
            print("NOTE: source and target have {:d} common {:s} parameters"
                  .format(len(overlap),ptype))
        return target,target_lack,overlap,lack_note,overlap_note

    def __transfer_bond(self,param_dict=None,parmed_struct=None,atom_map=[],action='replace',param='ek'):
        '''source_param is a dict where keys are tuple of atom sn in int and values are parmed bondtype/angletype object
        target is a parmded structure object, if not set then default to self.ps
        map is a list of dict of source sn to target sn, if sn same map will not be used'''
        # generate source_param dict
        source_param = {}
        if param_dict is not None:
            source_param = {k:pmd.BondType(v['k'],v['value']) for k,v in param_dict.items()}
        if parmed_struct is not None:
            for b in parmed_struct.bonds:
                source_param[(b.atom1.idx+1,b.atom2.idx+1)] = b.type
        target,target_lack,overlap,lack_note,overlap_note = self.__check_source_target(source_param,self.ps.bonds,atom_map)
        info_list = []
        if action in ['add','replace']:
            if len(target_lack) > 0:
                print("Add {:d}{:s} new bond parameters to origin parameter file"
                      .format(len(target_lack),lack_note))
            for sn,bond_type in target_lack.items():
                req = bond_type.req
                if bond_type.k is np.nan:
                    bond_type.k = 0
                k = bond_type.k
                r0, k0 = [np.nan,np.nan]
                a1,a2 = sn
                bond = pmd.Bond(self.sn2atom[a1],self.sn2atom[a2],type=bond_type)
                self.ps.bonds.append(bond)
                info_list.append(['-'.join([str(i) for i in sn]), r0, req, k0, k])
        if action in ['replace']:
            eq_replace = 0
            k_replace = 0
            for sn,bond_type in overlap.items():
                k = bond_type.k
                r0 = target[sn].type.req
                k0 = target[sn].type.k
                if 'k' not in param or k is np.nan:
                    bond_type.k = k0
                else:
                    k_replace += 1
                if 'e' not in param:
                    bond_type.req = r0
                else:
                    eq_replace += 1
                req = bond_type.req
                k = bond_type.k
                target[sn].type = bond_type
                info_list.append(['-'.join([str(i) for i in sn]), r0, req, k0, k])
            if len(overlap) > 0:
                print("Replace {:d}{:s} bond parameters ({:d} eq, {:d} k) in origin parameter file"
                      .format(len(overlap),overlap_note,eq_replace,k_replace))
        self.ps.bond_types = pmd.TrackedList()
        for b in self.ps.bonds:
            self.ps.bond_types.append(b.type)
        self.ps.bonds.claim()
        self.ps.bond_types.claim()
        # output parameters to file
        if len(info_list) > 0 and "b" in self.saveinfo:
            df = pd.DataFrame(info_list, columns=['atoms','old_eq','new_eq','old_k','new_k'])
            df['delta_eq'] = df['new_eq'] - df['old_eq']
            df['delta_k'] = df['new_k'] - df['old_k']
            if action == 'replace':
                print('Statistics on bond parameter replacement:')
                report = self.do_statistics(df,['delta_eq','delta_k'],groupby=None,transpose=True)
                print(report)
            try:
                os.mkdir('PARAM_REFINE')
            except:
                pass
            dfname = os.path.join('PARAM_REFINE',self.basename+'_RBEK.csv')
            df.to_csv(dfname,na_rep='MISSING')
            print('Save bond parameter data to {:s}'.format(dfname))

    def __transfer_angle(self,param_dict=None,parmed_struct=None,atom_map=[],action='replace',param='ek'):
        '''source_param is a dict where keys are tuple of atom sn in int and values are parmed bondtype/angletype object
        target is a parmded structure object, if not set then default to self.ps
        map is a list of dict of source sn to target sn, if sn same map will not be used'''
        source_param = {}
        if param_dict is not None:
            if self.angle_config['seminario'] == 'ms':
                print('Using modified Seminario methods (doi:10.1021/acs.jctc.7b00785) for angle parameters')
                source_param = {k:pmd.AngleType(v['km'],v['value']) for k,v in param_dict.items()}
            else:
                print('Using origin Seminario methods (Int. J. Quantum Chem. 1996, 60, 1271−1277) for angle parameters')
                source_param = {k:pmd.AngleType(v['k'],v['value']) for k,v in param_dict.items()}
        if parmed_struct is not None:
            for a in parmed_struct.angles:
                source_param[(a.atom1.idx + 1, a.atom2.idx + 1, a.atom3.idx + 1)] = a.type
        target,target_lack,overlap,lack_note,overlap_note = self.__check_source_target(source_param,self.ps.angles,atom_map)
        param2str = {'ek':'equilibrium angle and force constant',
                     'e':'equilibrium angle',
                     'k':'force constant'}
        info_list = []
        if action in ['add','replace']:
            if len(target_lack) > 0:
                print("Add {:d}{:s} new angle parameters to origin parameter file"
                      .format(len(target_lack),lack_note))
            for sn,angle_type in target_lack.items():
                req = angle_type.theteq
                if angle_type.k  is np.nan:
                    angle_type.k = 0
                k = angle_type.k
                r0, k0 = [np.NaN,np.NaN]
                a1,a2,a3 = sn
                angle = pmd.Angle(self.sn2atom[a1],self.sn2atom[a2],self.sn2atom[a3],type=angle_type)
                self.ps.angles.append(angle)
                info_list.append(['-'.join([str(i) for i in sn]), r0, req, k0, k])
        if action in ['replace']:
            eq_replace = 0
            k_replace = 0
            for sn,angle_type in overlap.items():
                k = angle_type.k
                r0 = target[sn].type.theteq
                k0 = target[sn].type.k
                if 'k' not in param or k is np.nan:
                    angle_type.k = k0
                else:
                    k_replace += 1
                if 'e' not in param:
                    angle_type.theteq = r0
                else:
                    eq_replace += 1
                req = angle_type.theteq
                k = angle_type.k
                target[sn].type = angle_type
                info_list.append(['-'.join([str(i) for i in sn]), r0, req, k0, k])
            if len(overlap) > 0:
                print("Replace {:d}{:s} angle parameters ({:d} eq, {:d} k) in origin parameter file"
                      .format(len(overlap),overlap_note,eq_replace,k_replace))
        self.ps.angle_types = pmd.TrackedList()
        for a in self.ps.angles:
            self.ps.angle_types.append(a.type)
        self.ps.angles.claim()
        self.ps.angle_types.claim()
        # output parameters to file
        if len(info_list) > 0 and "a" in self.saveinfo:
            df = pd.DataFrame(info_list, columns=['atoms','old_eq','new_eq','old_k','new_k'])
            df['delta_eq'] = df['new_eq'] - df['old_eq']
            df['delta_k'] = df['new_k'] - df['old_k']
            if action == 'replace':
                print('Statistics on angle parameter replacement:')
                report = self.do_statistics(df,['delta_eq','delta_k'],groupby=None,transpose=True)
                print(report)
            try:
                os.mkdir('PARAM_REFINE')
            except:
                pass
            dfname = os.path.join('PARAM_REFINE',self.basename+'_RAEK.csv')
            df.to_csv(dfname,na_rep='MISSING')
            print('Save angle parameter data to {:s}'.format(dfname))

    def __transfer_dihedral(self,param_dict=None,parmed_struct=None,atom_map=[],action='replace',param='ek'):
        '''source_param is a dict where keys are tuple of atom sn in int and values are parmed bondtype/angletype object
        target is a parmded structure object, if not set then default to self.ps
        map is a list of dict of source sn to target sn, if sn same map will not be used'''
        # begin param transfer
        # The dihedral type of source and target should be of parmed tracked list type
        source_param = {}
        if param_dict is not None:
            if len(self.ps.dihedral_types) > 0:
                dt = self.ps.dihedral_types[0]
                if type(dt) is pmd.DihedralType:
                    scee = dt.scee
                    scnb = dt.scnb
                else:
                    scee = dt[0].scee
                    scnb = dt[0].scnb
            else:
                print("dihedral type not found in target top\n"
                      "I will set scee (fudgeQQ) to 0.833333 and scnb (fudgeLJ) to 0.5")
                scee = 1.2
                scnb = 2
            source_param = {k:pmd.DihedralType(v['k'],v['period'],v['period']*v['value']-180, scee=scee, scnb=scnb)
                               for k,v in param_dict.items()}
        if parmed_struct is not None:
            for d in parmed_struct.dihedrals:
                if d.improper == False:
                    source_param[(d.atom1.idx + 1, d.atom2.idx + 1, d.atom3.idx + 1, d.atom4.idx + 1)] = d.type
        target,target_lack,overlap,lack_note,overlap_note = self.__check_source_target(source_param,self.ps.dihedrals,atom_map)
        info_list = []
        if action in ['add','replace']:
            if len(target_lack) > 0:
                print("Add {:d}{:s} new dihedral parameters to origin parameter file"
                      .format(len(target_lack),lack_note))
            for sn,dihedral_type in target_lack.items():
                if type(dihedral_type) is pmd.DihedralType:
                    if dihedral_type.phi_k is np.nan:
                        dihedral_type.phi_k = 0
                        dihedral_type.phase = 0
                        dihedral_type.per = 1
                    if 'k' not in param:
                        dihedral_type.phi_k = 0
                        dihedral_type.phase = 180 - dihedral_type.per%2*180
                    period = '{:d}'.format(int(dihedral_type.per))
                    k = '{:.3f}'.format(dihedral_type.phi_k)
                    phase = '{:d}'.format(int(dihedral_type.phase))
                elif type(dihedral_type) is pmd.DihedralTypeList:
                    period =';'.join(['{:d}'.format(int(i.per)) for i in dihedral_type])
                    k = ';'.join(['{:.3f}'.format(i.phi_k) for i in dihedral_type])
                    phase = ';'.join(['{:d}'.format(int(i.phase)) for i in dihedral_type])
                period0, phase0,k0 = [np.nan,np.nan,np.nan]
                a1,a2,a3,a4 = sn
                dihedral = pmd.Dihedral(self.sn2atom[a1],self.sn2atom[a2],self.sn2atom[a3],self.sn2atom[a4],
                                        improper=False,type=dihedral_type)
                self.ps.dihedrals.append(dihedral)
                info_list.append(['-'.join([str(i) for i in sn]), period0, period, k0, k,phase0,phase])
        if action in ['replace']:
            if len(overlap) > 0:
                print("Replace {:d}{:s} dihedral parameters in origin parameter file"
                      .format(len(overlap),overlap_note))
            for sn,dihedral_type in overlap.items():
                if type(dihedral_type) is pmd.DihedralType:
                    if dihedral_type.phi_k is np.nan:
                        dihedral_type.phi_k = 0
                        dihedral_type.phase = 0
                        dihedral_type.per = 1
                    period = '{:d}'.format(int(dihedral_type.per))
                    k = '{:.3f}'.format(dihedral_type.phi_k)
                    phase = '{:d}'.format(int(dihedral_type.phase))
                elif type(dihedral_type) is pmd.DihedralTypeList:
                    period =';'.join(['{:d}'.format(int(i.per)) for i in dihedral_type])
                    k = ';'.join(['{:.3f}'.format(i.phi_k) for i in dihedral_type])
                    phase = ';'.join(['{:d}'.format(int(i.phase)) for i in dihedral_type])
                if type(target[sn].type) is pmd.DihedralType:
                    period0 = '{:d}'.format(int(target[sn].type.per))
                    k0 = '{:.3f}'.format(target[sn].type.phi_k)
                    phase0 = '{:d}'.format(int(target[sn].type.phase))
                elif type(target[sn].type) is pmd.DihedralTypeList:
                    period0 =';'.join(['{:d}'.format(int(i.per)) for i in target[sn].type])
                    k0 =';'.join(['{:.3f}'.format(int(i.phi_k)) for i in target[sn].type])
                    phase0 =';'.join(['{:d}'.format(int(i.phase)) for i in target[sn].type])
                target[sn].type = dihedral_type
                info_list.append(['-'.join([str(i) for i in sn]), period0, period, k0, k,phase0,phase])
        self.ps.dihedral_types = pmd.TrackedList()
        for d in self.ps.dihedrals:
            self.ps.dihedral_types.append(d.type)
        self.ps.dihedrals.claim()
        self.ps.dihedral_types.claim()
        # output parameters to file
        if len(info_list) > 0 and "d" in self.saveinfo:
            df = pd.DataFrame(info_list, columns=['atoms','old_period','new_period','old_k','new_k','old_phase','new_phase'])
            try:
                os.mkdir('PARAM_REFINE')
            except:
                pass
            dfname = os.path.join('PARAM_REFINE',self.basename+'_RDPPK.csv')
            df.to_csv(dfname,na_rep='MISSING')
            print('Save dihedral parameter data to {:s}'.format(dfname))

    def __transfer_improper(self,param_dict=None,parmed_struct=None,atom_map=[],action='replace',param='ek'):
        '''note there are two kinds of improper dihedral type with improper=True or improper type'''
        source_param_period = {} 
        source_param_quadra = {}
        if param_dict is not None:
            if 'period' in next(iter(param_dict.values())):
                if len(self.ps.dihedral_types) > 0:
                    dt = self.ps.dihedral_types[0]
                    if type(dt) is pmd.DihedralType:
                        scee = dt.scee
                        scnb = dt.scnb
                    else:
                        scee = dt[0].scee
                        scnb = dt[0].scnb
                else:
                    print("dihedral type not found in target top\n"
                          "I will set scee (fudgeQQ) to 0.833333 and scnb (fudgeLJ) to 0.5")
                    scee = 0.833333
                    scnb = 0.5
                source_param_period = {k:pmd.DihedralType(v['k'],v['period'],180.0, scee=scee, scnb=scnb)
                                       for k,v in param_dict.items()}
            else:
                source_param_quadra = {k:pmd.ImproperType(v['k'],v['value'])
                                       for k,v in param_dict.items()}
        if parmed_struct is not None:
            for d in parmed_struct.dihedrals:
                if d.improper == True:
                    if type(d.type) is pmd.DihedralTypeList:
                        source_param_period[(d.atom1.idx + 1, d.atom2.idx + 1, d.atom3.idx + 1, d.atom4.idx + 1)] = d.type[0]
                    else:
                        source_param_period[(d.atom1.idx + 1, d.atom2.idx + 1, d.atom3.idx + 1, d.atom4.idx + 1)] = d.type
            for i in parmed_struct.impropers:
                source_param_quadra[(i.atom1.idx + 1, i.atom2.idx + 1, i.atom3.idx + 1, i.atom4.idx + 1)] = i.type
        tp,tlp,op,lnp,onp = self.__check_source_target(source_param_period,self.ps.dihedrals,atom_map,improper=True)
        tq,tlq,oq,lnq,onq = self.__check_source_target(source_param_quadra,self.ps.impropers,atom_map,improper=True)
        info_listp = []
        info_listq = []
        if action in ['add','replace']:
            if len(tlp) > 0:
                print("Add {:d}{:s} new periodic improper parameters to origin parameter file"
                      .format(len(tlp),lnp))
            for sn,dihedral_type in tlp.items():
                if dihedral_type.phi_k is np.nan or 'k' not in param:
                    dihedral_type.phi_k = 0
                    dihedral_type.phase = 180
                    dihedral_type.per = 2
                period = '{:d}'.format(int(dihedral_type.per))
                k = '{:.3f}'.format(dihedral_type.phi_k)
                phase = '{:d}'.format(int(dihedral_type.phase))
                period0,phase0,k0 = [np.nan,np.nan,np.nan]
                a1,(a2,a3,a4) = sn
                dihedral = pmd.Dihedral(self.sn2atom[a1],self.sn2atom[a2],self.sn2atom[a3],self.sn2atom[a4],
                                        improper=True,type=dihedral_type)
                self.ps.dihedrals.append(dihedral)
                info_listp.append(['-'.join([str(sn[0])]+[str(i) for i in sn[1]]), period0, period, k0, k,phase0,phase])
            if len(tlq) > 0:
                print("Add {:d}{:s} new quadratic improper parameters to origin parameter file"
                      .format(len(tlq),lnq))
            for sn,improper_type in tlq.items():
                if improper_type.psi_k is np.nan or 'k' not in param:
                    improper_type.psi_k = 0
                k = '{:.3f}'.format(improper_type.psi_k)
                phase = '{:d}'.format(int(improper_type.psi_eq))
                k0, phase0 = [np.nan,np.nan]
                a1,(a2,a3,a4) = sn
                improper = pmd.Improper(self.sn2atom[a1],self.sn2atom[a2],self.sn2atom[a3],self.sn2atom[a4],type=improper_type)
                self.ps.impropers.append(improper)
                info_listq.append(['-'.join([str(sn[0])]+[str(i) for i in sn[1]]),k0, k,phase0,phase])
        if action in ['replace']:
            if len(op) > 0:
                print("Replace {:d}{:s} improper parameters with periodic form in origin parameter file"
                      .format(len(op),onp))
            for sn,dihedral_type in op.items():
                if dihedral_type.phi_k is np.nan or 'k' not in param:
                    dihedral_type.phi_k = 0
                    dihedral_type.phase = 180
                    dihedral_type.per = 2
                period = '{:d}'.format(int(dihedral_type.per))
                k = '{:.3f}'.format(dihedral_type.phi_k)
                phase = '{:d}'.format(int(dihedral_type.phase))
                period0 = '{:d}'.format(int(tp[sn].type.per))
                k0 = '{:.3f}'.format(tp[sn].type.phi_k)
                phase0 = '{:d}'.format(int(tp[sn].type.phase))
                tp[sn].type = dihedral_type
                info_listp.append(['-'.join([str(sn[0])]+[str(i) for i in sn[1]]), period0, period, k0, k,phase0,phase])
            if len(oq) > 0:
                print("Replace {:d}{:s} improper parameters with quadratic form in origin parameter file"
                      .format(len(oq),onq))
            for sn,improper_type in oq.items():
                if improper_type.psi_k is np.nan or 'k' not in param:
                    improper_type.psi_k = 0
                k = '{:.3f}'.format(improper_type.psi_k)
                phase = '{:d}'.format(int(improper_type.psi_eq))
                k0 = '{:.3f}'.format(tq[sn].type.psi_k)
                phase0 = '{:d}'.format(int(tq[sn].type.psi_eq))
                tq[sn].type = improper_type
                info_listq.append(['-'.join([str(sn[0])]+[str(i) for i in sn[1]]), k0, k,phase0,phase])
        if len(info_listp) > 0:
            self.ps.dihedral_types = pmd.TrackedList()
            for d in self.ps.dihedrals:
                self.ps.dihedral_types.append(d.type)
            self.ps.dihedrals.claim()
            self.ps.dihedral_types.claim()
        if len(info_listq) > 0:
            self.ps.improper_types = pmd.TrackedList()
            for d in self.ps.impropers:
                self.ps.improper_types.append(d.type)
            self.ps.impropers.claim()
            self.ps.improper_types.claim()
        # output parameters to file
        if len(info_listp) > 0 and "i" in self.saveinfo:
            df = pd.DataFrame(info_listp, columns=['atoms','old_period','new_period','old_k','new_k','old_phase','new_phase'])
            try:
                os.mkdir('PARAM_REFINE')
            except:
                pass
            dfname = os.path.join('PARAM_REFINE',self.basename+'_RIPPK.csv')
            df.to_csv(dfname,na_rep='MISSING')
            print('Save periodic improper parameter data to {:s}'.format(dfname))
        if len(info_listq) > 0 and "i" in self.saveinfo:
            df = pd.DataFrame(info_listq, columns=['atoms','old_k','new_k','old_eq','new_eq'])
            try:
                os.mkdir('PARAM_REFINE')
            except:
                pass
            dfname = os.path.join('PARAM_REFINE',self.basename+'_RIEK.csv')
            df.to_csv(dfname,na_rep='MISSING')
            print('Save quadratic improper parameter data to {:s}'.format(dfname))

    def __transfer_charge(self,parmed_struct=None,atom_map=[],param=''):
        ''' if s in param then the charge of source will be multiple by a number to make it equal to target
        if d in param then the charge of source will be added a number to make the sum equal
        '''
        info_list = []
        target_charge = []
        source_charge = []
        delta_charge = []
        scale_charge = []
        for i,m in enumerate(atom_map):
            target_atom = m.values()
            source_atom = m.keys()
            chg_target = [self.ps.atoms[i-1].charge for i in target_atom]
            chg_source = [parmed_struct.atoms[i-1].charge for i in source_atom]
            if 's' in param:
                scale = sum(chg_target)/sum(chg_source)
                target_charge.append(sum(chg_target))
                source_charge.append(sum(chg_source))
                scale_charge.append(scale)
                delta_charge.append(0)
                for k,v in m.items():
                    info_list.append([k,v,parmed_struct.atoms[k-1].charge,self.ps.atoms[v-1].charge,0,scale])
                    self.ps.atoms[v-1].charge = parmed_struct.atoms[k-1].charge*scale
            elif 'd' in param:
                delta =  (sum(chg_source) - sum(chg_target)) /len(chg_source)
                target_charge.append(sum(chg_target))
                source_charge.append(sum(chg_source))
                scale_charge.append(1)
                delta_charge.append(delta)
                print('{:03d}: Origin charge: {:.3f}; replace charge: {:.3f}; delta charge: {:.3f}; delta per atom: {:.3e}'
                       .format(i+1,sum(chg_target),sum(chg_source),sum(chg_source) - sum(chg_target),delta))
                for k,v in m.items():
                    info_list.append([k,v,parmed_struct.atoms[k-1].charge,self.ps.atoms[v-1].charge,delta,1])
                    self.ps.atoms[v-1].charge = parmed_struct.atoms[k-1].charge + delta
            else:
                target_charge.append(sum(chg_target))
                source_charge.append(sum(chg_source))
                scale_charge.append(1)
                delta_charge.append(0)
                for k,v in m.items():
                    info_list.append([k,v,parmed_struct.atoms[k-1].charge,self.ps.atoms[v-1].charge,0,0])
                    self.ps.atoms[v-1].charge = parmed_struct.atoms[k-1].charge
        sum_delta = np.sum((np.array(source_charge) + np.array(delta_charge)) * np.array(scale_charge) - np.array(target_charge))
        print('{:d} groups of {:d} atom charges transfered'.format(len(atom_map),len(m)))
        print('Average origin charge of the group: {:.3f}, Average new charge of the group: {:.3f}\n'
              'charge of each atom is added by {:.3e} and scaled by {:.3f}\n'
              'total charge change after transfer is {:.3f} '
              .format(np.mean(target_charge),np.mean(source_charge),
                      np.mean(delta_charge),np.mean(scale_charge),sum_delta))
        if len(info_list) > 0 and "t" in self.saveinfo:
            df = pd.DataFrame(info_list, columns=['source_sn','target_sn','new_charge','old_charge','delta','scale'])
            try:
                os.mkdir('PARAM_REFINE')
            except:
                pass
            dfname = os.path.join('PARAM_REFINE',self.basename+'_TransCharge.csv')
            df.to_csv(dfname,na_rep='MISSING')
            print('Save atom charge transfer data to {:s}'.format(dfname))

    def __transfer_type(self,parmed_struct=None,atom_map=[],param=''):
        type_replace = []
        type_same = []
        info_list = []
        for _,m in enumerate(atom_map):
            for k,v in m.items():
                if self.ps.atoms[v-1].type != parmed_struct.atoms[k-1].type:
                    info_list.append([k,v,parmed_struct.atoms[k-1].type,self.ps.atoms[v-1].type])
                    type_replace.append(' --- '.join((self.ps.atoms[v-1].type,parmed_struct.atoms[k-1].type)))
                    self.ps.atoms[v-1].type = parmed_struct.atoms[k-1].type
                    self.ps.atoms[v-1].atom_type = parmed_struct.atoms[k-1].atom_type
                else:
                    type_same.append(self.ps.atoms[v-1].type)
        if len(type_replace) == 0:
            print('all atoms are of same type. No replacement happens')
        else:
            replace_count  = [k+' X '+ str(v) for k,v in Counter(type_replace).items()]
            print('{:d} atoms of same type are kept\n'
                  '{:d} atoms of different type are replaced by new type;\n'
                  'substituded atom types are:\n{:s}\n'
                  .format(len(type_same),len(type_replace),'\n'.join(replace_count)))
        if len(info_list) > 0 and "t" in self.saveinfo:
            df = pd.DataFrame(info_list, columns=['source_sn','target_sn','new_type','old_type'])
            try:
                os.mkdir('PARAM_REFINE')
            except:
                pass
            dfname = os.path.join('PARAM_REFINE',self.basename+'_TransType.csv')
            df.to_csv(dfname,na_rep='MISSING')
            print('Save atom type transfer data to {:s}'.format(dfname))

    def __transfer_vdw(self,parmed_struct=None,atom_map=[],param=''):
        pass

    def adj_dihedral(self,contract='auto',param_struct=''):
        '''contract all/none/auto dihedral types to dihedral type list '''
        if not param_struct:
            param_struct=self.ps
        d2dtype = defaultdict(str)
        for d in param_struct.dihedrals:
            a1 = d.atom1.idx
            a2 = d.atom2.idx
            a3 = d.atom3.idx
            a4 = d.atom4.idx
            d_idx = (a1,a2,a3,a4)
            atoms = (d.atom1,d.atom2,d.atom3,d.atom4)
            if type(d.type) is pmd.DihedralType:
                typelist = [d.type]
            elif type(d.type) is pmd.DihedralTypeList:
                typelist = [t for t in d.type]
            if  d_idx not in d2dtype:
                d2dtype[d_idx] =  {'typelist':typelist,'atoms':atoms,'ignore_end':d.ignore_end,'improper':d.improper}
            else:
                d2dtype[d_idx]['typelist'] += typelist
                d2dtype[d_idx]['ignore_end'] = d2dtype[d_idx]['ignore_end'] and d.ignore_end
        # remove d type with zero k and
        # for a,v in d2dtype.items():
        #     for dtype in v['typelist']:
        del param_struct.dihedrals[:]
        del param_struct.dihedral_types[:]
        param_struct.dihedral_types = pmd.TrackedList()
        param_struct.dihedrals = pmd.TrackedList()
        if contract == 'auto':
            for k,v in d2dtype.items():
                if len(v['typelist'])==1:
                    dtype = v['typelist'][0]
                    # dtype.list = param_struct.dihedral_types
                    inlist = 0
                    for d in param_struct.dihedral_types:
                        if type(d) is pmd.DihedralType:
                            if dtype == d:
                                dtype = d
                                inlist = 1
                    if inlist == 0:
                        param_struct.dihedral_types.append(dtype)
                    dihed = pmd.Dihedral(*v['atoms'],improper=v['improper'],ignore_end=v['ignore_end'],type=dtype)
                    param_struct.dihedrals.append(dihed)
                if len(v['typelist'])>1:
                    dtype = pmd.DihedralTypeList(v['typelist'], list=param_struct.dihedral_types)
                    inlist = 0
                    for d in param_struct.dihedral_types:
                        if type(d) is pmd.DihedralTypeList:
                            if dtype == d:
                                dtype = d
                                inlist = 1
                    if inlist == 0:
                        param_struct.dihedral_types.append(dtype)
                    dihed = pmd.Dihedral(*v['atoms'],improper=v['improper'],ignore_end=v['ignore_end'],type=dtype)
                    param_struct.dihedrals.append(dihed)

        if contract == 'all':
            for k,v in d2dtype.items():
                dtype = pmd.DihedralTypeList(v['typelist'],list=param_struct.dihedral_types)
                inlist = 0
                for d in param_struct.dihedral_types:
                    if dtype == d:
                        dtype = d
                        inlist =1
                if inlist == 0:
                    param_struct.dihedral_types.append(dtype)
                dihed = pmd.Dihedral(*v['atoms'], improper=v['improper'], ignore_end=v['ignore_end'], type=dtype)
                param_struct.dihedrals.append(dihed)

        if contract == 'none':
            for k,v in d2dtype.items():
                for dtype in v['typelist']:
                    dtype.list = param_struct.dihedral_types
                    inlist = 0
                    for d in param_struct.dihedral_types:
                        if dtype == d:
                            dtype = d
                            inlist = 1
                    if inlist == 0:
                        param_struct.dihedral_types.append(dtype)
                    dihed = pmd.Dihedral(*v['atoms'], improper=v['improper'], ignore_end=v['ignore_end'], type=dtype)
                    param_struct.dihedrals.append(dihed)

    def compare_structure(self,coords1,coords2):
        '''compare coord with minimized stucture'''
        coords1 = np.array(coords1)/10
        bad1 = self.get_badi(coords1)
        coords2 = np.array(coords2)/10
        bad2 = self.get_badi(coords2)
        return self.compare_BAD(bad1,bad2)

    def do_statistics(self,df,col_name,groupby='type',transpose=False):
        def mae(x):
            return x.abs().mean()
        def rmse(x):
            return (x**2).mean()**0.5
        agg_dict = {i:[mae,rmse,'mean','min','max','count'] for i in col_name}
        # print(agg_dict)
        if groupby:
            report = df.groupby(groupby).agg(agg_dict)
            report.columns = report.columns.droplevel()
        else:
            report = df.agg(agg_dict)
        if transpose:
            report = report.transpose()
        report.index.name = None
        return report

    def minimize(self):
        print('Begin geometry optimization by openmm ... ')
        start = timeit.default_timer()
        import openmm as mm
        import openmm.app as app
        from openmm.unit import kelvin, picoseconds, femtoseconds, nanometer
        integrator = mm.LangevinIntegrator(
            300 * kelvin,  # Temperature of heat bath
            1.0 / picoseconds,  # Friction coefficient
            2.0 * femtoseconds,  # Time step
        )
        try:
            if self.ps.box is not None:
                system = self.ps.createSystem(nonbondedMethod=app.PME, constraints=app.HBonds, nonbondedCutoff=1*nanometer)
            else:
                system = self.ps.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            sim = app.Simulation(self.ps.topology, system, integrator)
        except:
            self.ps.save('tempforopenmm.prmtop',overwrite=True)
            prmtop = app.AmberPrmtopFile('tempforopenmm.prmtop')
            if self.ps.box is not None:
                system = prmtop.createSystem(nonbondedMethod=app.PME, constraints=app.HBonds, nonbondedCutoff=1*nanometer)
            else:
                system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            sim = app.Simulation(prmtop.topology, system, integrator)
            os.remove('tempforopenmm.prmtop')
        # Create the integrator to do Langevin dynamics
        # Set the particle positions
        if self.ps.box is not None:
            sim.context.setPeriodicBoxVectors(*self.ps.box_vectors)
        ##### !!! this is not tested 
        sim.context.setPositions(self.ps.positions)
        sim.minimizeEnergy(tolerance=1)
        state = sim.context.getState(getEnergy=True)
        energy= state.getPotentialEnergy()
        positions = sim.context.getState(getPositions=True)
        coords = positions.getPositions(True)*10
        end = timeit.default_timer()
        print('Geometry optmization by openmm took {:.3f} seconds'.format(end-start))
        return coords,energy


    def get_badi_param(self, param_struct=''):
        bonds_param={}
        angles_param={}
        dihedrals_param={}
        impropers_param={}
        if not param_struct:
            param_struct = self.ps
        for b in param_struct.bonds:
            bonds_param[(b.atom1.idx+1,b.atom2.idx+1)] = b
        for a in param_struct.angles:
            angles_param[(a.atom1.idx + 1, a.atom2.idx + 1, a.atom3.idx + 1)] = a
        for d in param_struct.dihedrals:
            if d.improper == False:
                dihedrals_param[(d.atom1.idx + 1, d.atom2.idx + 1, d.atom3.idx + 1, d.atom4.idx + 1)] = d
            if d.improper == True:
                impropers_param[(d.atom1.idx + 1, d.atom2.idx + 1, d.atom3.idx + 1, d.atom4.idx + 1)] = d
        return bonds_param,angles_param,dihedrals_param,impropers_param

    def get_badi(self,coords='', param_struct=''):
        '''compute bond length angle and diheral for ps'''
        bonds=defaultdict(str)
        angles=defaultdict(str)
        dihedrals=defaultdict(str)
        impropers=defaultdict(str)
        if len(coords)==0:
            c = np.array(self.ps.coordinates)
        elif len(coords) == len(self.ps.coordinates):
            c = np.array(coords)
        else:
            print('Warning!!! The length of input coords {:d} do not match with coords in Parmed structures {:d}. I will omit input coords'.format(len(coords),len(self.ps.coordinates)))
            c = np.array(self.ps.coordinates)
        if not param_struct:
            param_struct = self.ps
        for b in param_struct.bonds:
            coords1 = c[b.atom1.idx]
            coords2 = c[b.atom2.idx]
            dist = np.linalg.norm(coords1 - coords2)
            bonds[(b.atom1.idx+1,b.atom2.idx+1)] = dist
        for a in param_struct.angles:
            c1 = c[a.atom1.idx]
            c2 = c[a.atom2.idx]
            c3 = c[a.atom3.idx]
            v1 = c3 - c2
            v2 = c1 - c2
            cosang = np.dot(v1, v2)
            sinang = np.linalg.norm(np.cross(v1, v2))
            angle = np.rad2deg(np.arctan2(sinang, cosang))
            angles[(a.atom1.idx+1,a.atom2.idx+1,a.atom3.idx+1)] = angle
        for d in param_struct.dihedrals:
            if d.improper == False:
                c1 = c[d.atom1.idx]
                c2 = c[d.atom2.idx]
                c3 = c[d.atom3.idx]
                c4 = c[d.atom4.idx]
                b0 = -1.0 *(c2-c1)
                b1 = c3 - c2
                b2 = c4 - c3
                b1 /= np.linalg.norm(b1)
                v = b0 - np.dot(b0, b1) * b1
                w = b2 - np.dot(b2, b1) * b1
                x = np.dot(v, w)
                y = np.dot(np.cross(b1, v), w)
                dihedral = np.degrees(np.arctan2(y, x))
                dihedrals[(d.atom1.idx+1,d.atom2.idx+1,d.atom3.idx+1,d.atom4.idx+1)] = dihedral
            else:
                impropers[(d.atom1.idx + 1, d.atom2.idx + 1, d.atom3.idx + 1, d.atom4.idx + 1)] = d
        # if param_struct is self.ps:
        #     for k,v in bonds_param.items():
        #         v.type.k=0
        #     for b in self.ps.bonds:
        #         print(b)

        return bonds,angles,dihedrals

    def compare_BAD(self,bad1,bad2):
        bond1,angle1,dihedral1 = bad1
        bond2,angle2,dihedral2 = bad2
        if list(bond1.keys()) == list(bond2.keys()):
            bond_diff = (np.array(list(bond1.values()))-np.array(list(bond2.values()))) * 10
            bond_info = zip(['-'.join(map(str,i)) for i in bond1.keys()],bond_diff)
            bond_info = [['bond'] + list(i) for i in bond_info]
        else:
            print('The bonds are different, so there will be no bond length comparison')
            bond_info = []
        if list(angle1.keys()) == list(angle2.keys()):
            angle_diff = np.array(list(angle1.values()))-np.array(list(angle2.values()))
            angle_info = list(zip(['-'.join(map(str,i)) for i in angle1.keys()],angle_diff))
            angle_info = [['angle'] + list(i) for i in angle_info]
        else:
            print('The angles are different, so there will be no angle comparison')
            angle_info = []
        if list(dihedral1.keys()) == list(dihedral2.keys()):
            raw_dihed_diff = np.array(list(dihedral1.values()))-np.array(list(dihedral2.values()))
            dihedral_diff = []
            for d in raw_dihed_diff:
                if d > 180:
                    d = 360-d
                if d < -180:
                    d = (d+360) * -1
                dihedral_diff.append(d)
            dihedral_diff = np.array(dihedral_diff)
            dihedral_info = list(zip(['-'.join(map(str,i)) for i in dihedral1.keys()],dihedral_diff))
            dihedral_info = [['dihedral'] + list(i) for i in dihedral_info]
        else:
            print('The dihedrals are different, so there will be no dihedral comparison')
            dihedral_info = []
        all_info = bond_info + angle_info + dihedral_info
        df = pd.DataFrame(all_info,columns=['type','atom','delta'])
        report = self.do_statistics(df,['delta'])
        return report

        # bond_mae = np.mean(np.abs(bond_diff))*10
        # bond_mse = np.mean(bond_diff)*10
        # bond_rmse = np.sqrt(np.mean(bond_diff**2))*10
        # bond_max = np.max(bond_diff)*10
        # bond_min = np.min(bond_diff)*10
        # angle_mae = np.mean(np.abs(angle_diff))
        # angle_mse = np.mean(angle_diff)
        # angle_rmse = np.sqrt(np.mean(angle_diff**2))
        # angle_max = np.max(angle_diff)
        # angle_min = np.min(angle_diff)
        # dihedral_mae = np.mean(np.abs(dihedral_diff))
        # dihedral_mse = np.mean(dihedral_diff)
        # dihedral_rmse = np.sqrt(np.mean(dihedral_diff**2))
        # dihedral_max = np.max(dihedral_diff)
        # dihedral_min = np.min(dihedral_diff)
        # info_list = [['bond',bond_mae,bond_rmse,bond_mse,bond_max,bond_min,len(bond_diff)]]
        # info_list.append(['angle',angle_mae,angle_rmse,angle_mse,angle_max,angle_min,len(angle_diff)])
        # info_list.append(['dihedral',dihedral_mae,dihedral_rmse,dihedral_mse,dihedral_max,dihedral_min,len(dihedral_diff)])
        # df = pd.DataFrame(info_list,columns=['type','mae','rmse','mse','max','min','count'])
        # df.set_index('type',inplace=True)
        # df.index.name = None
        # print(df)
