from .paramrefine import ParamRefine
from collections import defaultdict
import parmed as pmd
import copy
import sys
import os
import pandas
import shutil
import glob
import timeit
sys.path.append('..')
import coordmagic as cm
import networkx as nx
import numpy as np
import re


class ResidueComposer:
    '''generate Residue by cutting bond from molecule'''
    def __init__(self):
        self.res_lib = {}

    def _get_bond(self,ps, idx):
        for b in ps.bonds:
            if b.atom1.idx in idx and b.atom2.idx in idx:
                return b
    def _get_atoms(self,ps, idx):
        atoms = []
        for i in idx:
            a = [a for a in ps.atoms if a.idx == i][0]
            atoms.append(a)
        return atoms
    def _get_angles(self,ps, idx):
        angles = []
        for a in ps.angles:
            if (a.atom1.idx in idx or a.atom3.idx in idx) and a.atom2.idx == idx[1]:
                angles.append(a)
        return angles
    def _get_dihedrals(self,ps, idx):
        dihedrals = []
        for d in ps.dihedrals:
            if (d.atom1.idx == idx[0] and d.atom2.idx == idx[1]) or (
                    d.atom3.idx == idx[1] and d.atom4.idx == idx[0]):
                dihedrals.append(d)
        return dihedrals
    def _get_inter_dihedrals(self,ps,idx):
        dihedrals = []
        for d in ps.dihedrals:
            if (d.atom2.idx == idx[0] and d.atom3.idx == idx[1]) or (
                    d.atom3.idx == idx[0] and d.atom2.idx == idx[1]):
                dihedrals.append(d)
        return dihedrals

    def parse_residue(self,res_str):
        '''input a res string and add label,res_object pair to res lib'''
        res_label = res_str.split(';')[0]
        res_dict = {i.split('=')[0]: i.split('=')[1] for i in res_str.split(';')[1:]}
        if 'p' in res_dict:
            top= res_dict['p']
            pr = ParamRefine(top_file = top)
            pr.adj_dihedral(contract='all')
            if 'x' in res_dict:
                pr.load_coord(res_dict['x'])
            if 'resname' in res_dict:
                resname = res_dict['resname']
            else:
                resname = os.path.splitext(os.path.basename(top))[0][:3]
            if 'comment' in res_dict:
                comment = res_dict['comment']
            else:
                comment = ''
            if 'site' in res_dict:
                site = {i.split(':')[0]: i.split(':')[1] for i in res_dict['site'].split(',')}
                res = self.gen_res(pr, resname=resname, **site)
            else:
                res = self.gen_res(pr, resname=resname)
            res['label'] = res_label
            res['defstr'] = res_str
            res['comment'] = comment
            res['top'] = top
            res['coord'] = res_dict['x']
            res['resname'] = resname
            self.res_lib[res_label] = res
            # print(res['sites'])

    def print_lib(self):
        frag_list = []
        print_info = ['label', 'comment']
        for lbl,res in self.res_lib.items():
            frag = defaultdict(str)
            sites = []
            for k,v in res['sites'].items():
                sitestr = '-'.join([i.element_name+str(len(i.bonds)-1) for i in v['atoms']])
                sites.append(k+':'+sitestr)
            frag.update({k:v for k,v in res.items() if k in print_info})
            frag['formula'] = " " + res['cm_frag']['formula']
            frag['sites'] = " "+', '.join(sites)
            frag_list.append(frag)
        df = pandas.DataFrame(frag_list)
        if(all(df['comment']=='')):
            df = df[['label','formula','sites']]
        else:
            df = df[['label','formula','sites','comment']]
        print("Fragments in the lib:")
        print(df.to_string(index=False))

    def load_lib(self):
        cwd = os.getcwd()
        try:
            os.chdir("FRAG_LIB")
        except:
            print("Error! Could not enter FRAG_LIB directory")
        frgs = glob.glob("*.frg")
        all_def = []
        for frg in frgs:
            all_def += [i.strip() for i in open(frg,"r").readlines()]
        for frg_def in set(all_def):
            self.parse_residue(frg_def)
        os.chdir(cwd)

    def save_lib(self):
        cwd = os.getcwd()
        wkdir = "FRAG_LIB"
        try:
            os.mkdir(wkdir)
        except FileExistsError:
            pass
        os.chdir(wkdir) 
        all = open('ALL.frg','w')
        for lbl,res in self.res_lib.items():
            flnm = '_'.join([res['resname'],res['label']])+'.frg'
            all.write(res['defstr']+'\n')
            with open(flnm,'w') as frg:
                frg.write(res['defstr']+'\n')
            shutil.copy(os.path.join(cwd,res['top']),res['top'])
            shutil.copy(os.path.join(cwd,res['coord']),res['coord'])
        all.close()
        print('Fragments in the lib have been saved in FRAG_LIB')
        os.chdir(cwd)

    def gen_res(self,paramrf,resname='',**cutting_site):
        '''generate a defaultdict to represent a residue
        this dict have keys of 'cm_frag' which is a coordMagic fragment object
        'pmd_struct' which is a parmed structure object
        and 'sites' the value of which are a dict of dict as
        '{siteA:{atoms:(a1,a2),bonds:parmed_bond_obj,angles:[parmed_angle_objs],
        dihedrals:[]}'
        '''
        paramrf.adj_dihedral(contract='all')
        # convert to amber parm type to add dihedral
        # gromacs type can not gen pairt form 1,4 automatically
        # parmed_struct = pmd.amber.AmberParm.from_structure(paramrf.ps)
        parmed_struct = paramrf.ps
        res=defaultdict(str)
        res['sites'] = defaultdict(dict)
        st = cm.conver_structure(parmed_struct,'parmed')
        print('Generating graph of {:s} ... '.format(paramrf.basename),end='',flush=True)
        start=timeit.default_timer()
        st.G.gen_mol()
        end=timeit.default_timer()
        print('Graph generation took {:.3f} seconds'.format(end-start))
        cm_frag = cm.frag_by_breakbond(st.molecules[1],**cutting_site)
        if len(cm_frag) > 1:
            print('Error! More than one fragments obtained by cutting bonds of {:s}\n'
                  'You should check the site specification!'.format(paramrf.basename))
            sys.exit()
        elif len(cm_frag) == 1:
            cm_frag = cm_frag[0]
        else:
            print('Error! no Fagments obtained by cutting bonds of {:s}\n'
                  'You should check the site specification!'.format(paramrf.basename))
            sys.exit()
        if resname:
            parmed_struct.residues[0].name = resname
        for k,v in cutting_site.items():
            atoms = tuple([int(i)-1 for i in v.split('-')])
            ia, oa = atoms
            atoms = self._get_atoms(parmed_struct, [ia,oa])
            bond = self._get_bond(parmed_struct, [oa,ia])
            angles = self._get_angles(parmed_struct, [oa, ia])
            dihedrals = self._get_dihedrals(parmed_struct, [oa, ia])
            inter_dihedrals = self._get_inter_dihedrals(parmed_struct,[oa,ia])
            res['sites'][k]['atoms'] = atoms
            res['sites'][k]['bond'] = bond
            res['sites'][k]['angles'] = angles
            res['sites'][k]['dihedrals'] = dihedrals
            res['sites'][k]['inter_dihedrals'] = inter_dihedrals
        res['cm_frag'] = cm_frag
        res['pmd_struct'] = parmed_struct
        return res

    def connect_res(self,A,B,joinsite=''):
        '''joinsite has the format of
         D1-A1,D2-A2
         where D and A are residue label of A and B, respectively
         1 and 2 are site index on each residue.
         If the joinsite is specified as D-A
         then the program will try to find site starts with D in frag A site starts with A in frag B.

        if joinsite is not specified, then the function will try to find the
        match which has same last letter and different head letter in the site label
        of two frags.
         '''
        def set_bond(ps, bondA, mapA, bondB):
            bond_typeA = bondA.type
            bond_typeB = bondB.type
            req = (bond_typeA.req + bond_typeB.req) / 2
            k = (bond_typeA.k + bond_typeB.k) / 2
            new_bondtype = pmd.BondType(k, req, list=ps.bond_types)
            ps.bond_types.append(new_bondtype)
            idxA1 = mapA[bondA.atom1.idx + 1] - 1
            idxA2 = mapA[bondA.atom2.idx + 1] - 1
            atomA1 = [a for a in ps.atoms if a.idx == idxA1][0]
            atomA2 = [a for a in ps.atoms if a.idx == idxA2][0]
            if abs(bond_typeA.req - bond_typeB.req) > 0.05:
                print('Warning! BOND LENGTH discrepancy {:.3f} vs {:.3f}'
                      .format(bond_typeA.req, bond_typeB.req))
            if abs(bond_typeA.k - bond_typeB.k) > 10:
                print('Warning! BOND FORCE CONSTANT discrepancy {:.3f} vs {:.3f}'
                      .format(bond_typeA.k, bond_typeB.k))
            bond = pmd.Bond(atomA1, atomA2, type=new_bondtype)
            ps.bonds.append(bond)

        def set_angle(ps, angles, mapping):
            for a in angles:
                at = a.type
                new_angletype = pmd.AngleType(at.k, at.theteq, list=ps.angle_types)
                ps.angle_types.append(new_angletype)
                idx1 = mapping[a.atom1.idx + 1] - 1
                idx2 = mapping[a.atom2.idx + 1] - 1
                idx3 = mapping[a.atom3.idx + 1] - 1
                atom1 = [a for a in ps.atoms if a.idx == idx1][0]
                atom2 = [a for a in ps.atoms if a.idx == idx2][0]
                atom3 = [a for a in ps.atoms if a.idx == idx3][0]
                angle = pmd.Angle(atom1, atom2, atom3, type=new_angletype)
                ps.angles.append(angle)

        def set_dihedral(ps, dihedrals, mapping):
            '''set dihedrals with three atom in one frag and one atom in another frag'''
            for d in dihedrals:
                dtype = copy.deepcopy(d.type)
                ps.dihedral_types.append(dtype)
                idx = [d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx]
                idx = [mapping[i + 1] - 1 for i in idx]
                dihedral_set = 0
                for cd in ps.dihedrals:
                    sn = tuple([cd.atom1.idx, cd.atom2.idx, cd.atom3.idx, cd.atom4.idx])
                    if sn == tuple(idx) or sn == tuple(idx[::-1]):
                        cd.type = dtype
                        dihedral_set = 1
                if dihedral_set == 0:
                    # print('Dihedral {:s} not found in pmd structure, add new dihedrals'
                    #       .format('-'.join([str(i+1) for i in idx])))
                    atoms = [ps.atoms[i] for i in idx]
                    dihedral = pmd.Dihedral(*atoms, improper=d.improper, ignore_end=d.ignore_end, type=dtype)
                    ps.dihedrals.append(dihedral)
            ps.dihedral_types.claim()
        # def combine_dihedral(da, db, dihed_idx):
        #     '''combine two dihedrals type and return a dihedrals dict'''
        #     # first generate two dict with period as keys and dihed type as value
        #     dihed_lbl = '-'.join([str(i + 1) for i in dihed_idx])
        #     da_list = defaultdict(list)
        #     db_list = defaultdict(list)
        #     scee = da[0].scee
        #     scnb = da[0].scnb
        #     for d in da:
        #         da_list[d.per].append(d)
        #     for d in db:
        #         db_list[d.per].append(d)
        #     # second remove redundant dihed type in same period and issue warnings
        #     da_dict = {}
        #     db_dict = {}
        #     for k, v in da_list.items():
        #         if len(v) > 1:
        #             print('Warning!!! {:d} dihedrals type correspond to same period {:d} in dihedral {:s} of fragA. ' \
        #                   'I will keep one with largest K'.format(len(v), k, dihed_lbl))
        #             print(v)
        #             d = sorted(v, key=lambda x: x.phi_k, reversed=True)[0]
        #         else:
        #             d = v[0]
        #         da_dict[k] = d
        #     for k, v in db_list.items():
        #         if len(v) > 1:
        #             print('Warning!!! {:d} dihedrals type correspond to same period {:d} in dihedral {:s} of fragB. ' \
        #                   'I will keep one with largest K'.format(len(v), k, dihed_lbl))
        #             print(v)
        #             d = sorted(v, key=lambda x: x.phi_k, reversed=True)[0]
        #         else:
        #             d = v[0]
        #         db_dict[k] = d
        #     combined_dihed = defaultdict(str)
        #     for i in range(1, 10):
        #         ka, kb = [-1000, -1000]
        #         phasea, phaseb = [0, 0]
        #         if i in da_dict:
        #             ka = da_dict[i].phi_k / 2
        #             phasea = da_dict[i].phase
        #         if i in db_dict:
        #             kb = db_dict[i].phi_k / 2
        #             phaseb = db_dict[i].phase
        #         if abs(ka - kb) > 0.1:
        #             print('Warning! There is a DIHEDRAL FORCE CONSTANT DISCREPANCY when connecting two residues:\n'
        #                   '{:.3f} from A vs {:.3f} from B, the period is {:d} and the dihedral is {:s}. I will take the average here.'
        #                   .format(ka, kb, i, dihed_lbl))
        #         if abs(phasea - phaseb) > 0.1:
        #             print('Warning!!! There is a DIHEDRAL PHASE DISCREPANCY when connecting two residues:\n'
        #                   '{:.3f} from A vs {:.3f} from B, the period is {:d} and the dihedral is {:s}. I will use the value of fragment A.'
        #                   .format(phasea, phaseb, i, dihed_lbl))
        #         if ka != -1000 and kb != -1000:
        #             phi_k = (ka + kb)
        #             combined_dihed[i] = (phi_k, phasea, scee, scnb)
        #         elif ka == -1000 and kb != -1000:
        #             combined_dihed[i] = (kb, phasea, scee, scnb)
        #         elif ka != -1000 and kb == -1000:
        #             combined_dihed[i] = (ka, phasea, scee, scnb)
        #     return combined_dihed
        # def set_inter_dihedral(ps, interdA, mapA, interdB, mapB):
        #     ''' set dihedral angle with two atoms on fragA and two atoms on fragB
        #     the mapA and mapB map sn in fragA and fragB to their combined fragC
        #     ps is the fragC
        #     interdA and interdB are dihedral objects from parmed object
        #     '''
        #     dihedral_joined = 0
        #     improper_joined = 0
        #     for da in interdA:
        #         for db in interdB:
        #             idx_da = [da.atom1.idx, da.atom2.idx, da.atom3.idx, da.atom4.idx]
        #             idx_da = [mapA[i + 1] - 1 for i in idx_da]
        #             idx_db = [db.atom1.idx, db.atom2.idx, db.atom3.idx, db.atom4.idx]
        #             idx_db = [mapB[i + 1] - 1 for i in idx_db]
        #             matched = 0
        #             if da.improper == False and db.improper == False:
        #                 if idx_da == idx_db or idx_da == idx_db[::-1]:
        #                     dihedral_joined += 1
        #                     matched = 1
        #             if da.improper == True and db.improper == True:
        #                 if len(set(idx_da) - set(idx_db)) == 0:
        #                     improper_joined += 1
        #                     matched = 1
        #             if matched == 1:
        #                 combine_dihed = combine_dihedral(da.type, db.type, idx_da)
        #                 if len(combine_dihed) > 1:
        #                     new_DT = pmd.DihedralTypeList(list=ps.dihedral_types)
        #                     for per, p in combine_dihed.items():
        #                         DT = pmd.DihedralType(p[0], per, p[1], scee=p[2], scnb=p[3])
        #                         new_DT.append(DT)
        #                     ps.dihedral_types.append(new_DT)
        #                 elif len(combine_dihed) == 1:
        #                     per, p = list(combine_dihed.items())[0]
        #                     new_DT = pmd.DihedralType(p[0], per, p[1], scee=p[2], scnb=p[3], list=ps.dihedral_types)
        #                     ps.dihedral_types.append(new_DT)
        #                 dihedral_set = 0
        #                 for d in ps.dihedrals:
        #                     sn = tuple([d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx])
        #                     if sn == tuple(idx_da) or sn == tuple(idx_da[::-1]):
        #                         d.type = new_DT
        #                         dihedral_set = 1
        #                 if dihedral_set == 0:
        #                     print('Dihedral {:s} not found in pmd structure, add new dihedrals'
        #                           .format('-'.join([str(i+1) for i in idx_da])))
        #                     atom1 = [a for a in ps.atoms if a.idx == idx_da[0]][0]
        #                     atom2 = [a for a in ps.atoms if a.idx == idx_da[1]][0]
        #                     atom3 = [a for a in ps.atoms if a.idx == idx_da[2]][0]
        #                     atom4 = [a for a in ps.atoms if a.idx == idx_da[3]][0]
        #                     dihedral = pmd.Dihedral(atom1, atom2, atom3, atom4, improper=da.improper, type=new_DT)
        #                     ps.dihedrals.append(dihedral)
        #     print('{:d}/{:d} pairs of dihedrals/improper are matched when connecting fragments'
        #           .format(dihedral_joined, improper_joined))
        # first connect coordinates
        cm_Cfrag = cm.connect_frag(A['cm_frag'],B['cm_frag'],joinsite=joinsite)
        # print(A['cm_frag']['mol']['struct'].coord)
        mapA = cm_Cfrag[1].copy() # key is sn of old frag, value is sn of connected frag
        mapB = cm_Cfrag[2].copy()
        # second connect parmed structure
        amber_mask_A = '@' + ','.join([str(i) for i in cm_Cfrag[1].keys()])
        amber_mask_B = '@' + ','.join([str(i) for i in cm_Cfrag[2].keys()])
        parmed_Afrag = A['pmd_struct'][amber_mask_A]
        parmed_Bfrag = B['pmd_struct'][amber_mask_B]
        parmed_Cfrag = parmed_Afrag + parmed_Bfrag
        # third update atom index in parmed structure to make its atom order same to cm fragments
        new_idxA = {i: v[1] for i, v in enumerate(sorted(mapA.items(), key=lambda x: x[0]))}
        # key is idx of joined parmed structure, value is sn of connected frag
        new_idxB = {i + len(new_idxA): v[1] for i, v in enumerate(sorted(mapB.items(), key=lambda x: x[0]))}
        new_idxA.update(new_idxB)
        parmed_Cfrag.atoms.claim()
        parmed_Cfrag.atoms.sort(key=lambda x: new_idxA[x.idx])
        parmed_Cfrag.atoms.claim()
        # forth, set coordinate in parmed structure by cm structure
        sn2coords = nx.get_node_attributes(cm_Cfrag[0]['graph'],'coord')
        #print(sn2coords)
        for a in parmed_Cfrag.atoms:
            a.xx = sn2coords[a.idx+1][0]
            a.xy = sn2coords[a.idx+1][1]
            a.xz = sn2coords[a.idx+1][2]
        # fourth transfer site atom parameter from each parmed fragment to the connected parmed structure
        # print(new_ia_idx,new_ib_idx)
        # print(ia,ib)
        for annihilated_site in cm_Cfrag[3]:
            a,b = annihilated_site
            sa,ia,oa,iia,ooa = a
            sb,ib,ob,iib,oob = b
            ooa2iib = {i:mapB[j] for  i,j in zip(ooa,iib)}
            oob2iia = {i:mapA[j] for  i,j in zip(oob,iia)}
            new_ib = mapB[ib]
            new_ia = mapA[ia]
            mapA.update({oa:new_ib})
            mapA.update(ooa2iib)
            mapB.update({ob:new_ia})
            mapB.update(oob2iia)
            bondA = A['sites'][sa]['bond']
            bondB = B['sites'][sb]['bond']
            set_bond(parmed_Cfrag,bondA,mapA,bondB)
            set_angle(parmed_Cfrag,A['sites'][sa]['angles'],mapA)
            set_angle(parmed_Cfrag,B['sites'][sb]['angles'],mapB)
            set_dihedral(parmed_Cfrag,A['sites'][sa]['dihedrals'],mapA)
            set_dihedral(parmed_Cfrag,B['sites'][sb]['dihedrals'],mapB)
            interdA = A['sites'][sa]['inter_dihedrals']
            interdB = B['sites'][sb]['inter_dihedrals']
            set_dihedral(parmed_Cfrag,interdA,mapA)

            # for d in interdA:
            #     print(d)
            # set_inter_dihedral(parmed_Cfrag,interdA,mapA,interdB,mapB)
            # for d in parmed_Cfrag.dihedrals:
            #     if d.ignore_end == True and d.improper == False:
            #         print(d)
            # parmed_Cfrag.update_dihedral_exclusions()
            # for d in parmed_Cfrag.dihedrals:
                # if d.type.scee > 1.3:
                # print(d.type.scee)
            # for d in parmed_Cfrag.dihedrals:
            #     print(d.type.scee)
        res=defaultdict(str)
        res['sites'] = defaultdict(dict)
        for k, v in cm_Cfrag[0]['site'].items():
            ia, oa = np.array(v)-1
            atoms = self._get_atoms(parmed_Cfrag, [ia, oa])
            bond = self._get_bond(parmed_Cfrag, [oa, ia])
            angles = self._get_angles(parmed_Cfrag, [oa, ia])
            dihedrals = self._get_dihedrals(parmed_Cfrag, [oa, ia])
            inter_dihedrals = self._get_inter_dihedrals(parmed_Cfrag, [oa, ia])
            res['sites'][k]['atoms'] = atoms
            res['sites'][k]['bond'] = bond
            res['sites'][k]['angles'] = angles
            res['sites'][k]['dihedrals'] = dihedrals
            res['sites'][k]['inter_dihedrals'] = inter_dihedrals
        res['cm_frag'] = cm_Cfrag[0]
        res['pmd_struct'] = parmed_Cfrag
        return res

    def compose_mol(self,compose_str):
        base_frag = None
        expand_str = ''
        result = re.split(r'(\[[a-zA-Z]+\]\d*)',compose_str)
        for s in result:
            if '[' in s:
                x = [i for i in re.split(r'[\[\]]',s) if i ]
                ex = ''
                if len(x) == 2:
                    ex = x[0] * int(x[1])
                elif len(x) == 1:
                    ex = x[0]
                else:
                    print('Error!!! Wrong format in build string {:s}'.format(compose_str))
                expand_str += ex
            else:
                expand_str += s
        for res_label in expand_str:
            if res_label not in self.res_lib:
                print(self.res_lib.keys())
                print('Error residue labelled with {:s} not defined'.format(res_label))
                sys.exit()
            else:
                res = self.res_lib[res_label]
            if not base_frag:
                base_frag = res
                old_label = res['label']
            else:
                new_label = res['label']
                base_frag = self.connect_res(base_frag,res)
                old_label = res['label']
        if len(base_frag['sites']) != 0:
            print('There are remaining connection site {:s} on frangment {:s}'
                  .format(','.join(base_frag['sites'].keys()),compose_str))
        else:
            print('{:s} build susscessful'.format(compose_str))
        return base_frag

    def compose_system(self,compose_str):
        pass
