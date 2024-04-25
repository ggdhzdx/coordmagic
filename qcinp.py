#!/usr/bin/env python
"""
Created on Tue Nov  1 14:44:17 2016

@author: Cheng Zhong
"""
""" todo list
1 multiple add_keywords
2 multiple rm_keywords
3 so -P will be overwrite by  -a or -r option
4 read freeze in gau log 
5 support block input 

"""
import itertools
import argparse
import os.path
import sys
import re
import subprocess
import copy
import numpy as np

from collections import OrderedDict
from collections import Counter

parser=argparse.ArgumentParser(description='generate gaussian/orca input file',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-p',dest='program',help='set the program. gau (gaussian) and orca is now available. Default is gau')
parser.add_argument('-k',dest='keywords',help='set the keywords for the input file. Seperate these keywords by space')
parser.add_argument('-a',dest='add_keywords',help='keep the orgin keywords and add addtional keywords.\n'\
                    'use | to generate a set of input file with different addition keywords (for benchmark).\n'\
                    'e.g. -a "b3lyp|pbe1pbe|m062x"',default='')
parser.add_argument('-r',dest='rm_keywords',help='remove the keywords from orgin keywords, sperate multiple keywords by space.')
parser.add_argument('-b',dest='basis_set',help='input two basis set, seperate by comma, for light and heavy atoms, respectively. e.g. 6-31g(d),lanl2dz')
parser.add_argument('-A',dest='app_name',help='append name to the file name.')
parser.add_argument('-T',dest='trim_name',help='remove append name from the file name.')
parser.add_argument('-n',dest='nproc',help='set the number of processes of the input gjf file. Default: read from file or 24')
parser.add_argument('-m',dest='mem',help='set the memory for the input gjf file, unit default as GB. Defaul: read from file or 16GB')
parser.add_argument('-B',dest='block',help='set the block input for orca. e.g. "%%tddft;nroots 5;DoNTO true;NTOStates 1,2;TRIPLETS TRUE;end"',default="")
parser.add_argument('-P',dest='profile',help='store the common keywords in profile, avail options are:\n'
                    'opt : use keywords "opt pbe1pbe def2svp emp=gd3bj g09default iop(5/13=1)"\n'
                    'ts : read log file and add "opt(ts,calcall,noeigentest) freq"\n'
                    'freq : read log file, remove opt and add freq\n'
                    'irc : read log file, remove "opt freq", add "irc(calcall,gs2,maxpoints=30)"\n'
                    'uv : read log file, remove "opt freq", add "td(singlets,nstates=30)"\n'
                    'fluo: read log file, add "opt td"\n'
                    'resp : use keywords "opt b3lyp def2SVP scrf=pcm emp=gd3bj pop=mk iop(6/33=2) iop(6/42=6)"\n'
                    'soc : read log file, remove opt, add "td(50-50,nstates=10) 6D 10F nosymm GFInput" and write rwf\n'
                    'st : step 1 "opt pbe1pbe def2svp em(gd3bj) g09default:\n'
                    '     step 2 "geom=allcheck guess=read pbe1pbe def2svp td(50-50,nstates=10) 6D 10F nosymm GFInput\n'
                    'otddh: tddft calculation for orca using SCS-PBE-QIDH\n'
                    'nac : read log file, remove "opt freq" add "td prop(fitcharge,field) iop(6/22=-4,6/29=1,6/30=0,6/17=2) nosymm"\n')
parser.add_argument('-f',dest='file',help='use keywords form other file')
parser.add_argument('-F',dest='fragment',help='define fragment, available options are:\n'
                    'check : check the molecules in current geometry\n'
                    'auto : each mol is a frag, range from large to small\n'
                    'autotype : each mol type is a frag, range from large to small\n'
                    '1,3-7;9-12 : atom serial separated by semicolon, the last ";" means use remain atoms for last fragment \n'
                    'L3-L1; : first frag is last 3 to last 1 atom, second frag is the remaining atoms \n'
                    'm1-3;m2,4 : mol index separated by semicolon\n'
                    't1,2;t3-4 : mol type index separated by semicolon\n', default='')
parser.add_argument('--fragset',dest='fragset',help='define distance threshold to separate molecules\n'
                    'the defaults are: cov_scale:1.1;cov_delta:0;usrcovr:"";usrcovth:""\n'
                    'the threshold for determining bond is defined as:\n'
                    '(covalent_radii_atom1 + covalent radii_atom2) * cov_scale + cov_delta\n'
                    'usrcovr can define covalent radii for the specific element.\n'
                    'The format is like "C=3.5,H=1.2\n'
                    'usrcovth can define specific covalent distance threshold for specific element pairs\n'
                    'The format is like "C-H=1.0,C-S=1.7"\n')
parser.add_argument('-c',dest='charge_spin',help='set the charge and spin for the system. Seperated by space. e.g. \"-1 2\". Default: read from file. If not exit, set to 0,1',default='')
parser.add_argument('--readchk',help='By default, using this option will use guess=read keywords if chk file has been found.',action='store_true')
parser.add_argument('--irc', help='Generate reactant and product input file based on irc output file.\n'
                    'example: --irc p1-5 means product is the formation of bond between atom\n'
                    '1 and atom 5. --irc r1,5 means reactant is the bond breaking between 3 and 4\n',
                    type=str, default='')
parser.add_argument('--addinfo',help='additional information that append in Gaussian input, seperate multiple line by ;')
parser.add_argument('--steps', help='set keywords for multiple step job. Seperate multiple step keywords by ;', type=str, default='')
parser.add_argument('--rwf', help='write rwf file with same name as chk', action='store_true')
parser.add_argument('--freeze', help='set no to unfreeze all atoms')
parser.add_argument('--element_only',help='By default, addition infomation that follows element (such as fragment info) will be included.\n'
                    'Use this option to prohibit this action.',action='store_true')
parser.add_argument('--kwtype',help='set the user input kwtype. gaussian and orca is now available. Input first three letters is enough. Default depends first on file type and then program type')
parser.add_argument('--noextra',help='By default, the extra info in gjf file will be kept. Use this option to discard these extra information.\n'
                                'along with geom=connectivity and modredundant in keywords',action='store_true')
parser.add_argument('inputfile',nargs='+',help='the input structures can be gjf, mol, xyz, Gaussian log,\n'
                    'cdx(one cdx should contain only 1 structure, and latest version of obabel should be in PATH)')

class Fragment:
    def __init__(self,coord,fragset=''):
        import coordmagic as cm
        from coordmagic.atomorder import snr2l, snl2r
        self.snr2l = snr2l
        self.snl2r = snl2r
        st = cm.conver_structure(coord,obj_type='xyz',islist=True)
        if fragset:
            fragset_dict = {i.split(':')[0]:i.split(':')[1] for i in fragset.split(';')}
            st.graph.set_threshold(**fragset_dict)
        st.graph.gen_mol(silent=True)
        self.frag2idx = {}
        self.st = st
        self.df = self.st.mol_df[["id","type_id","formula",'sn_range']]
    def parser_str(self,frag_str):
        print(self.df.to_string(index=False))
        mol2idx = self.df[["id",'sn_range']].set_index('id').to_dict()['sn_range']
        mtype2idx = self.df.groupby(['type_id']).agg({'sn_range': lambda x: ','.join(list(x))}).to_dict()['sn_range']
        if frag_str == 'auto':
            self.frag2idx = mol2idx
        elif frag_str == 'autotype':
            self.frag2idx = mtype2idx
        elif frag_str != 'check':
            for i,s in enumerate(frag_str.split(';')):
                if s.startswith('m'):
                    ids = ','.join([mol2idx[i] for i in self.snr2l(s.strip('m'))])
                    self.frag2idx[i+1] = ids
                elif s.startswith('t'):
                    ids = ','.join([mtype2idx[i] for i in self.snr2l(s.strip('t'))])
                    self.frag2idx[i+1] = ids
                else:
                    if s != '':
                        self.frag2idx[i+1] = self.snl2r(self.snr2l(s,total=len(self.st.atoms)))
                    else:
                        remain = self.snr2l(','.join(self.frag2idx.values()),total=len(self.st.atoms),complement=True)
                        s = self.snl2r(remain)
                        self.frag2idx[i+1] = s
        if frag_str != 'check':
            print("Fragments:")
            print(str(self.frag2idx).replace(', ', '\n').replace(': ', ':\t').replace('{', '').replace('}', ''))
            self.frag2idx_list = {k:self.snr2l(v,total=len(self.st.atoms)) for k,v in self.frag2idx.items()}

class basisSet:
    '''analysis basis set'''
    def __init__(self,kw_str,kwtype):
        self.gau_pop_basis = ['sto-3g','3-21','4-31','6-21','6-311']
        self.gau_cc_basis = ['cc-pvdz','cc-pvtz','cc-pvqz','cc-pv5z','cc-pv6z']
        self.gau_aldrich_basis = ['sv','svp','tzv','tzvp','qzvp']
        self.gau_pseudo_basis = ['gen','genecp']
        self.kwtype=kwtype
        self.kw_str=kw_str
        self.basis='unknown'
        self.basis_class='unknown'
        self._perceive_basis()

    def _perceive_basis(self):
        if self.kwtype.lower().startswith('gau'):
            for k in self.kw_str.split():
                if any(b in k for b in self.gau_pop_basis):
                    self.basis = k
                    self.basis_class='pople'
                if any(b in k for b in self.gau_cc_basis):
                    self.basis = k
                    self.basis_class='cc'
                if any(b in k for b in self.gau_aldrich_basis):
                    self.basis = k
                    self.basis_class='aldrich'
                if any(b in k for b in self.gau_pseudo_basis):
                    self.basis = k
                    self.basis_class='pseudo'

    def convert2(self,program):
        if program.startswith('dal'):
            if self.basis_class == 'pople':
                self.basis=self.basis.replace('(d)','*')
                self.basis=self.basis.replace('(d,p)','**')
            if self.basis_class == 'pseudo':
                self.basis=self.basis = 'pseudo'
            if self.basis_class == 'aldrich':
                self.basis=self.basis.replace('f2s','f2_s')
                self.basis=self.basis.replace('f2t','f2_t')
                self.basis=self.basis.replace('f2q','f2_q')
        return  self.basis

class Keyword:
    def __init__(self,keywords_string,kwtype='gau'):
        self.keywords_str=keywords_string
        self.inp_kwtype=kwtype
        self.kw_dict=self.kw_str2dict(keywords_string)

    def kw_str2dict(self,keywords):
        kw_dict=OrderedDict()
        if self.inp_kwtype.startswith('gau'):
            kw_list=re.split(r"\s+",keywords)
            for i in kw_list:
                if i.lower().startswith('iop') or i.startswith('6-31') or i.startswith('3-21') or '(p)' in i:
                    head_i=i
                    tail_i=''
                else:
                    head_i = re.split(r'=|\(',i)[0]
                    tail_i = re.split(r'\(|,|\)',i[len(head_i):].strip('='))
                tail_dict=OrderedDict()
                for s in tail_i:
                    if '=' in s:
                        s_k,s_v=s.split('=')
                        tail_dict[s_k.lower()]=s_v.lower()
                    elif s:
                        tail_dict[s.lower()]=''
                kw_dict[head_i]=tail_dict
        return kw_dict

    def kw_dict2str(self):
        keywords_list=[]
        for k,v in self.kw_dict.items():
            param_list=[]
            for k1,v1 in v.items():
                if v1:
                    param_list.append(k1+'='+v1)
                elif k1:
                    param_list.append(k1)
            param_str=','.join(param_list)
            if param_str:
                keywords_list.append(k+'('+param_str+')')
            else:
                keywords_list.append(k)
        return ' '.join(keywords_list)

    def kw_dict2block(self,kw_dict):
        """kw_dict is a dict of dict"""
        if not kw_dict:
            return ''
        else:
            block_str=''
            for k,v in kw_dict.items():
                block_str+='%'+k
                for k1,v1 in v.items():
                    block_str+='\t'+k1+'\t'+v1+'\n'
                block_str+='end\n'
            return block_str

    def add_kw(self,add_dict):
        for k,v in add_dict.items():
            if k in self.kw_dict:
                self.kw_dict[k].update(v)
            elif k:
                self.kw_dict[k]=v
    def rm_kw(self,rm_dict):
        for key,v_dict in rm_dict.items():
            if key in self.kw_dict:
                if len(v_dict) != 0:
                    new_dict={k:v for k,v in self.kw_dict[key].items() if k not
                            in v_dict}
                    self.kw_dict[key]=new_dict
                else:
                    self.kw_dict.pop(key,None)

    def gen_keywords(self,output_kw_type):
        if output_kw_type.startswith('gau'):
            return self._gen_gau_keywords()
        if output_kw_type.startswith('orca'):
            return self._gen_orca_keywords()

    def _gen_gau_keywords(self):
        return self.kw_dict2str()

    def _gen_orca_keywords(self):
        block_kw={}
        for k,v in self.kw_dict.items():
            if k.lower()=='scrf':
                if 'solvent' in v.keys():
                    sol=v['solvent']
                else:
                    sol='water'
                if 'smd' in  v.keys():
                    block_kw['cpcm']={'smd':'true','solvent':'"'+sol+'"'}
                    sol_mod='cpcm'
                else:
                    sol_mod='cpcm'
                self.kw_dict[sol_mod]={sol:""}
                del self.kw_dict[k]
            if k.lower().startswith('td'):
                if 'nstates' in v.keys():
                    ns=v['nstates']
                else:
                    ns="3"
                block_kw["tddft"]={"NROOTS":str(ns)}
                if "50-50" in v.keys() or 'triplets' in v.keys():
                    block_kw["tddft"]['TRIPLETS'] = "TRUE"
                    block_kw["tddft"]['DOSOC'] = "TRUE"
                if "root" in v.keys():
                    block_kw["tddft"]['IRoot'] = v['root']
                    block_kw["tddft"]['FollowIRoot'] = "TRUE"
                if "nac" in v.keys():
                    block_kw["tddft"]['NACME'] = "TRUE"
                    block_kw["tddft"]['ETF'] = "TRUE"
        kw_str=self.kw_dict2str()+'\n'+self.kw_dict2block(block_kw)
        return kw_str




class GeomAna:
    '''
    extract pure coords via gen_coord function and extract other relevant information
    '''
    def __init__(self,coords):
        self.coords=coords
        self.ele2num = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
                    'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
                    'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
                    'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
                    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
                    'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
                    'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
                    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
                    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62,
                    'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
                    'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
                    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'U': 92,'Bq': 0,
                    }
        self.num2ele = dict((value,key) for key,value in self.ele2num.items())
        self._ana_coord()
        self._ana_ele()

    def _ana_coord(self):
        self.elem_num = []
        self.elem = []
        self.xyz_coord = []
        self.gau_add_info = []
        for s in self.coords:
            e = s.split()[0].split('(')[0]
            try:
                a = re.search(r'\((.*?)\)',s).group(1)
            except AttributeError:
                a = ''
            s = re.sub(r'\((.*?)\)','',s)
            c = [i for i in map(float,s.split()[1:4])]
            try:
                ele_num = int(e)
            except ValueError:
                ele_num = self.ele2num[e]
            self.elem_num.append(ele_num)
            self.elem.append(self.num2ele[ele_num])
            self.xyz_coord.append(c)
            self.gau_add_info.append(a)

    def _ana_ele(self):
        self.l_atom = []
        self.h_atom = []
        for a in set(self.elem):
            if self.ele2num[a] <= 18:
                self.l_atom.append(a)
            else:
                self.h_atom.append(a)

    def compute_formula(self):
        ele_counter = Counter(self.elem)
        self.e_n=sorted([(k,str(v)) for k,v in ele_counter.items()],key = lambda x:x[0])
        total_electron=0
        for en in self.e_n:
            ele_num=self.ele2num[en[0]]*int(en[1])
            total_electron += ele_num
        print('{:s} found. {:d} electrons in total'
              .format(''.join([i for c in self.e_n for i in c]),total_electron ))

    def gen_coord(self):
        coords=[]
        for i,e in enumerate(self.elem):
            c = '{:10s}{:12.6f}{:12.6f}{:12.6f}'.format(e,*self.xyz_coord[i])
            coords.append(c)
        return coords



class inputParam:
    def __init__(self,args,inputfile):
        self.ele2num = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,
        'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
        'K':19,'Ca':20,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,
        'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Pd':46,'Ir':77,
        'Pt':78,'Ag':47,'Au':79,'Cd':48,'Hg':80,'I':53,'Pb':82,'Ce':58,'Hf':72}
        self.num2ele = dict((value,key) for key,value in self.ele2num.items())
        self.coords=[]
        self.frags = {}
        self.freeze_flag=[]
        self.connect_info = []
        self.gjf_add_info = []
        self.new_chk = 0 # determine if need to generate new chk file
        self.read_chk = 0 # determine wether need to read chkfile
        self.args=args
        self.multi_file_gen=0
        self.filename=inputfile
        self.init_value = {'program':'gaussian','nproc':'32','mem':'100GB',
                      'keywords':'','add_keywords':'','rm_keywords':'','kwtype':'gau',
                      'basis_set':'','charge_spin':'0 1',
                      'app_name':'','trim_name':'',
                      'readchk':False,'noextra':False,'element_only':False,'rwf':False,
                       'addinfo':'', 'steps':'','irc':''}
        self._init_attr()
        self._init_name()
        self.update_param(irc=self.args.irc)

    def _apply_profile(self):
        self.profile={'opt':{'keywords':'opt pbe1pbe def2svp emp=gd3bj g09default scf(maxcycle=64) iop(5/13=1)'},
                      'ts':{'add_keywords':'opt(ts,calcall,noeigentest) freq'},
                      'freq':{'rm_keywords':'opt','add_keywords':'freq'},
                      'irc':{'rm_keywords':'opt freq','add_keywords':'irc(calcall,lqa,maxpoints=50)'},
                      'uv':{'rm_keywords':'opt freq','add_keywords':'td(singlets,nstates=30)'},
                      'fluo':{'add_keywords':'opt td'},
                      'resp':{'keywords':'opt b3lyp def2SVP scrf=pcm emp=gd3bj pop=mk iop(6/33=2) iop(6/42=6)','program':'gaussian'},
                      'soc':{'rm_keywords':'opt freq','add_keywords':'td(50-50,nstates=10) 6D 10F nosymm GFInput','rwf':True,'program':'gaussian'},
                      'st':{'keywords':'opt pbe1pbe def2svp em(gd3bj) g09default',
                            'steps':'geom=allcheck guess=read pbe1pbe def2svp td(50-50,nstates=10) 6D 10F nosymm GFInput',
                            'rwf':True,'program':'gaussian'},
                      'nac':{'add_keywords':'td prop(fitcharge,field) iop(6/22=-4,6/29=1,6/30=0,6/17=2) nosymm',
                             'rm_keywords':'opt freq','program':'gaussian'},
                      'otddh':{'keywords':"RIJCOSX RI-SCS-PBE-QIDH def2-SVP def2/J def2-SVP/C TIGHTSCF",
                               'block':"%tddft;dcorr 1;nroots 5;triplets true;DoNTO true;NTOStates 1,2,3;NTOThresh 1e-4;tda true;printlevel 3;end",
                               'program':"orca"}
                      }
        if self.args.profile:
            self.update_param_by_args(arg_dict=self.profile[self.args.profile])


    def _init_attr(self):
        for k,v in self.init_value.items():
            setattr(self,k,v)

    def update_param(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

    def update_param_by_args(self,arg_dict=None):
        if arg_dict is None:
            arg_dict=self.args.__dict__
        for k,v in arg_dict.items():
            if v:
                setattr(self,k,v)

    def _init_name(self):
        self.base_name = os.path.splitext(self.filename)[0]
        self.origin_name = self.base_name
        self.suffix = os.path.splitext(self.filename)[1]

    def modify_name(self):
        # modify file name
        if self.trim_name in self.base_name and self.trim_name:
            self.base_name = self.base_name.split(self.trim_name)[0]
        if self.app_name:
            self.base_name = self.base_name + self.app_name

    def check_avail_file(self):
        if  self.readchk:
            if os.path.isfile(self.origin_name+'.chk'):
                self.read_chk = 1
        if self.read_chk == 1 and self.origin_name != self.base_name:
                self.new_chk = 1

    def process_geom(self):
        self.geom=GeomAna(self.coords)
        self.geom.compute_formula()
        self.all_elem=self.geom.elem
        self.xyz_coord= self.geom.xyz_coord
        self.pure_coords = self.geom.gen_coord()
        if self.element_only or not self.suffix.endswith('gjf'):
            self.coords = self.pure_coords

    def generate_fragment(self):
        basenames = [self.base_name]
        coord_list = [self.coords]
        if self.args.fragment != '':
            frag = Fragment(self.pure_coords,fragset=self.args.fragset)
            frag.parser_str(self.args.fragment)
            if self.args.fragment != 'check':
                basenames = []
                coord_list = []
                for f,idx in frag.frag2idx_list.items():
                    basenames.append('{:s}_{:02d}'.format(self.base_name,f))
                    coord_list.append([self.coords[i-1] for i in idx])
        self.basenames = basenames
        self.coord_list = coord_list

    def gen_atomic_basis(self):
        if self.basis_set:
            if len(self.basis_set.split(','))==2:
                lb,hb=self.basis_set.split(',')
                self.atomic_basis={lb:self.geom.l_atom,hb:self.geom.h_atom}
                self.ecp_basis={hb:self.geom.h_atom}
            else:
                self.atomic_basis={self.basis_set:self.geom.elem}
                self.ecp_basis=''
        else:
            self.atomic_basis=''
            self.ecp_basis=''

    def _prep_gau_kw(self):
        if self.connect_info and not self.noextra:
            self.add_keywords += ' geom=connectivity'
        else:
            self.rm_keywords += ' geom'
        if self.noextra:
            self.rm_keywords += ' opt=modredundant'
        #if not self.readchk:
        #    self.rm_keywords += ' guess'
        #if self.read_chk == 1:
        #    self.add_keywords += ' guess=read'
        if self.irc:
            self.rm_keywords += ' irc'
            self.add_keywords += ' opt freq'
        self.kw_obj.rm_kw(self.kw_obj.kw_str2dict(self.rm_keywords))
        self.kw_obj.add_kw(self.kw_obj.kw_str2dict(self.add_keywords))
        self.keywords=self.kw_obj.gen_keywords(self.program)
        if not self.keywords.strip().startswith('#'):
            self.keywords='#p ' + self.keywords

    def _prep_dal_kw(self):
        self.basis_set=self.bs_obj.convert2('dal')

    def _prep_orca_kw(self):
        self.keywords=self.kw_obj.gen_keywords(self.program)
        if not self.keywords.strip().startswith('!'):
            self.keywords='! ' + self.keywords
        if self.block:
            self.block = self.block.split(";")
        else:
            self.block=[]


    def prep4out(self):
        self._apply_profile()
        self.update_param_by_args()
        self.modify_name()
        self.check_avail_file()
        self.process_geom()
        self.generate_fragment()
        self.gen_atomic_basis()
        self.bs_obj=basisSet(self.keywords,kwtype=self.kwtype)
        self.kw_obj=Keyword(self.keywords,kwtype=self.kwtype)
        if self.program.startswith('gau'):
            self._prep_gau_kw()
        if self.program.startswith('orc'):
            self._prep_orca_kw()
        if self.program.startswith('dal'):
            self._prep_dal_kw()



    def read_input_file(self):
        if self.suffix=='.gjf' or self.suffix=='.com':
            self._read_gauinp()
        if self.suffix=='.log' or self.suffix=='.out':
            self._read_gauout()
        if self.suffix=='.cdx':
            self._read_cdx()
        if self.suffix=='.xyz':
            self._read_xyz()
        if self.suffix=='.sdf':
            self._read_sdf()

        #idnetify all the elements in the structure
    def _read_gauout(self):
        with open(self.filename,'r') as inp:
            read_coord = 2
            read_fix = 0
            kw_flag = 3
            read_init = 0
            all_coords = []
            freeze_flag = []
            fix_info = []
            for l in inp:
                if 'Multiplicity =' in l and 'Charge =' in l:
                    read_init = 1
                    charge = l.strip().split()[2]
                    spin = l.strip().split()[5]
                    charge_spin = [charge,spin]
                    self.update_param(charge_spin=' '.join(charge_spin))
                    continue
                if 'The following ModRedundant input section has been read' in l:
                    read_fix = 1
                    continue
                if read_fix ==1:
                    if re.match(r'^\s*$',l):
                        read_fix = 0
                    else:
                        fix_info.append(l)
                if read_init == 1 and re.match(r'^\s*$',l):
                    read_init = 0
                if read_init == 1 and len(l.strip().split()) > 3:
                    freeze_flag.append(l.split()[1])
                if l.startswith(' %nproc'):
                    nproc = l.strip().split('=')[1]
                    self.update_param(nproc=nproc)
                if l.startswith(' %mem'):
                    mem = l.strip().split('=')[1]
                    self.update_param(mem=mem)
                if '----------' in l and kw_flag == 3:
                    kw_flag = 2
                elif '----------' in l and kw_flag == 1:
                    kw_flag = 0
                if l.startswith(' #') and kw_flag == 2:
                    kw_flag = 1
                    kw = l[1:].rstrip("\n")
                    continue
                if kw_flag == 1:
                    kw = kw + l[1:].rstrip("\n")
                if 'Standard orientation' in l or 'Input orientation' in l:
                    coords = []
                    read_coord = -1
                if read_coord == -1 and '----------' in l:
                    read_coord = 0
                    continue
                if read_coord == 0 and '----------' in l:
                    read_coord = 1
                    continue
                if read_coord == 1 and '----------' in l:
                    all_coords.append(coords)
                    read_coord = 2
                    continue
                if read_coord == 1:
                    atom = l.split()[1]
                    xyz = l.split()[3:]
                    coords.append('{:<15s}{:>14s}{:>14s}{:>14s}'.format(atom,*xyz))
            if all([i in ['0','-1','-2','-3'] for i in freeze_flag]):
                self.freeze_flag = freeze_flag
            else:
                self.freeze_flag = []
            self.update_param(keywords = kw.strip())
            self.coords=self._choose_coords(all_coords)
            self.kw_type = 'gau'
            for l in fix_info:
                self.gjf_add_info.append(re.sub(r'(?<=\d)\s+(?=\d)', ',', l).strip())
            print('keywords read: \"{:s}\"'.format(kw))

    def _choose_coords(self,allcoords):
        def dist(coords,a1,a2):
            """return distance between a1 and a2 in coords"""
            c1=[float(i) for i in coords[a1-1].split()[1:]]
            c2=[float(i) for i in coords[a2-1].split()[1:]]
            return np.linalg.norm(np.array(c1)-np.array(c2))
        if self.irc.upper().startswith('P'):
            self.app_name = self.app_name + '_prod'
        elif self.irc.upper().startswith('R'):
            self.app_name = self.app_name + '_react'
        else:
            coords=allcoords[-1]
        if '-' in self.irc:
            a1,a2 = [int(i) for i in self.irc[1:].split('-')]
            coords = sorted(allcoords, key=lambda c: dist(c,a1,a2))[0]
        if ',' in self.irc:
            a1,a2 = [int(i) for i in self.irc[1:].split(',')]
            coords = sorted(allcoords, key=lambda c: dist(c,a1,a2))[-1]
        return coords

    def _read_gauinp(self):
        cs_read = 0
        file_end = 0
        kw_flag = 0
        freeze_flag = []
        with open(self.filename,'r') as inp:
            coords=[]
            for l in inp:
                if re.match(r'^\s*-?\d\s+\d\s?',l) and cs_read == 0:
                    self.charge_spin = l.strip()
                    cs_read = 1
                    kw_flag = 0
                    continue
                if re.match(r'^\s*%nproc',l):
                    self.update_param(nproc=l.strip().split('=')[1])
                if re.match(r'^\s*%mem',l):
                    self.update_param(mem=l.strip().split('=')[1])
                if re.match(r'^\s*#',l):
                    keywords = l.strip()
                    kw_flag=1
                    continue
                if kw_flag == 1 and not re.match(r'^\s*$',l):
                    keywords = keywords +" "+ l.strip()
                else:
                    kw_flag = 0
                if cs_read == 1 and re.match(r'^\s*$',l):
                    cs_read = 2
                    continue
                if cs_read == 1:
                    if len(l.strip().split()) > 4:
                        freeze_flag.append(l.split()[1])
                        coords.append("  ".join([l.split()[0]]+l.split()[2:]).strip())
                    elif len(l.strip().split()) == 4:
                        coords.append(l.strip())
                if cs_read == 2:
                    if file_end == 1 and re.match(r'^\s*$',l):
                        file_end = 2
                        break
                    if re.match(r'^\s*$',l):
                        file_end = 1
                    else:
                        file_end = 0
                    if file_end < 2:
                        self.gjf_add_info.append(l.strip())
            #extract connectivity info
            self.connect_info=[]
            checker = 0
            endpt=-1
            for i,l in enumerate(self.gjf_add_info):
                if l:
                    h = l.strip().split()[0]
                else:
                    h = ''
                if h == '1':
                    self.connect_info = []
                    checker = 1
                    self.connect_info.append(l)
                    checker += 1
                    continue
                elif h == str(checker):
                    self.connect_info.append(l)
                    checker += 1
                    continue
                elif len(self.connect_info)>2 :
                    endpt = i
                    break
            self.gjf_add_info = self.gjf_add_info[endpt+1:]
            if len([x for x in self.gjf_add_info if x ]) == 0:
                self.gjf_add_info = []
            self.kw_type = 'gau'
            self.update_param(keywords = keywords.strip())
            self.coords=coords
            if all([i in ['0','-1','-2','-3'] for i in freeze_flag]) and self.args.freeze != "no":
                self.freeze_flag = freeze_flag
            else:
                self.freeze_flag = []
            print('keywords read: \"{:s}\"'.format(keywords))

    def _read_cdx(self):
        #first to check if obabel exist
        no_babel=0
        try:
            pbabel=subprocess.Popen(['obabel',self.filename,'-O','obabel_temp.xyz','--gen3d','--conformer','--fast','--score','energy','--ff','GAFF'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        except OSError:
            print('Error! no obabel command found. Plead install openbabel or add the obabel command to your PATH environment variables. cdx file will not be converted.')
            no_babel=1
        out,err=pbabel.communicate()
        out=out.decode(encoding='UTF-8')
        if 'Could not setup force field.' in out:
            print('MMFF failed, try GAFF')
            pbabel=subprocess.Popen(['obabel',self.filename,'-O','obabel_temp.xyz','--gen3d','--conformer','--fast','--score','energy','--ff','GAFF'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            out,err=pbabel.communicate()
        if 'Could not setup force field.' in out:
            print('GAFF failed, try UFF')
            pbabel=subprocess.Popen(['obabel',self.filename,'-O','obabel_temp.xyz','--gen3d','--conformer','--fast','--score','energy','--ff','UFF'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            out,err=pbabel.communicate()
        if 'Could not setup force field.' in out:
            print('UFF failed, can not convert cdx')
            no_babel=1
        if no_babel == 0:
            try:
                n_mol = int(out[0].split()[0])
                if n_mol > 1:
                    print('Warning! {:d} molecules converted for {:s}. Only first molecule will be read.'.format(n_mol,self.filename))
            except:
                pass
            # read temp.xyz
            coords=[]
            xyz = open('obabel_temp.xyz','r')
            read_flag = 0
            for l in xyz:
                if len(l.split()) == 4:
                    coords.append(l.strip())
                    read_flag = 1
                if re.match(r'^\d+$',l) and read_flag == 1 :
                    break
            self.coords=coords

    def _read_xyz(self):
        coords=[]
        xyz = open(self.filename,'r')
        read_flag = 0
        nol=0
        for l in xyz:
            nol+=1
            if len(l.split()) > 3 and nol > 2:
                coords.append(l.strip())
                read_flag = 1
            if re.match(r'^\d+$',l) and read_flag == 1 :
                break
        self.coords=coords

    def _read_sdf(self):
        coords=[]
        xyz = open(self.filename,'r')
        read_flag = 0
        nol=0
        for l in xyz:
            nol+=1
            if len(l.split()) > 10 and nol > 4:
                ele=l.strip().split()[3]
                coord = l.strip().split()[:3]
                coords.append("{:s} {:s} {:s} {:s} ".format(ele,*coord))
                read_flag = 1
            if len(l.split()) < 10 and read_flag == 1 :
                break
        self.coords=coords

class writeOut:
    def __init__(self,param):
        self.param=param

    def write_qcinput(self):
        for i,bn in enumerate(self.param.basenames):
            if self.param.program.lower().startswith('gau'):
                self._write_gaussian(basename=bn,coord=self.param.coord_list[i])
            elif self.param.program.lower().startswith('orc'):
                self._write_orca(basename=bn,coord=self.param.coord_list[i])
            elif self.param.program.lower().startswith('dal'):
                self._write_dalton(basename=bn,coord=self.param.coord_list[i])
            else:
                print('-p option {:s} not recognized'.format(self.param.program))

    def _write_gaussian(self,basename='',coord=None):
        if basename is None:
            basename = self.param.base_name
        flnm = basename + '.gjf'
        print('{:s}: keywords used: \"{:s}\"'.format(flnm,self.param.keywords))
        with open(flnm,'w') as gjf:
            if self.param.new_chk == 0 or self.param.read_chk == 1:
                gjf.write('%chk='+basename+'.chk\n')
            if self.param.new_chk == 1 and self.param.read_chk == 1:
                gjf.write('%oldchk='+self.param.origin_name+'.chk\n')
            if self.param.rwf:
                gjf.write('%rwf='+basename+'.rwf\n')
            gjf.write('%nproc='+self.param.nproc+'\n')
            gjf.write('%mem='+self.param.mem+'\n')
            gjf.write(self.param.keywords+'\n')
            gjf.write('\n')
            gjf.write(basename+'\n')
            gjf.write('\n')
            gjf.write(self.param.charge_spin+'\n')
            if coord is None:
                coord = self.param.coords
            if len(self.param.freeze_flag) ==  len(coord) and self.param.args.freeze != "no":
                for i,c in enumerate(coord):
                    gjf.write(c.split()[0]+'  '+self.param.freeze_flag[i] + '  ' + '  '.join(c.split()[1:])+'\n')
            else:
                for i,c in enumerate(coord):
                    gjf.write(c+'\n')
            gjf.write('\n')
            if not self.param.noextra:
                if self.param.connect_info:
                    for i in self.param.connect_info:
                        gjf.write(i+'\n')
                    gjf.write('\n')
                for i in self.param.gjf_add_info:
                    gjf.write(i+'\n')
                for i in self.param.addinfo.split(';'):
                    if i:
                        gjf.write(i+'\n')
            #write basis set
            if self.param.atomic_basis:
                for k,v in self.param.atomic_basis.items():
                    if ' '.join(v)+' 0' not in self.param.gjf_add_info and v:
                        gjf.write(' '.join(v)+' 0\n')
                        gjf.write(k+'\n')
                        gjf.write('****\n')
            if self.param.ecp_basis:
                gjf.write('\n')
                for k,v in self.param.ecp_basis.items():
                    if ' '.join(v)+' 0' not in self.param.gjf_add_info and v:
                        gjf.write(' '.join(v)+' 0\n')
                        gjf.write(k+'\n')
            if self.param.steps:
                for i,com in enumerate(self.param.steps.split(';')):
                    gjf.write('--Link{:d}--\n'.format(i+1))
                    gjf.write('%rwf='+basename+'.rwf\n')
                    gjf.write('%chk='+basename+'.chk\n')
                    gjf.write('%nproc='+self.param.nproc+'\n')
                    gjf.write('%mem='+self.param.mem+'\n')
                    if not com.startswith('#'):
                        com = '#P ' + com
                    gjf.write(com+'\n')
            gjf.write('\n')
            gjf.write('\n')

    def _write_xyz(self,basename=None,coord=None):
        if basename is None:
            basename = self.param.base_name
        if coord is None:
            coord = self.param.coords
        xyz = open(basename+'.xyz','w')

        xyz.write('{:d}\n'.format(len(coord)))
        xyz.write('{:s}\n'.format(basename))
        for c in self.param.coords:
            xyz.write(c+'\n')
        xyz.close()


    def _write_orca(self,basename=None,coord=None):
        if basename is None:
            basename = self.param.base_name
        if coord is None:
            coord = self.param.coords
        flnm=basename + '.inp'
        inp = open(flnm,'w')
        print('{:s}: keywords used: \"{:s}\"'.format(flnm,self.param.keywords.strip()))
        inp.write(self.param.keywords)
        for line in self.param.block:
            inp.write(line+'\n')
        if self.param.mem.lower().endswith('gb'):
            mem=int(self.param.mem[:-2])*1024
        elif self.param.mem.lower().endswith('mw'):
            mem=int(self.param.mem[:-2])*8
        else:
            print('Error! memory format {:s} not recognized'.format(self.param.mem))
        mem_per_core = int(mem/int(self.param.nproc))
        inp.write('%pal nprocs {:s}\n'.format(self.param.nproc))
        inp.write('     end\n')
        inp.write('%MaxCore {:d}\n'.format(mem_per_core))
        if 'opt' in self.param.keywords:
            xyzfile=basename+'.xyz'
            inp.write('* xyzfile {:s} {:s}\n'.format(self.param.charge_spin,xyzfile))
            self._write_xyz(basename=basename,coord=coord)
        else:
            inp.write('* xyz {:s}\n'.format(self.param.charge_spin))
            for c in coord:
                inp.write(c+'\n')
            inp.write('*\n')

    def _write_dalton(self,basename=None,coord=None):
        if basename is None:
            basename = self.param.base_name
        if coord is None:
            coord = self.param.coords
        ec = sorted(zip(self.param.all_elem,self.param.xyz_coord),key=lambda
                    x:self.param.ele2num[x[0]])
        elements = []
        groups = []
        for k,g in itertools.groupby(ec,key=lambda x:x[0]):
            elements.append(k)
            groups.append(list(g))
        hb = ''
        if self.param.atomic_basis:
            if len(self.param.basis_set.split(','))==1:
                lb = self.param.basis_set
            if len(self.param.basis_set.split(','))==2:
                lb = self.param.basis_set.split(',')[0]
                hb = self.param.basis_set.split(',')[1]
        elif self.param.basis_set == 'unknown' or self.param.basis_set == 'pseudo':
            print('{:s} basis cannot be convert to dalton basis, please input basis set by -b option'.format(self.methods.basis))
            sys.exit()
        else:
            lb = self.param.basis_set
        inp = open(basename+'.mol','w')
        inp.write('ATOMBASIS\n')
        inp.write(basename+'\n')
        inp.write('dalton structure input generated by inpgen_qc.py\n')
        inp.write('Atomtypes={:d} Charge={:s} Angstrom\n'.
                  format(len(elements),self.param.charge_spin.split()[0]))
        for i in range(len(elements)):
            atom_charge=self.param.ele2num[elements[i]]
            if self.param.atomic_basis:
                print(self.param.atomic_basis)
                for k,v in self.param.atomic_basis.items():
                    if elements[i] in v:
                        basis = 'Basis={:s}'.format(k)
            else:
                basis = 'Basis={:s}'.format(self.param.basis_set)
            if self.param.ecp_basis:
                for k,v in self.param.atomic_basis.items():
                    if elements[i] in v:
                        basis = basis + 'ECP={:s}'.format(k)
            inp.write('Charge={:.1f} Atoms={:d} {:s}\n'.format(float(atom_charge),len(groups[i]),basis))
            for j in range(len(groups[i])):
                ele = groups[i][j][0]+str(j+1)
                coord = list(groups[i][j][1])
                ec='{:<5s}{:>13.6f}{:>13.6f}{:>13.6f}'.format(*([ele]+coord))
                inp.write(ec+'\n')
        dal = open(basename+'.dal','w')
        dal.write('**DALTON INPUT\n')
        dal.write('.RUN RESPONSE\n')
        dal.write('#**INTEGRALS\n')
        dal.write('#.MNF-SO\n')
        dal.write('**WAVE FUNCTIONS\n')
        dal.write('.DFT\n')
        dal.write('B3LYP\n')
        dal.write('**RESPONSE\n')
        dal.write('#*LINEAR\n')
        dal.write('*QUADRATIC\n')
        dal.write('.CPPHEC\n')
        dal.write('#.CPPHMF\n')
        dal.write('.PRINT\n')
        dal.write('3\n')
        dal.write('.ROOTS\n')
        dal.write('1\n')
        dal.write('**END OF DALTON INPUT\n')

class multiArgs:
    def __init__(self,args):
        self.arg_list=[]
        self.args=args
        if '|' in self.args.add_keywords:
            add_kw_list=args.add_keywords.split('|')
            app_name_list=[]
            for s in add_kw_list:
                s=s.translate(None,'!@#$%^&*()=/')
                app_name='_'+s
                app_name_list.append(app_name)
            for i,k in enumerate(add_kw_list):
                new_args=copy.deepcopy(self.args)
                new_args.add_keywords=k
                new_args.app_name=app_name_list[i]
                self.arg_list.append(new_args)
        else:
            self.arg_list=[args]

args=parser.parse_args()
multi_args=multiArgs(args)
for i in args.inputfile:
    for arg in multi_args.arg_list:
        param = inputParam(arg,i)
        param.read_input_file()
        if args.file:
            # read keywords from other file
            kwparam = inputParam(arg,args.file)
            kwparam.read_input_file()
            param.keywords = kwparam.keywords
        param.prep4out()
        out=writeOut(param)
        out.write_qcinput()

