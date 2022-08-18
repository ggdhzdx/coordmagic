import numpy as np
import re
import copy
import sys
import pandas as pd
import itertools
import networkx as nx
import networkx.algorithms.graph_hashing as graph_hashing
import networkx.algorithms.isomorphism as isomorph
from collections import defaultdict
from scipy.spatial import cKDTree
from . import structure

# add label edge


class LabelAtom:
    '''generate graph of the structure'''
    def __init__(self, structure):
        '''the threshold for determining bond is defined as:
         (covalent_radii_atom1 + covalent radii_2) * cov_scale + cov_delta
         the threshold for determining VDW interaction is defined as
         (VDW_radii_atom1 + VDW radii_atom2) * VDW_scale + VDW_delta
         usrvdwr/usrcovr can define VDW/covalent radii for the specific element.
         The format is like "C=3.5,H=1.2"
         usrcovth/usrvdwth can define specific covalent/vdw distance threshold for specific element pairs
         The format is like "C-H=1.0,C-S=1.7"
         These parameters will be used in self.gen_graph() function
         '''
        self.S = structure

    def add_atoms_label(self):
        pass

    def label_atoms_by(self,rules,onebyone=False,verbose=1):
        '''assign fragment related or descriptor related information to every atoms
        note that the infor in graph and infor in dict list are updated simutaneously'''
        frag_rules = {'conjugate':self.label_by_conjugation,
                      'cycle':self.label_by_cycle}
        if len(self.S.molecules) < 1:
            self.S.G.gen_mol()
        hash2frag = {}
        info_printed = []
        if onebyone == True:
            for id,mol in self.S.molecules.items():
                if mol['hash'] in info_printed:
                    print_level = 0
                else:
                    print_level = verbose
                frag_rules[rules](mol,print_level = print_level)
                info_printed.append(mol['hash'])
        # writer= StructureWriter()
        # st=self.S.extract_struc(self.S.molecules[428]['sn'])
        # writer.write_file(st,basename="428",ext='res')
        # st=self.S.extract_struc(self.S.molecules[429]['sn'])
        # writer.write_file(st,basename="429",ext='res')
        else:
            for id,mol in self.S.molecules.items():
                key = mol['hash']+mol['elem']+mol['degree']
                if key not in hash2frag:
                    if mol['hash'] in info_printed:
                        print_level = 0
                    else:
                        print_level = verbose
                    hash2frag[key] = frag_rules[rules](mol,print_level = print_level)
                    info_printed.append(mol['hash'])
                else:
                    for i,n in enumerate(sorted(mol['graph'].nodes())):
                        mol['graph'].nodes[n].update(hash2frag[key][i])
                    info_printed.append(mol['hash'])
        self.S.atoms = [defaultdict(str,v) for i,v in self.S.graph.nodes.data()]


    def label_by_cycle(self,mol):
        print('under develop')
        pass



    def label_by_conjugation(self,mol,print_level=1):
        '''Rules:B,C,Si with three connections are considered as conjugated
        N P with 2 or 1 connections are considered as conjugated
        O S with 2 or 1 connections are considered as conjugated
        N P with 3 connections that connected to conjugated atoms are conjugated
        F Cl Br I with 1 connections that connected to conjugated atoms are conjugated
        other atoms connect to conjugated atoms are half conjugated
        other atoms connect to half-conjugated or non-conjugated  atoms are  non-congjugated
        return list of dictionary, each element correspond to a atom (a node in the G)
        each dictionary contain three key:value pair
        [{'conju_state':"fc/hc/uc",'conju_degree':'1/2/3/4','conju_size':'a number'}]
        '''
        # need to add conjulength
        G=mol['graph']
        formula = mol['formula']
        mol_type = mol['type_id']
        # First loop: find absolute conjugate atoms
        for n,v in G.nodes.data():
            degree = G.degree(n)
            elem = v['elem']
            G.nodes[n]['conju_state'] = 'undefined'
            if elem in ['B','C','Si']:
                if degree == 4 :
                    G.nodes[n]['conju_state'] = 'uc'
                elif degree < 4 and degree > 1:
                    G.nodes[n]['conju_state'] = 'fc'
                elif degree > 4:
                    print('Waring, atom {:d}:{:s} is hypervalence({:d})'
                          .format(v['sn'],elem,degree))
                elif degree == 1:
                    print('Waring, atom {:d}:{:s} is hypovalence({:d})'
                          .format(v['sn'],elem,degree))
            elif elem in ['N','P']:
                if degree ==2 or degree == 1:
                    G.nodes[n]['conju_state'] = 'fc'
                elif degree > 3:
                    print('Waring, atom {:d}:{:s} is hypervalence({:d})'
                          .format(v['sn'],elem,degree))
            elif elem in ['O','S']:
                if degree == 1:
                    G.nodes[n]['conju_state'] = 'fc'
                elif degree > 3:
                    print('Waring, atom {:d}:{:s} is hypervalence({:d})'
                          .format(v['sn'],elem,degree))
        # second loop: find conditional  conjugate atoms
        for n,v in G.nodes.data():
            neighbors = [G.nodes[i] for i in G.neighbors(n)]
            degree = G.degree(n)
            elem = v['elem']
            if elem in ['N','P']:
                if degree == 3:
                    if any([i['conju_state'] == 'fc' for i in neighbors]):
                        G.nodes[n]['conju_state'] = 'fc'
                        # print('conditional conjugation elem:{:s}'.format(elem))
            elif elem in ['S','O']:
                if degree == 2:
                    if any([i['conju_state'] == 'fc' for i in neighbors]):
                        G.nodes[n]['conju_state'] = 'fc'
                        # print('conditional conjugation elem:{:s}'.format(elem))
            elif elem in ['F','Cl','Br','I']:
                if degree == 1:
                    if any([i['conju_state'] == 'fc' for i in neighbors]):
                        G.nodes[n]['conju_state'] = 'fc'
                        # print('conditional conjugation elem:{:s}'.format(elem))
                if degree > 1:
                    print('Waring, atom {:d}:{:s} is hypervalence({:d})'
                          .format(v['sn'],elem,degree))
            else:
                if G.nodes[n]['conju_state'] == 'undefined' or G.nodes[n]['conju_state'] == 'uc':
                    if any([i['conju_state'] == 'fc' for i in neighbors]):
                        G.nodes[n]['conju_state'] = 'hc'
                        # print('conditional conjugation elem:{:s} set to hc'.format(elem))
                    else:
                        G.nodes[n]['conju_state'] = 'uc'
        # compute conjuation system size
        conju_nodes = (n for n,d in G.nodes.data() if d['conju_state'] == 'fc')
        conju_graph = G.subgraph(conju_nodes)
        sn2conju_size = defaultdict(lambda:0) # map node sn to conjugate system size
        fc_len_list = []
        for c in nx.connected_components(conju_graph):
            fc_len_list.append(len(c))
            for sn in c:
                sn2conju_size[sn]=len(c)
        fc_len_list = [str(i) for i in sorted(fc_len_list,reverse=True)]
        # unconju_nodes = (n for n,d in G.nodes.data() if d['conju_state'] in ['uc','hc'])
        # unconju_graph = G.subgraph(unconju_nodes)
        # uc_len_list = []
        # for c in nx.connected_components(unconju_graph):
        #     uc_len_list.append(len(c))
        # uc_len_list = [str(i) for i in sorted(uc_len_list,reverse=True)]
        # third loop, find conjugate degree and conjugate system size for each node
        for n,v in G.nodes.data():
            neighbors = [G.nodes[i] for i in G.neighbors(n)]
            v['conju_degree'] = sum([i['conju_state'] == 'fc' for i in neighbors])
            v['conju_size'] = sn2conju_size[n]
        # print('summary is needed for conjugation fragmentation process')
        if print_level > 0:
            # print('=============== Fragmentation By Conjugation ===============')
            print('Mol Type:{:d}   Formula:{:s}'.format(mol_type,formula))
            print('There are {:d} conjugate systems in this molecule.\nTheir conju_size are {:s}'
                  .format(len(fc_len_list), ','.join(fc_len_list)))
            df = pd.DataFrame.from_dict(dict(G.nodes.data()), orient='index').loc[:,
                 ['elem', 'conju_state', 'conju_degree', 'conju_size']]
            cs = [(k,v) for k,v in df.groupby(['conju_state']).size().items()]
            css = ', '.join(sorted([str(k)+':'+str(v) for k,v in cs]))
            print("conju_state: {:s}".format(css))
            cd = sorted([(k,v) for k,v in df.groupby(['conju_degree']).size().items()],key=lambda x:x[1],reverse=True)
            cds = ', '.join(sorted([str(k)+':'+str(v) for k,v in cd]))
            print("conju_degree: {:s}".format(cds))
            if print_level > 1:
                print('conju_state: (fc--full conjugated, hc--half conjugated, uc--unconjugated)')
                print(df.groupby(['elem','conju_state']).size().reset_index(name='counts'))
                print('conju_degree: (number of conjugated atom connected to this atom)')
                print(df.groupby(['elem','conju_degree']).size().reset_index(name='counts'))
                print('conju_size: (size of conjugated system containing this atom)')
                print(df.groupby(['elem','conju_size']).size().reset_index(name='counts'))
            print('')
        return [{'conju_state':d['conju_state'],
                'conju_degree':d['conju_degree'],
                'conju_size':d['conju_size']} for sn,d in sorted(G.nodes.data(),key=lambda x:x[0])]

