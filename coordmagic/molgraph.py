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
from .cell import *
from . import structure
from .parameter import Parameter
from .measurement import angle as measure_angle
from .measurement import torsion as measure_dihedral
#from .structurewriter import  write_structure

__all__ = [
    'n2formula'
]

def n2formula(graph, n):
    '''convet node label to molecule formula
    n is a list of node labels
    '''
    elems = list(nx.get_node_attributes(graph.subgraph(n), 'elem').values())
    c = [[c, elems.count(c)] for c in set(elems)]
    sc = sorted(c, key=lambda x: x[0])
    carbon = [i for i in sc if i[0] == 'C']
    hydrogen = [i for i in sc if i[0] == 'H']
    other = [i for i in sc if i[0] not in ['C','H']]
    hillsys = carbon+hydrogen + other
    formula = ''.join([i[0] + str(i[1]) for i in hillsys])
    return formula

def n2weight(graph,n):
    '''convet node label to mol weight
    n is a list of node labels
    '''
    elems = list(nx.get_node_attributes(graph.subgraph(n), 'elem').values())
    param = Parameter()
    mol_weight = sum([param.elem2an[i] for i in elems])
    return mol_weight


class MolGraph:
    '''generate graph of the structure'''
    def __init__(self, structure):
        self.S = structure
        self.set_threshold()

    def set_threshold(self, cov_scale=1.1,vdw_scale=1.1,cov_delta=0, vdw_delta=0,
                    usrcovr="",usrvdwr="",usrcovth="",usrvdwth=""):
        '''the threshold for determining bond is defined as:
         (covalent_radii_atom1 + covalent radii_atom2) * cov_scale + cov_delta
         the threshold for determining VDW interaction is defined as
         (VDW_radii_atom1 + VDW radii_atom2) * VDW_scale + VDW_delta
         usrvdwr/usrcovr can define VDW/covalent radii for the specific element.
         The format is like "C=3.5,H=1.2"
         usrcovth/usrvdwth can define specific covalent/vdw distance threshold for specific element pairs
         The format is like "C-H=1.0,C-S=1.7"
         These parameters will be used in self.gen_graph() function
         '''
        param = Parameter()
        if len(usrvdwth) > 0:
            e2vdw_th = {frozenset(i.split('=')[0].split('-')):float(i.split('=')[1]) for i in usrvdwth.split(',')}
        else:
            e2vdw_th = {}
        if len(usrcovth) > 0:
            e2bond_th = {frozenset(i.split('=')[0].split('-')):float(i.split('=')[1]) for i in usrcovth.split(',')}
        else:
            e2bond_th = {}
        covr=param._covr.copy()
        if len(usrcovr) > 0:
            covr.update({i.split('=')[0]:float(i.split('=')[1]) for i in usrcovr.split(',')})
        vdwr = param._vdwr.copy()
        if len(usrvdwr) > 0:
            vdwr.update({i.split('=')[0]:float(i.split('=')[1]) for i in usrvdwr.split(',')})
        self.vdw_scale = float(vdw_scale)
        self.cov_scale = float(cov_scale)
        self.vdw_delta = float(vdw_delta)
        self.cov_delta = float(cov_delta)
        self.covr=covr
        self.vdwr=vdwr
        self.e2bond_th = e2bond_th
        self.e2vdw_th = e2vdw_th
        self.if_trimmed = defaultdict(str) # a state to record if same type molecule has been trimmed
        self.trimmed_idx = defaultdict(list)
        self.unwrap_mol_record = defaultdict(str)



    def graph2struct(self, graph, pbc='auto'):
        '''generate structure object from a graph
        you may not need this function if the graph is a subgraph
        of the self.S.graph. In this case you could use
        st=self.S.extract_struc(subgraph.nodes())
        to generate a structure
        The pbc option is auto indicate that if the self.S
        object has pbc, the newly generated structure will have same pbc param
        '''
        st = structure.Structure()
        if pbc == 'auto' or pbc == 'on':
            st.cell_param = self.S.cell_param
            st.cell_vect = self.S.cell_vect
        else:
            st.period_flag = 0
        for sn in sorted(list(graph.nodes())):
            st.atoms.append(defaultdict(str,graph.nodes[sn]))
            st.complete_self(wrap=False)
        return  st
            # for atom in copy.deepcop

    def gen_graph(self,edge_dist_th=5,expand_length=[5,5,5]):
        '''generate graph and molecules lists for structure
        the node properties is a copy (not reference) of the atom dict
        if the structure is periodic, then the image atom within the expand_length (A) to the
        a,b,c axis are added to construct the graph, in order to keep the edge that across the
        boundary. Inter atom distance smaller than edge_dist_th will be designate as an edge
        the self.S.graph properties will be generated
        '''



        if self.S.period_flag:
            s = add_image_atom(self.S, expand_length)
        else:
            s = self.S
        # print(s.atoms)
        tree = cKDTree(s.coord)
        dist_mat = cKDTree.sparse_distance_matrix(tree, tree, edge_dist_th)
        G = nx.from_scipy_sparse_matrix(dist_mat)
        mapping = {i:j['sn'] for i,j in enumerate(s.atoms)}
        G = nx.relabel_nodes(G, mapping, copy=True)
        nx.set_node_attributes(G,{i['sn']:i for i in self.S.atoms})
        G.n2formular = n2formula
        self.S.graph = G
        self.edge_dist_th=edge_dist_th
        print('Graph of whole structure generated')

    def refine_edge(self):
        '''refine edges, remove edges longer than the sum of covalent radii * cov_scale + cov_delta and edges of same atom
            issue a warning if the length of edge is shorter than 0.5.
            The edge that is smaller than the sum of vdw radii * vdw_scale + vdw_scale will be collected to built the super graph
            usrcovr/usrvdwr in the format of "C=1.5,H=1.0" will overwrite default covlant radii/vdw radii
            usrcovth/usrvdwth in the format of "C-S=3.5,C-H=3.0" will overwrite distance threshold of element pair
            this function will modify self.S.graph and generate self.vdw_edge
            '''
        G = self.S.graph
        edge2len = {}
        vdw_edge = []
        remove_edge = []
        G.remove_edges_from(nx.selfloop_edges(G))
        for i,e in  enumerate(G.edges):
            e1 = G.nodes[e[0]]['elem']
            e2 = G.nodes[e[1]]['elem']
            edge = frozenset([e1,e2])
            if edge in edge2len:
                bond_len,vdw_len = edge2len[edge]
            else:
                bond_len = self.covr[e1][0] + self.covr[e2][0]
                vdw_len = self.vdwr[e1] + self.vdwr[e2]
                edge2len[edge] = [bond_len,vdw_len]
            if edge in self.e2bond_th:
                cov_th = self.e2bond_th[edge]
            else:
                cov_th = self.covr[e1][0]*self.cov_scale+3*self.covr[e1][1] + \
                         self.covr[e2][0]*self.cov_scale+3*self.covr[e2][1] + self.cov_delta
                self.e2bond_th[edge] = cov_th
            if edge in self.e2vdw_th:
                vdw_th = self.e2vdw_th[edge]
            else:
                vdw_th = self.vdwr[e1] * self.vdw_scale + self.vdwr[e2] * self.vdw_scale + self.vdw_delta
                self.e2vdw_th[edge] = vdw_th
            if G.edges[e]['weight'] > cov_th:
                remove_edge.append(e)
                if G.edges[e]['weight'] < vdw_th:
                    G.edges[e]['vdw_deviation'] = G.edges[e]['weight'] - vdw_len
                    vdw_edge.append((e[0],e[1],G.edges[e]))
            elif G.edges[e]['weight'] < 0.5:
                print('Warning! Close contact ({:.2f}) between {:d}:{:s} and {:d}:{:s} detected'
                      .format(G.edges[e]['weight'],e[0],e1,e[1],e2))
            else:
                G.edges[e]['deviation'] = G.edges[e]['weight'] - bond_len

        G.remove_edges_from(remove_edge)
        self.vdw_edge = vdw_edge
                # G.edges[e]['len'] = bond_len
                # G.edges[e]['ele'] = (e1,e2)

    def identify_unique_molecule(self, silent=False):
        '''identify unique molecules in self.S.molecules'''
        # to deal with same molecule with different atom order
        ids = []
        formulas = []
        hashs = []
        fingerprints = []
        for id,mol in self.S.molecules.items():
            #fingerprint is a string of element and degree of nodes
            #used to identify
            ids.append(mol['id'])
            formulas.append(mol['formula'])
            hashs.append(mol['hash'])
            fingerprints.append(mol['elem']+mol['degree'])
        mol_df = pd.DataFrame.from_dict({'id':ids,'formula':formulas,'hash':hashs,'fingerprint':fingerprints})
        # print(mol_df.loc[1,'fingerprint'])
        # print(mol_df)
        mol_df_c = mol_df.groupby(['formula','hash'])\
                    .agg({'id':'count',
                          'fingerprint':lambda x:len(set(x))}).reset_index()
        mol_df_type = mol_df.groupby(['formula','hash']).agg({'id': lambda x:list(x)}).reset_index()
        for idx,row in mol_df_type.iterrows():
            type_id = idx+1
            for m in [m for id,m in self.S.molecules.items() if id in row['id']]:
                m['type_id'] = type_id
        mol_str = []
        for idx,row in mol_df_c.iterrows():
            if row['fingerprint']==1:
                mol_str.append('Type {:d}: {:s} X {:d}'.format(idx+1,row['formula'],row['id']))
            else:
                mol_str.append('Type {:d}: {:s} X {:d} (diff. order)'
                      .format(idx+1,row['formula'],row['id'],row['fingerprint']))
        self.S.mol_str = mol_str
        self.S.mol_df = mol_df
        if not silent:
            print("\n".join(mol_str))

    def gen_internal_coords(self, measure=False):
        '''generate bond angle dihedral for each molecule in self.S.molecules
        therefore self.gen_mol should be run first
        this will generate a dictionary where keys are atom sn and values are measurements
        '''
        for _,mol in self.S.molecules.items():
            graph = mol['graph']
            bonds = {k:{'value':v['weight']} for k,v in graph.edges().items()}
            angle_sn = []
            dihedral_sn = []
            improper_sn = []
            # generate angle 
            for n in graph.nodes():
                atom13 = list(itertools.combinations(graph[n],2))
                angle_sn += [(i[0],n,i[1]) for i in atom13]
            angles = defaultdict(dict,{i:{} for i in angle_sn})
            if measure:
                for sn in angle_sn:
                    c1,c2,c3 = [graph.nodes[i]['coord'] for i in sn]
                    angles[sn]['value'] = measure_angle(c1,c2,c3)
            # generate dihedral
            for b in bonds.keys():
                for atom1 in graph[b[0]]:
                    for atom4 in graph[b[1]]:
                        if atom1 != b[1] and atom4 != b[0] and atom1 != atom4 :
                            dihedral_sn.append((atom1,b[0],b[1],atom4))
            dihedrals = defaultdict(dict,{i:{} for i in dihedral_sn})
            if measure:
                for sn in dihedral_sn:
                    c1,c2,c3,c4 = [graph.nodes[i]['coord'] for i in sn]
                    dihedrals[sn]['value'] = measure_dihedral(c1,c2,c3,c4)
                    n1 = len(graph[sn[1]]) - 1
                    n2 = len(graph[sn[2]]) - 1
                    dihedrals[sn]['period'] = (n1,n2)
            # generate improper ABCD, where A has only three neighbor BCD the dihedral is between ABC and BCD
            # therefor the order of BCD matters, here the order of BCD is by the order of sn
            for n in graph.nodes():
                if len(graph[n]) == 3:
                    improper_sn += [tuple([n] + [i for i,v in graph[n].items()])]
            impropers = defaultdict(dict,{i:{} for i in improper_sn})
            if measure:
                for sn in improper_sn:
                    c1,c2,c3,c4 = [graph.nodes[i]['coord'] for i in sn]
                    impropers[sn]['value'] = measure_angle([c1,c2,c3],[c2,c3,c4],mtype='pp')
            # add internal coords to mol defaultdict
            mol['bonds'] = bonds
            mol['angles'] = angles
            mol['dihedrals'] = dihedrals
            mol['impropers'] = impropers
        # add internal coords to structure
        allb = [m['bonds'] for m in self.S.molecules.values()]
        alla = [m['angles'] for m in self.S.molecules.values()]
        alld = [m['dihedrals'] for m in self.S.molecules.values()]
        alli = [m['impropers'] for m in self.S.molecules.values()]
        self.S.bonds = {k:v for b in allb for k,v in b.items()}
        self.S.angles = {k:v for a in alla for k,v in a.items()}
        self.S.dihedrals = {k:v for d in alld for k,v in d.items()}
        self.S.impropers = {k:v for i in alli for k,v in i.items()}

    def gen_bond_order(self,formula=""):
        '''generate bond_order key:value pair for each bond in self.S.bonds dict 
        by the "deviation" properties of graph edge with is the deviation from standart single covalent bond length
        the bond_order is a float number and should be conver to integer in downsteam function
        two formulas are availabe, default is:
        0.78*(n**0.33-1) = deviation
        and Pauling equation will be used if formula.startswith("p")
        0.71*log(n) = deviation
        '''
        for k,v in self.S.bonds.items():
            dev = v['deviation']
            if formula.startswith('p'):
                order = 10**(dev/0.71)
            else:
                order = np.exp(np.log(dev/0.78+1)/0.33)
            multi_bond_elem = ['C','N','O','S','P','Se','As']
            if self.S.graph.nodes[k[0]]['elem'] in multi_bond_elem and self.S.graph.nodes[k[1]]['elem'] in multi_bond_elem:
                v['bond_order'] = order
            else:
                v['bond_order'] = 1

    def gen_mol(self, silent=False):
        '''generate molecules properties for self.S
        besed on graph and connected_components
        which is a dict of default dict, keys are molid and
        elements of which are default dict of molecules
        the atom properties of "molid" and "formula" has been generated
        self.S.mol_list is generated which is a dataframe of mol id and formula an
        '''
        self.gen_graph()
        self.refine_edge()
        G = self.S.graph
        sn2molid = {}  #dict map atom serial number to molid
        sn2formula = {} #dict map atom serial number to molid
        molecules = []
        for i,c in enumerate(sorted(nx.connected_components(G),key=lambda x:len(x),reverse=True)):
            c=sorted(c)
            mol = defaultdict(str)
            molid=i+1
            sn2molid.update({sn:{'molid':molid} for sn in c})
            formula = n2formula(G,c)
            weight = n2weight(G,c)
            sn2formula.update({sn:{'formula':formula} for sn in c})
            sn2formula.update({sn:{'mol_weight':weight} for sn in c})
            mol['id'] = molid
            mol['sn'] = c
            mol['formula'] = formula
            mol['graph'] = G.subgraph(c)  # !!!the node order is changed,  graph, edge and node attributes are shared with the original graph.
            mol['hash'] = graph_hashing.weisfeiler_lehman_graph_hash(mol['graph'],node_attr='elem')
            mol['elem'] = ''.join([i[1] for i in sorted(mol['graph'].nodes(data='elem'),key=lambda x:x[0])])
            mol['degree'] = ''.join([str(i[1]) for i in sorted(mol['graph'].degree,key=lambda x:x[0])])
            mol['struct'] = self.S
            # mol['cell_param'] = copy.deepcopy(self.S.cell_param)
            mol['cell_vect'] = copy.deepcopy(self.S.cell_vect)
            molecules.append(mol)
        nx.set_node_attributes(G,sn2molid)
        nx.set_node_attributes(G,sn2formula)
        mol2id = {i['id']:i for i in molecules}
        mol_list = pd.DataFrame(molecules)[['id','formula']].set_index('id')
        self.S.mol_list = mol_list
        self.S.molecules = mol2id
        self.identify_unique_molecule(silent=silent)
        # update atoms after atom properties of "molid" and "formula" has been added to the graph
        self.S.atoms = [defaultdict(str, v) for i, v in G.nodes.data()]
        self.gen_supergraph()


    def unwrap_mol(self, mol):
        '''mol is a default dict in the dict of self.S.molecules
        this function will add a 'unwrap' properties to atoms
        '''
        # check if the molecule is already unwraped
        if self.unwrap_mol_record[mol['id']]:
            return
        if not (self.S.cell_param or self.S.cell_vect):
            return
        s = self.S.extract_struc(mol['sn'])
        tree = cKDTree(s.coord)
        dist_mat = cKDTree.sparse_distance_matrix(tree, tree, self.edge_dist_th)
        G = nx.from_scipy_sparse_matrix(dist_mat)
        mapping = {i: j['sn'] for i, j in enumerate(s.atoms)}
        G = nx.relabel_nodes(G, mapping, copy=True)
        nx.set_node_attributes(G, {i['sn']: i for i in s.atoms})
        # refine edge
        remove_edge = []
        G.remove_edges_from(nx.selfloop_edges(G))
        for i, e in enumerate(G.edges):
            e1 = G.nodes[e[0]]['elem']
            e2 = G.nodes[e[1]]['elem']
            edge = frozenset([e1, e2])
            if edge in self.e2bond_th:
                cov_th = self.e2bond_th[edge]
            else:
                cov_th = self.covr[e1][0] * self.cov_scale + 3 * self.covr[e1][1] + \
                         self.covr[e2][0] * self.cov_scale + 3 * self.covr[e2][1] + self.cov_delta
                self.e2bond_th[edge] = cov_th
            if G.edges[e]['weight'] > cov_th:
                remove_edge.append(e)
            elif G.edges[e]['weight'] < 0.5:
                print('Warning! Close contact ({:.2f}) between {:d}:{:s} and {:d}:{:s} detected'
                      .format(G.edges[e]['weight'], e[0], e1, e[1], e2))
        G.remove_edges_from(remove_edge)
        # compare the molecule graph to the full molecule graph and found the missing_edge and the
        # disconnected node of each fragment
        # generate a list of fragment, each fragment is represented as a dictionary with keys:
        # 'edge_node':disconnected node, 'sn':atom sn belong to the frag, 'length':size of the frag,'id',
        # 'pos':cell position [0,0,0] is the origin cell
        # start from the largest fragment in cell. For each of its edge_node (fix_node or fn)
        # find its counterpart in missing edge (moving_node or mn). Compute the displacement vector from
        # fn to mn in fraction coord. round this fraction coord to integer and used as the cell displace_vect
        # update the pos properties of the fragment containing mn to cell displace vect
        # and make this fragment as a member of new fix frag
        # remove the fn and mn from missing_edge
        template = dict(mol['graph'].degree())
        missing_edge = {}
        disconnected_node = []
        for k, v in G.degree():
            if template[k] > v:
                disconnected_node.append(k)
                for node in mol['graph'][k].keys():
                    if node not in G[k]:
                        missing_edge[tuple(sorted([k, node]))] = mol['graph'][k][node]['weight']
            elif template[k] < v:
                print('Warning!!! more edges in fragment than in complete molecule graph, Something wrong!')
        # find the fragment
        frags = []
        for id, frag in enumerate(list(nx.connected_components(G))):
            frag_info = {'edge_node': [], 'sn': frag, 'length': len(frag), 'id': id, 'pos': [0, 0, 0]}
            for node in disconnected_node:
                if node in frag:
                    frag_info['edge_node'].append(node)
            frags.append(frag_info)
        fix_frags = [sorted(frags, key=lambda x: x['length'])[-1]]  # fix the largest fragments
        connected_frags = [i for i in fix_frags]
        while len(missing_edge) > 0:
            next_fix_frags = []
            for ff in fix_frags:  # ff stands for fix_frag
                for fn in ff['edge_node']:  # fn stands for fix node, which is disconnected node in fix frag
                    connected_edge = []
                    for me in missing_edge.keys():
                        if fn in me:
                            mn = [n for n in me if n != fn][0]  # mn stands for moving node
                            fn_fcoord = np.array(self.S.get_atom([fn])[0]['fcoord']) + np.array(ff['pos'])
                            fn_coord = self.S.frac2cart(fcoord=fn_fcoord)
                            moving_frag = [i for i in frags if mn in i['edge_node']][0]
                            mn_fcoord = np.array(self.S.get_atom([mn])[0]['fcoord']) + np.array(moving_frag['pos'])
                            mn_coord = self.S.frac2cart(fcoord=mn_fcoord)
                            displace_vect = np.array(fn_coord) - np.array(mn_coord)
                            displace_cell = np.rint(np.matmul(displace_vect, np.linalg.inv(self.S.cell_vect)))
                            if any(np.rint(displace_cell) != 0):
                                if all(np.array(moving_frag['pos']) == 0):
                                    moving_frag['pos'] = displace_cell
                                else:
                                    print('Warning!!! conflict moving fragment (id:{:d}) of molecule '
                                          '(id:{:d}, formula:{:s}) from {:s} to {:s} when unwrap this molecule'
                                          .format(moving_frag['id'], mol['id'], mol['formula'],
                                                  ','.join([str(int(i)) for i in moving_frag['pos']]),
                                                  ','.join([str(int(i)) for i in displace_cell]))
                                          )
                            connected_edge.append(tuple(sorted([fn, mn])))
                            next_fix_frags.append(moving_frag['id'])
                    for e in connected_edge:
                        missing_edge.pop(e)
            fix_frags = [i for i in frags if i['id'] in next_fix_frags]

        # update the coords in self.S or generate and return new structure object depend on inplace parameter
        for f in frags:
            for atom in self.S.get_atom(f['sn']):
                atom['fcoord'] = np.array(atom['fcoord']) + np.array(f['pos'])
                atom['coord'] = self.S.frac2cart(atom['fcoord'])
                atom['unwrap'] = f['pos']
        self.S.coord = self.S.getter('coord')
        self.S.fcoord = self.S.getter('fcoord')
        nx.set_node_attributes(self.S.graph, {i['sn']: i for i in self.S.atoms})
        self.unwrap_mol_record[mol['id']] = 1
        # print("{:f}-{:f}-{:f}-{:f}-{:f}-{:f}-{:f}".format(t11-t1,t12-t11,t13-t12,t14-t13,t15-t14,t16-t15,t17-t16))
        # print("{:f}-{:f}-{:f}-{:f}".format(t2-t1,t3-t2,t4-t3,t5-t4))
        # s = copy.deepcopy(self.S)
        # for f in frags:
        #     for atom in s.get_atom(f['sn']):
        #         atom['fcoord'] = np.array(atom['fcoord']) +  np.array(f['pos'])
        #         atom['coord'] = self.S.frac2cart(atom['fcoord'])
        # s.complete_self(wrap=False)
        # return s
        # sw=StructureWriter()
        # sw.write_file(self.S,"test",ext='mol2')

    def structure_matcher(self,G1,G2):
        '''return a list of mapping from G1 to G2
        G1 should be larger or equal to G2
         don't forget you question on stackoverflow!!!'''
        nm = isomorph.categorical_node_match("elem")
        GM = isomorph.GraphMatcher(G1, G2, node_match = nm)
        for map in GM.subgraph_isomorphisms_iter():
            print(map)


    def gen_supergraph(self):
        '''generate supergraph, which is a multigraph. The nodes are id of individual molecules
        and the edges are weak interactions (stored in self.vdw_edge)
        each edge has three properties:
        key is a tuple of the atom serial number that form the edge
        weight is the length of the edge
        vdw_deviation is the deviation from standard vdw distance'''
        SG = nx.MultiGraph()
        sgnodes = [(id,mol) for id,mol in self.S.molecules.items()]
        SG.add_nodes_from(sgnodes)
        for e in self.vdw_edge:
            id1 = self.S.graph.nodes[e[0]]['molid']
            id2 = self.S.graph.nodes[e[1]]['molid']
            if id1 != id2:
                SG.add_edge(id1,id2,key=(e[0],e[1]),sn=(e[0],e[1]),**e[2])
        self.S.supergraph = SG
        # writer= StructureWriter()
        # st=self.S.extract_struc(self.S.supergraph.nodes[1]['sn'])
        # writer.write_file(st,basename="st1",ext='res')

    def remove_SGedge_byedge(self,include="",exclude=""):
        '''
        include or exclude is a list of conditions,
        for include list, an edge satisfy and condition will be kept
        for exclude list, an edge satisfy any condition will be removed
        If both list are available, exclude list has higher priority
        for each condition, the format is
        key1=value1,key2=value2---key1=value1,key2=value2:distance_threshold
        the key values pairs are used to select atoms
        and the distance_threshold is the upper limit for include condition, default 1000
        and lower limit for exclude condition, default 0
        for example:
        formula:C12H20,elem=C---formula:C6H6O2,elem=O:3.0
        '''
        ops = {"=": (lambda x, y: x == y),
               "==": (lambda x, y: x == y),
               ">": (lambda x, y: float(x) > float(y)),
               "<": (lambda x, y: float(x) < float(y)),
               ">=": (lambda x, y: float(x) >= float(y)),
               "<=": (lambda x, y: float(x) >= float(y)),
               }
        remove_edge = []
        keep_edge = []
        if len(include) > 0:
            include_list = []
            for con in include.split(';'):
                if ':' in con:
                    dist_th = float(con.split(':')[1])
                    if dist_th == "":
                        dist_th = 1000
                    atom_pair = con.split(':')[0]
                else:
                    dist_th = 1000
                    atom_pair = con
                i1 = [tuple(re.split('(==|>=|<=|=|>|<)', i)) for i in atom_pair.split('---')[0].split(',')]
                i2 = [tuple(re.split('(==|>=|<=|=|>|<)', i)) for i in atom_pair.split('---')[1].split(',')]
                include_list.append((i1, i2, dist_th))
            for e, datadict in self.S.supergraph.edges.items():
                n1 = self.S.graph.nodes[datadict['sn'][0]]
                n2 = self.S.graph.nodes[datadict['sn'][1]]
                dist = datadict['weight']
                for con in include_list:
                    if all(ops[o](n1[k], v) for k, o, v in con[0]) and \
                       all(ops[o](n2[k], v) for k, o, v in con[1]) and \
                       dist < con[2] and e not in keep_edge:
                        keep_edge.append(e)
                    elif all(ops[o](n2[k], v) for k, o, v in con[0]) and \
                         all(ops[o](n1[k], v) for k, o, v in con[1]) and \
                         dist < con[2] and e not in keep_edge:
                        keep_edge.append(e)
            for e, _ in self.S.supergraph.edges.items():
                if e not in keep_edge:
                    remove_edge.append(e)

        if len(exclude) > 0:
            exclude_list = []
            for con in exclude:
                if ':' in con:
                    dist_th = float(con.split(':')[1])
                    if dist_th == "":
                        dist_th = 1000
                    atom_pair = con.split(':')[0]
                else:
                    dist_th = 1000
                    atom_pair = con
                e1 = [tuple(re.split('(==|>=|<=|=|>|<)',i)) for i in atom_pair.split('---')[0].split(',')]
                e2 = [tuple(re.split('(==|>=|<=|=|>|<)',i)) for i in atom_pair.split('---')[1].split(',')]
                exclude_list.append((e1,e2,dist_th))
            for e, datadict in self.S.supergraph.edges.items():
                n1 = self.S.graph.nodes[datadict['sn'][0]]
                n2 = self.S.graph.nodes[datadict['sn'][1]]
                dist = datadict['weight']
                for con in exclude_list:
                    if all(ops[o](n1[k], v) for k, o, v in con[0]) and \
                       all(ops[o](n2[k], v) for k, o, v in con[1]) and \
                       dist > con[2] and e not in remove_edge:
                       remove_edge.append(e)
                    elif all(ops[o](n2[k], v) for k, o, v in con[0]) and \
                         all(ops[o](n1[k], v) for k, o, v in con[1]) and \
                         dist > con[2] and e not in remove_edge:
                         remove_edge.append(e)
        print("Removing {:d} weak contacts by atom pair".format(len(remove_edge)))
        self.S.supergraph.remove_edges_from(remove_edge)

    def remove_SGedge_byatom(self,include="",exclude=""):
        '''
        select atoms that include in the supergraph
        the format is key1=value1,key2<value2
        '''
        ops = {"=": (lambda x, y: x == y),
               "==": (lambda x, y: x == y),
               ">": (lambda x, y: float(x) > float(y)),
               "<": (lambda x, y: float(x) < float(y)),
               ">=": (lambda x, y: float(x) >= float(y)),
               "<=": (lambda x, y: float(x) >= float(y)),
               }
        remove_edge = []
        keep_edge = []
        if len(include) > 0:
            include_list = []
            for con in include.split(';'):
                i = [tuple(re.split('(==|>=|<=|=|>|<)', i)) for i in con.split(',')]
                include_list.append(i)
            for e, datadict in self.S.supergraph.edges.items():
                n1 = self.S.graph.nodes[datadict['sn'][0]]
                n2 = self.S.graph.nodes[datadict['sn'][1]]
                for con in include_list:
                    if all(ops[o](n1[k], v) for k, o, v in con) and e not in keep_edge:
                        keep_edge.append(e)
                    elif all(ops[o](n2[k], v) for k, o, v in con) and e not in keep_edge:
                        keep_edge.append(e)
            for e, _ in self.S.supergraph.edges.items():
                if e not in keep_edge:
                    remove_edge.append(e)
        if len(exclude) > 0:
            exclude_list = []
            for con in exclude.split(';'):
                e = [tuple(re.split('(==|>=|<=|=|>|<)',i)) for i in con.split(',')]
                exclude_list.append(e)
            for e, datadict in self.S.supergraph.edges.items():
                n1 = self.S.graph.nodes[datadict['sn'][0]]
                n2 = self.S.graph.nodes[datadict['sn'][1]]
                for con in exclude_list:
                    if all(ops[o](n1[k], v) for k,o,v in con) and e not in remove_edge:
                         remove_edge.append(e)
                    elif all(ops[o](n2[k], v) for k, o, v in con) and e not in remove_edge:
                        remove_edge.append(e)
        print("Removing {:d} weak contacts by atom".format(len(remove_edge)))
        self.S.supergraph.remove_edges_from(remove_edge)


    def analysis_close_contact(self,prop=''):
        '''analysis close contact based on super graph
        prop is a list of atom properties
        the close contact is a list of distance grouped by
        formula element and user defined atom properties'''

        if len(self.S.supergraph.nodes()) > 1:
            clusters = []
            for m in nx.connected_components(self.S.supergraph):
                types = [self.S.supergraph.nodes[i]['type_id'] for i in m]
                c = [[c, types.count(c)] for c in set(types)]
                sc = sorted(c, key=lambda x: x[0])
                cluster_compos = '-'.join([str(i[1]) + 'T' + str(i[0]) for i in sc])
                clusters.append(cluster_compos)
            c = [[c, clusters.count(c)] for c in set(clusters)]
            sc = sorted(c, key=lambda x: ((len(x[0]),x[0])), reverse=True)
            struc_compse = ';'.join([str(i[1]) + ' * [' + str(i[0]) + ']' for i in sc])
            print('{:d} clusters detected. Their compositions are {:s}'.format(len(clusters), struc_compse))
            close_contact = {}
            std_contact = {}
            for e, datadict in self.S.supergraph.edges.items():
                f1 = self.S.supergraph.nodes[e[0]]['formula']
                f2 = self.S.supergraph.nodes[e[1]]['formula']
                elem1 = self.S.graph.nodes[datadict['sn'][0]]['elem']
                elem2 = self.S.graph.nodes[datadict['sn'][1]]['elem']
                prop1 = '+'.join([str(self.S.graph.nodes[datadict['sn'][0]][i]) for i in prop.split(',') if i])
                prop2 = '+'.join([str(self.S.graph.nodes[datadict['sn'][1]][i]) for i in prop.split(',') if i])
                p=sorted([(f1,elem1,prop1),(f2,elem2,prop2)],key=lambda x:x[0])
                key = (p[0][0],p[0][1],p[0][2],p[1][0],p[1][1],p[1][2])
                distance = datadict['weight']
                if key in close_contact:
                    close_contact[key].append(distance)
                else:
                    close_contact[key] = [distance]
                if key not in  std_contact:
                    standard = self.vdwr[elem1] + self.vdwr[elem2]
                    threshold = standard * self.vdw_scale + self.vdw_delta
                    std_contact[key] = (standard,threshold)
            sorted_contact=sorted([(k,v) for k,v in close_contact.items()],key=lambda x:(x[0][0],x[0][3],x[0][1],x[0][4],x[0][2],x[0][5]))
            head_str='Formula:elem:' + '+'.join(prop.split(','))
            max_key_len = max([len(":".join(i))+4 for i in close_contact.keys()] +[len(head_str)] )
            klen = str(max_key_len + 2)
            print('Close contact with VDW_scale={:.2f} and VDW_delta={:.2f} :'.format(self.vdw_scale,self.vdw_delta))
            print(('{:<'+klen+'s}{:>4s}{:>7s}{:>7s}{:>7s}{:>7s}{:>7s}')
                  .format(head_str,'n','mean','min','max','norm','thresh'))
            for k,v in sorted_contact:
                label = '{:s}:{:s}:{:s} --- {:s}:{:s}:{:s}'.format(*k)
                dist_str = ('{:<'+klen+'s}{:>4d}{:>7.3f}{:>7.3f}{:>7.3f}{:>7.3f}{:>7.3f}').format(
                           label,len(v),np.mean(v),np.min(v),np.max(v),std_contact[k][0],std_contact[k][1])
                print(dist_str)
            print('')




