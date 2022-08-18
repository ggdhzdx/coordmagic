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
from .structurewriter import  write_structure

class Cluster:
    '''generate super graph of the structure'''
    def __init__(self, structure,cov_scale=1.1,vdw_scale=1.1,cov_delta=0, vdw_delta=0,
                    usrcovr="",usrvdwr="",usrcovth="",usrvdwth=""):
        self.S = structure
        pass




    def find_cluster(self,num=2,neighbor_per_node=1,cluster_compos=''):
        '''extract cluster of size n and nedge great than nedge
        and nedge per node great than nedge_per_node
        return a list of subgraphs
        the node label of subgraph are mol id
        and node attribute are dict of that mol '''
        all_connected_subgraphs=[]
        G = self.S.supergraph.copy()
        for n in self.S.supergraph.nodes():
            ego=nx.generators.ego_graph(G,n,radius=num-1,center=False)
            for SG in (self.S.supergraph.subgraph(sn+(n,)) for sn in itertools.combinations(ego, num-1)):
                if nx.is_connected(SG):
                    all_connected_subgraphs.append(SG)
            G.remove_node(n)
        # filter cluster by other conditions
        accept_cluster = []
        for  SG in all_connected_subgraphs:
            accept_flag = 1 # default accept all subgraph
            if cluster_compos:
                accept_flag = 0
                types = [j for i, j in SG.nodes.data('type_id')]
                c = [[c, types.count(c)] for c in set(types)]
                sc = sorted(c, key=lambda x: x[0])
                compos = '-'.join([str(i[1]) + 'T' + str(i[0]) for i in sc])
                if compos in cluster_compos.split(','):
                    accept_flag = 1
            if neighbor_per_node > 1:
                accept_flag = 0
                if min([len(SG[i]) for i in SG.nodes()]) >= neighbor_per_node:
                    accept_flag = 1
            if accept_flag ==1:
                accept_cluster.append(SG)
        # print output cluster info
        all_compos = []
        for  SG in accept_cluster:
            types = [j for i, j in SG.nodes.data('type_id')]
            c = [[c, types.count(c)] for c in set(types)]
            sc = sorted(c, key=lambda x: x[0])
            compos = '-'.join([str(i[1]) + 'T' + str(i[0]) for i in sc])
            all_compos.append(compos)
        c = [[c, all_compos.count(c)] for c in set(all_compos)]
        sc = sorted(c, key=lambda x: ((len(x[0]), x[0])), reverse=True)
        struc_compose = '\n'.join([str(i[1]) + ' * [' + str(i[0]) + ']' for i in sc])
        print('=============== Clusters Found ===============')
        print('{:d} clusters of size {:d} found. Their compositions are:\n{:s}'
              .format(len(all_compos),num, struc_compose))
        print('')
        return accept_cluster
        # print("--- %s seconds ---" % (time.time() - stime))

    def expand_by_dist(self, ):
        '''generate a structure within distance around give atoms'''
        pass

    def expand_by_edge(self, step):
        '''expand along supergraph by step layer of nodes'''
        pass

    def find_cluster(self, num=2, neighbor_per_node=1, cluster_compos=''):
        '''extract cluster of size n and nedge great than nedge
        and nedge per node great than nedge_per_node
        return a list of subgraphs
        the node label of subgraph are mol id
        and node attribute are dict of that mol '''
        all_connected_subgraphs = []
        G = self.S.supergraph.copy()
        for n in self.S.supergraph.nodes():
            ego = nx.generators.ego_graph(G, n, radius=num - 1, center=False)
            for SG in (self.S.supergraph.subgraph(sn + (n,)) for sn in itertools.combinations(ego, num - 1)):
                if nx.is_connected(SG):
                    all_connected_subgraphs.append(SG)
            G.remove_node(n)
        # filter cluster by other conditions
        accept_cluster = []
        for SG in all_connected_subgraphs:
            accept_flag = 1  # default accept all subgraph
            if cluster_compos:
                accept_flag = 0
                types = [j for i, j in SG.nodes.data('type_id')]
                c = [[c, types.count(c)] for c in set(types)]
                sc = sorted(c, key=lambda x: x[0])
                compos = '-'.join([str(i[1]) + 'T' + str(i[0]) for i in sc])
                if compos in cluster_compos.split(','):
                    accept_flag = 1
            if neighbor_per_node > 1:
                accept_flag = 0
                if min([len(SG[i]) for i in SG.nodes()]) >= neighbor_per_node:
                    accept_flag = 1
            if accept_flag == 1:
                accept_cluster.append(SG)
        # print output cluster info
        all_compos = []
        for SG in accept_cluster:
            types = [j for i, j in SG.nodes.data('type_id')]
            c = [[c, types.count(c)] for c in set(types)]
            sc = sorted(c, key=lambda x: x[0])
            compos = '-'.join([str(i[1]) + 'T' + str(i[0]) for i in sc])
            all_compos.append(compos)
        c = [[c, all_compos.count(c)] for c in set(all_compos)]
        sc = sorted(c, key=lambda x: ((len(x[0]), x[0])), reverse=True)
        struc_compose = '\n'.join([str(i[1]) + ' * [' + str(i[0]) + ']' for i in sc])
        print('=============== Clusters Found ===============')
        print('{:d} clusters of size {:d} found. Their compositions are:\n{:s}'
              .format(len(all_compos), num, struc_compose))
        print('')
        return accept_cluster
        # print("--- %s seconds ---" % (time.time() - stime))

    def unwrap_cluster(self, cluster):
        '''cluster is a connected subgraph of supergraph
        return an id2pos dictionary mapping moledule id to its position in supercell
        the atom properties are not updated in this case because
        the molecular position in different cluster may be different
        the molecules in cluster will be unwraped first
        '''
        id2mol = self.S.molecules
        id2pos = {i: [0, 0, 0] for i in list(cluster.nodes())}
        # the largert molecule is keep fixed at first round
        fix_mol = [sorted([i for i in cluster.nodes()], key=lambda x: len(id2mol[x]['sn']))[-1]]
        left_node = list(cluster.nodes())
        left_node.remove(fix_mol[0])  # remove fix node from left node
        self.unwrap_mol(id2mol[fix_mol[0]])  # unwrap the fixed node
        while len(left_node) > 0:  # loop until all nodes in the cluster are fixed
            next_fix_mol = []  # initial list to store mol to fix in next round
            for fm in fix_mol:  # fm stands for fixed molecule
                # move and then fix node that are connect to the fixed node
                for mm in [i for i in cluster.neighbors(fm) if i in left_node]:  # mm for moving molecule
                    self.unwrap_mol(id2mol[mm])
                    edge = list(cluster[fm][mm].items())[0]
                    fn = [i for i in id2mol[fm]['sn'] if i in edge[0]][0]  # id atoms in fixed molecule
                    mn = [i for i in id2mol[mm]['sn'] if i in edge[0]][0]  # id atoms in moving molecule
                    fn_fcoord = np.array(self.S.get_atom([fn])[0]['fcoord']) + np.array(id2pos[fm])
                    fn_coord = self.S.frac2cart(fcoord=fn_fcoord)
                    mn_fcoord = np.array(self.S.get_atom([mn])[0]['fcoord']) + np.array(id2pos[mm])
                    mn_coord = self.S.frac2cart(fcoord=mn_fcoord)
                    displace_vect = np.array(fn_coord) - np.array(mn_coord)
                    displace_cell = np.rint(np.matmul(displace_vect, np.linalg.inv(self.S.cell_vect)))
                    if all(np.array(id2pos[mm]) == 0) and any(np.array(displace_cell) != 0):
                        id2pos[mm] = list(displace_cell)
                    elif any(np.array(displace_cell) != 0):
                        print('Warning!!! conflict moving molecule (id:{:d},formula:{:s})'
                              ' from {:s} to {:s} when unwrap this cluster ({:s})'
                              .format(mm, id2mol[mm]['formula'],
                                      ','.join([str(int(i)) for i in id2pos[mm]]),
                                      ','.join([str(int(i)) for i in displace_cell]),
                                      '-'.join([str(i) for i in cluster.nodes()]))
                              )
                    next_fix_mol.append(mm)
                    left_node.remove(mm)
            fix_mol = next_fix_mol
        return id2pos
        # print(id2pos)
        # st = Structure()
        # st.cell_param = self.S.cell_param
        # st.cell_vect = self.S.cell_vect
        # for molid in list(cluster.nodes()):
        #     for atom in copy.deepcopy(self.S.get_atom(id2mol[molid]['sn'])):
        #         atom['fcoord'] = np.array(atom['fcoord']) +  np.array(id2pos[molid])
        #         atom['coord'] = self.S.frac2cart(atom['fcoord'])
        #         st.atoms.append(atom)
        # st.complete_self(wrap=False)
        # writer=StructureWriter()
        # writer.write_file(st,name,ext='mol2')

    def unwrap_mol(self, mol):
        '''mol is a default dict in the list of self.S.molecules
        this function will add a 'unwrap' properties to atoms
        '''
        # check if the molecule is already unwraped
        if self.unwrap_mol_record[mol['id']]:
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