import networkx as nx
import numpy as np
import coordmagic as cm
import copy
import sys

__all__ = [
    'snr2l',
    'snl2r',
    'sort_atom_by_sn',
    'sort_atom_by_mol',
]


def compute_atom_score(mol):
    G=mol['graph']
    # find the length of longest shorest path in the graph
    n1 = list(nx.bfs_layers(G,1))[-1][-1]
    mol_len = len(list(nx.bfs_layers(G,n1)))
    init_score = {n:a['atomnum'] * np.float64(2**(mol_len/2)) for n,a in G.nodes(data=True)}
    add_score = {}
    current_score = init_score
    print(mol_len)
    for i in range(mol_len):
        for n in G.nodes():
            add_score[n] = sum([current_score[i] for i in G.neighbors(n)])/4
        for i,s in add_score.items():
            current_score[i] += s

    print({G.nodes()[i]['sn']:s for i,s in current_score.items()})


    #     if a['atomnum'] == 1:
    #         a['topo_score'] = a['atomnum'] * 10000
    # for n,a in G.nodes(data=True):
    #     a['topo_score']
def sort_atom_by_sn(struct,sn_list):
    '''sort atoms in struct by sn list and return a new structure'''
    s = copy.deepcopy(struct)
    s.atoms = [s.atoms[i-1] for i in sn_list]
    s.complete_coord()
    s.reset_sn()
    return s

def sort_atom_by_mol(struct):
    '''sort atoms in struct by mol
    However, the atom order in each mol is not controlled
    and the order of mol is not controlled
    '''
    if len(struct.molecules) == 0:
        struct.graph.gen_mol()
    newsn = []
    for id, mol in struct.molecules.items():
        newsn += mol['sn']
    new_st = sort_atom_by_sn(struct, newsn)
    return new_st

def snr2l(snr,total=0,complement=False):
    '''input a sn range str, return a sn list
    of string is lead by L or l, means last n
    l1 means the last atom, l3 means the last 3 atom
    if comlement == True， return complement serial list from total
    '''
    def parse_sn(n,total):
        '''conter str to int
        and convert ln (count from last) to forward order sn'''
        if n.isdigit():
            sn = int(n)
        elif n.startswith('l') or n.startswith('L'):
            sn = total - int(n[1:]) + 1
        else:
            sys.exit('Error!!! Can not parse sn str {:s}'.format(n))
        if sn <= 0:
            sys.exit('Error!!! molecule serial {:d} from {:s} small than 1'.format(sn,n))
        return sn
    l = []
    for i in snr.split(','):
        if '-' in i and len(i.split('-'))==2:
            i1 = parse_sn(i.split('-')[0],total=total)
            i2 = parse_sn(i.split('-')[1],total=total)
            if i1 > i2:
                i1, i2 = i2, i1
            l = l + list(range(i1, i2 + 1))
        else:
            l.append(parse_sn(i,total=total))
    if complement:
        remain_atom_sn = [i+1 for i in range(total) if i+1 not in l]
        l = remain_atom_sn
    return  l


def snl2r(lst):
    '''input a list of int, return a sn range'''
    s = e = None
    r = []
    for i in sorted(lst):
        if s is None:
            s = e = i
        elif i == e or i == e + 1:
            e = i
        else:
            if e > s:
                r.append("-".join([str(i) for i in [s,e]]))
            else:
                r.append(str(s))
            s = e = i
    if s is not None:
        if e > s:
            r.append("-".join([str(i) for i in [s,e]]))
        else:
            r.append(str(s))
    return ','.join(r)


# st=cm.read_structure("mbb.gjf")
# st.G.gen_mol()

# compute_atom_score(st.molecules[1])



