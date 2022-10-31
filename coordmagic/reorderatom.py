import networkx as nx
import numpy as np
import coordmagic as cm

__all__ = [
    'snr2l',
    'snl2r'
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

def snr2l(snr):
    '''input a sn range str, return a sn list'''
    l = []
    for i in snr.split(','):
        if i.isdigit():
            l.append(int(i))
        elif '-' in i and len(i.split('-'))==2:
            s,e = [int(x) for x in i.split('-')]
            l += list(range(s,e+1))
        else:
            print('Error!!! Can not parse sn range str {:s}'.format(snr))
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



