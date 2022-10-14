import networkx as nx
import numpy as np
import coordmagic as cm

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






st=cm.read_structure("mbb.gjf")
st.G.gen_mol()

compute_atom_score(st.molecules[1])



