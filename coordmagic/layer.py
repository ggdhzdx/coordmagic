
class Layer:

    def __init__(self, structure):
        self.S = structure

    def gen_layers(self, layer_th=0.1):
        '''chechk all layers in the z direction
        atoms are in the same layer if their z coordinate difference
        do not larger than layer_th'''
        s = self.S.sort_atoms(by='z')
        layers = []
        for i, c in enumerate(s.coord):
            if len(layers) == 0:
                L = Structure()
                L.cell_param = s.cell_param
                L.basename = s.basename
                L.add_atom(s.atoms[i])
                layers.append(L)
            elif abs(layers[-1].top - c[2]) < layer_th:
                layers[-1].add_atom(s.atoms[i])
            else:
                L = Structure()
                L.cell_param = s.cell_param
                L.basename = s.basename
                L.add_atom(s.atoms[i])
                layers.append(L)
        self.S.layers = layers

    def shift_btn_layer(self, z=0.1):
        '''shift button layer to z. Button layer is defined as having the largest
        distance  to the layer below it '''
        s = copy.deepcopy(self.S)
        s.L.check_layer(z_th=0.01, adist=False)
        top_idx = s.layer_data['z-dist'].idxmax()
        if top_idx < len(s.layer_data.index)-1:
            zpos = s.layer_data['z-pos'].iloc[top_idx+1]
        else:
            zpos = s.layer_data['z-pos'].iloc[0]
        dz = z - zpos
        return s.T.shift_xyz([0, 0, dz], pbc=True)

    def extract_layer(self, layer_idx):
        self.S.L.gen_layers()
        s = copy.deepcopy(self.S)
        s.atoms = []
        for i in layer_idx:
            s.atoms = s.atoms + self.S.layers[i].atoms
        s.complete_self()
        return s

    def check_layer(self, z_th=0.2, a_th=5, bw=0.2, adist=True):
        '''view the cell as layered structure and print infor of each layer
        layers' z coord speration smaller than z_th will be viewed as one layer
        interlayer atomic distance smaller than a_th will be printed
        '''
        # tobe develop for large system print layers one by one
        # for molecule system
        # this function convert atom distance dataframe to strings
        s = self.S

        def atom_dist_str(dist_df):
            count = []
            for _, df in dist_df.groupby('atoms'):
                dl = []
                for _, row in df.loc[df['dist'] < a_th].iterrows():
                    dl.append('{:.2f}'.format(row['dist'])+'*' +
                              '{:d}'.format(int(row['count'])))
                if len(dl) == 0:
                    row = df.iloc[0]
                    dl.append('{:.2f}'.format(row['dist'])+'*' +
                              '{:d}'.format(int(row['count'])))
                count.append(df['atoms'].iloc[0]+':'+','.join(dl))
            return ';'.join(count)

        # ps_up is the unit cell that on the top of the origin cell
        s_up = s.C.shift_cell_origin([0, 0, 1])
        s.L.gen_layers(layer_th=z_th)
        s_up.L.gen_layers(layer_th=z_th)
        # icz is the inter cell z distance
        icz = s_up.layers[0].button - s.layers[-1].top
        # ica is the inter cell cell atom distance
        formula = []
        dist = []
        pos = []
        thickness = []
        atom_dist = []
        for i, l in enumerate(s.layers):
            if i > 0:
                dist.append(l.button-s.layers[i-1].top)
            formula.append(l.formula)
            thickness.append(l.thick)
            pos.append(l.z)
        dist.append(icz)
        layers = pd.DataFrame({'formula': formula, 'z-dist': dist, 'z-pos': pos,
                               'thickness': thickness})
        t1 = datetime.now()
        if adist:
            # ps_ext is the unit cell with surrounding image atoms.
            s_ext = s.C.add_image_atom([10, 10, 0])
            s_ext.L.gen_layers(layer_th=z_th)
            ica = atom_dist_str(s_up.layers[0].dist_to(s_ext.layers[-1], bw=bw))
            for i, l in enumerate(s.layers):
                if i > 0:
                    atom_dist.append(atom_dist_str(l.dist_to(s_ext.layers[i-1], bw=bw)))
            atom_dist.append(ica)
            layers.loc[:, 'atom_dist'] = atom_dist
        t2 = datetime.now()
        self.S.layer_data = layers