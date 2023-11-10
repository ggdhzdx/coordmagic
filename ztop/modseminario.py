from collections import defaultdict, OrderedDict
import numpy as np
import sys
import coordmagic as cm
import os
'''todo list
1. linear angle case: (done)
https://github.com/aa840/ModSeminario_Py/blob/master/Python_Modified_Seminario_Method/force_angle_constant.py
2. If log file do not contain internal coords, Use Multiwfn to generate (done)
3. modseminaro  sum or mean (they replied that mean is more fit to experimental value)?
4. identify equal bond length or angle for same atom and calculate average
some note:
formular is E = k(b-b0)^2 + k(a-a0)^2 + k(d-d0)^2 + k(O-O0)^2
unit for k is kcal/mol/radii^2
or kcal/mol/A^2
'''
class ModSeminario:
    def __init__(self, struct_file, hessian_file = '', vibrational_scaling=1):
        '''modified means use modified seminario method'''
        self.struct_file = struct_file
        self.hessian_file = hessian_file
        self.vibrational_scaling_squared = vibrational_scaling**2
        self.ele2num = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
                        'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
                        'K':19,'Ca':20,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,
                        'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,
                        'Rh':45,'Pd':46,'Pt':78,'Ag':47,'Au':79,'Cd':48,'Hg':80,'I':53,
                        'Ir':77,'Pb':82,'Ce':58}
        self.num2ele = dict((value,key) for key,value in self.ele2num.items())
        self.st = ''
        # generate all internal coords and corresponding measurements in four dict:
        # self.bond_s2p self.angle_s2p self.dihedral_s2p self.improper_s2p
        self.gen_internal_coords()
        if self.hessian_file:
            self.read_hessian()
            self.modified_Seminario_method()
        else:
            # if hessian_file not available set all k to zero
            self.set_nan_k()

    def set_nan_k(self):
        for _,v in self.bonds.items():
            v['k'] = np.nan
        for _,v in self.angles.items():
            v['k'] = np.nan
            v['km'] = np.nan
        for _,v in self.dihedrals.items():
            v['k'] = np.nan
            v['period'] = 1
        for _,v in self.impropers.items():
            v['k'] = np.nan

    def read_hessian(self):
        '''Function extracts xyz and hessian from the .fchk output file from Gaussian, this
        provides the coordinates,hessian of the molecules'''
        _,ext = os.path.splitext(self.hessian_file)
        if ext in ['.fch','.fchk']:
            self.hessian_from_fchk()
        else:
            self.hessian = []
        if len(self.hessian) > 0:
            hessian = np.zeros((3*self.natom,3*self.natom))
            # Change from Hartree/bohr to kcal/mol /ang
            unprocessed_hessian = np.array(self.hessian) * 627.509391 / (0.52917721067121**2)
            # Change from 1d array to low triangle matrix
            hessian[np.tril_indices(3*self.natom)] = unprocessed_hessian
            # change from low triangle matrix to symmetrical matrix
            self.hessian = np.where(hessian,hessian,hessian.T)

    def hessian_from_fchk(self):
        fchk = open(self.hessian_file, 'r')
        hessian = []
        read_hessian = 0
        #Get atomic number from fchk
        for line in fchk:
            #Gets Hessian
            if 'Cartesian Force Constants' in line:
                read_hessian = 1
                continue
            if read_hessian == 1:
                if line.strip()[:1].isalpha():
                    read_hessian = 0
                else:
                    hessian = hessian + [float(i) for i in line.split()]
        self.hessian = hessian
        if not hessian:
            print('Warning! Hessian data not extracted from fchk file {:s}\n'
                  'You need to make sure its a freq job\n'
                  'and check "Cartesian Force Constants" in fchk file'
                  .format(hessian_file))

    def gen_internal_coords(self, measure=True):
        '''This function extracts a list of bond and angles from the Gaussian .log file'''
        st = cm.read_structure(self.struct_file)
        st.G.gen_mol()
        st.G.gen_internal_coords(measure = measure)
        self.st= st
        self.natom = len(self.st.atoms)
        self.coords = np.array(st.coord)
        self.bonds = st.bonds
        self.angles = st.angles
        self.dihedrals = st.dihedrals
        self.impropers = st.impropers

    def bonds_calculated(self):
        '''This function uses the Seminario method to find the
        bond parameters and print them to file'''
        for bond in self.bonds.keys():
            a = bond[0] - 1
            b = bond[1] - 1
            AB = self.force_constant_bond(a, b)
            BA = self.force_constant_bond(b, a)
            # k_b = np.real((AB+BA)/2*self.vibrational_scaling_squared)*4.184*100*2 #for gmx
            k_b = np.real((AB+BA)/2*self.vibrational_scaling_squared)
            self.bonds[bond]['k'] = k_b

    def force_constant_bond(self, a, b):
        '''#Force Constant - Equation 10 of Seminario paper - gives force
           constant for bond
           a and b are atom index
           '''
        eigenvalues_AB = self.eigenvalues[a, b, :]
        eigenvectors_AB = self.eigenvectors[0:3,0:3, a, b]
        diff_AB = self.coords[a] - self.coords[b]
        unit_vectors_AB = diff_AB / np.linalg.norm(diff_AB)
        k_AB = 0
        for i in range(3):
            k_AB = k_AB + eigenvalues_AB[i] * abs(np.dot(unit_vectors_AB,eigenvectors_AB[:,i]))
        k_AB = -1 * k_AB/2.0
        # divided by 2 because the formular is E=k(b-b0)^2
        return k_AB

    def dihedrals_calculated(self):
        for dihed in self.dihedrals.keys():
            quadratic_k = self.force_dihedral_constant(*dihed)
            # estimate n by max neighor number
            # convert k in k(d-d0)^2 to k in k(1+cos(n*d - d0))
            period = max(self.dihedrals[dihed]['period'])
            if period > 4:
                period = 4
            k = 2*quadratic_k/period**2
            self.dihedrals[dihed]["k"] = k
            self.dihedrals[dihed]["period"] = period

    def force_dihedral_constant(self, atom_A,atom_B,atom_C,atom_D):
        a = atom_A - 1
        b = atom_B - 1
        c = atom_C - 1
        d = atom_D - 1
        diff_AB = self.coords[b] - self.coords[a]
        u_AB = diff_AB / np.linalg.norm(diff_AB)
        diff_CB = self.coords[b] - self.coords[c]
        u_CB = diff_CB / np.linalg.norm(diff_CB)
        bond_length_AB = self.calc_distance(a,b)
        v_NABC = np.cross(u_CB, u_AB)
        if np.linalg.norm(v_NABC) < 1E-4:
            return 0.0
        u_NABC = v_NABC/np.linalg.norm(v_NABC)
        diff_BC = self.coords[c] - self.coords[b]
        u_BC = diff_BC / np.linalg.norm(diff_BC)
        diff_DC = self.coords[c] - self.coords[d]
        u_DC = diff_DC / np.linalg.norm(diff_DC)
        bond_length_DC = self.calc_distance(c,d)
        v_NBCD = np.cross(u_BC, u_DC)
        if np.linalg.norm(v_NBCD) < 1E-4:
            return 0.0
        u_NBCD = v_NBCD/np.linalg.norm(v_NBCD)
        diff_BC = self.coords[c] - self.coords[b]
        eigenvalues_DC = -1*self.eigenvalues[d, c, :]
        eigenvectors_DC = self.eigenvectors[0:3, 0:3, d, c]
        eigenvalues_AB = -1*self.eigenvalues[a, b, :]
        eigenvectors_AB = self.eigenvectors[0:3, 0:3, a, b]
        k_AB = np.sum(eigenvalues_AB * np.abs(np.dot(u_NABC,eigenvectors_AB)))
        k_DC = np.sum(eigenvalues_DC * np.abs(np.dot(u_NBCD,eigenvectors_DC)))
        k_dihed = (1/(bond_length_AB**2*np.linalg.norm(v_NABC)**2*k_AB) + 
                   1/(bond_length_DC**2*np.linalg.norm(v_NBCD)**2*k_DC))**-1/2
        # divided by 2 because the formular is E=k(d-d0)^2
        return np.real(k_dihed)

    def impropers_calculated(self):
        for sn in self.impropers.keys():
            k = self.force_improper_constant(*sn)
            self.impropers[sn]["k"] = k

    def force_improper_constant(self, atom_A, atom_B, atom_C, atom_D):
        a = atom_A - 1
        b = atom_B - 1
        c = atom_C - 1
        d = atom_D - 1
        d = atom_D - 1
        diff_AB = self.coords[b] - self.coords[a]
        diff_BC = self.coords[c] - self.coords[b]
        u_BC = diff_BC / np.linalg.norm(diff_BC)
        diff_DC = self.coords[c] - self.coords[d]
        u_DC = diff_DC / np.linalg.norm(diff_DC)
        v_NBCD = np.cross(u_BC, u_DC)
        u_NBCD = v_NBCD/np.linalg.norm(v_NBCD)
        v_HA = -1*diff_AB-np.dot(-1*diff_AB,u_BC)*u_BC
        h_ABCD = np.linalg.norm(v_HA - np.dot(v_HA,u_NBCD)*u_NBCD)
        eigenvalues_AB = self.eigenvalues[a, b, :]
        eigenvectors_AB = self.eigenvectors[0:3, 0:3, a, b]
        eigenvalues_AC = self.eigenvalues[a, c, :]
        eigenvectors_AC = self.eigenvectors[0:3, 0:3, a, c]
        eigenvalues_AD = self.eigenvalues[a, d, :]
        eigenvectors_AD = self.eigenvectors[0:3, 0:3, a, d]
        k_AB = np.sum(eigenvalues_AB * np.abs(np.dot(u_NBCD,eigenvectors_AB)))
        k_AC = np.sum(eigenvalues_AC * np.abs(np.dot(u_NBCD,eigenvectors_AC)))
        k_AD = np.sum(eigenvalues_AD * np.abs(np.dot(u_NBCD,eigenvectors_AD)))
        k_AN = k_AB + k_AC + k_AD
        k = 0.5 * k_AN * h_ABCD**2 
        return k

    def angles_calculated(self):
        '''This function uses the modified Seminario method to find the angle'''
        # A structure is created with the index giving the central atom of the
        # angle, an array then lists the angles with that central atom.
        # ie. central_atoms_angles[3] contains an array of angles with central atom 3
        central_atoms_angles = OrderedDict()
        for i,a in enumerate(self.angles.keys()):
            angle_list = [[a[0],a[2],i],[a[2],a[0],i]]
            if a[1] in central_atoms_angles:
                central_atoms_angles[a[1]] += angle_list
            else:
                central_atoms_angles[a[1]] = angle_list
        # For the angle at central_atoms_angles[i] the corresponding
        # u_PA is the vector in ABC plane and perpendicular to AB, where ABC
        # corresponds to the order of the arguements
        # This is why the reverse order was also added'''
        unit_PA_all_angles = OrderedDict()
        for k,v in central_atoms_angles.items():
            unit_PA_all_angles[k] = []
            for l in v:
                unit_PA_all_angles[k].append(self.u_PA_from_angles(l[0],k,l[1]))
        # Finds the contributing factors from the other angle terms
        # Goes through the list of angles with the same central atom
        # And computes the term need for the modified Seminario method
        # if two angle in same plane share a same bond, then the scaling factor for this bond is 2
        # if two angle in perpendicular plane share a same bond, then the scaling factor for this bond is 1
        scaling_factor_all_angle = OrderedDict()
        for k,v in central_atoms_angles.items(): 
            # k is central atom serial, v this [atom_a atom_c index_in_anglelist] for k
            for i1,l1 in enumerate(v):
                additional_contributions = []
                for i2, l2 in enumerate(v):
                    if i1 != i2 and l1[0] == l2[0]:
                        additional_contributions.append(abs(np.dot(unit_PA_all_angles[k][i1],unit_PA_all_angles[k][i2]))**2)
                if len(additional_contributions) > 0:
                    #scaling_factor = 1 + np.sum(additional_contributions)
                    scaling_factor = 1 + np.mean(additional_contributions)
                else:
                    scaling_factor = 1
                if (k,l1[0],l1[1]) not in scaling_factor_all_angle:
                    scaling_factor_all_angle[(k,l1[0],l1[1])] = scaling_factor
                else:
                    print("sth wrong")
        # Finds the angle force constants with the scaling factors included for each angle
        for a in self.angles.keys():
            ABC_k, ABC_k_mod = self.force_angle_constant(a[0],a[1],a[2],scaling_factor_all_angle)
            CBA_k, CBA_k_mod = self.force_angle_constant(a[2],a[1],a[0],scaling_factor_all_angle)
            # k_theta = (ABC_k+CBA_k) * 4.184  # gmx unit
            k_theta = (ABC_k+CBA_k) / 2
            k_theta_mod = (ABC_k_mod+CBA_k_mod) / 2
            self.angles[a]['k'] = k_theta
            self.angles[a]['km'] = k_theta_mod

    def force_angle_constant(self,atom_A,atom_B,atom_C,scale_factor):
        '''Force Constant- Equation 14 of seminario calculation paper - gives force
        constant for angle (in kcal/mol/rad^2) and equilibrium angle in degrees'''
        a = atom_A - 1
        b = atom_B - 1
        c = atom_C - 1
        diff_AB = self.coords[b] - self.coords[a]
        u_AB = diff_AB / np.linalg.norm(diff_AB)
        diff_CB = self.coords[b] - self.coords[c]
        u_CB = diff_CB / np.linalg.norm(diff_CB)
        bond_length_AB = self.calc_distance(a,b)
        bond_length_BC = self.calc_distance(b,c)
        eigenvalues_AB = self.eigenvalues[a, b, :]
        eigenvectors_AB = self.eigenvectors[0:3, 0:3, a, b]
        bond_length_CB = bond_length_BC
        eigenvalues_CB = self.eigenvalues[c, b, :]
        eigenvectors_CB = self.eigenvectors[0:3, 0:3, c, b]
        # Normal vector to angle plane found
        u_N = np.cross(u_CB, u_AB)
        if np.linalg.norm(u_N) > 0.0001:
            u_N = u_N / np.linalg.norm(u_N)
            u_PA = np.cross(u_N, u_AB)
            u_PA = u_PA / np.linalg.norm(u_PA)
            u_PC = np.cross(u_N, u_CB)
            u_PC = u_PC / np.linalg.norm(u_PC)
            # Projections of eigenvalues to the norm of bond AB and bond CB
            sum_first = eigenvalues_AB * np.abs(np.dot(u_PA,eigenvectors_AB))
            sum_second = eigenvalues_CB * np.abs(np.dot(u_PC,eigenvectors_CB))
            # sum and Scaling due to additional angles - Modified Seminario Part
            k1 = np.sum(sum_first)/scale_factor[(atom_B,atom_A,atom_C)]
            k2 = np.sum(sum_second)/scale_factor[(atom_B,atom_C,atom_A)]
            k_theta =  (1/(bond_length_AB**2*k1) + 1/(bond_length_CB**2*k2))**-1
            k_theta_mod = abs(k_theta*0.5)
            # origin seminario version
            k1 = np.sum(sum_first)
            k2 = np.sum(sum_second)
            k_theta =  (1/(bond_length_AB**2*k1) + 1/(bond_length_CB**2*k2))**-1
            k_theta = abs(k_theta*0.5)
            # divided by 2 because the formular is E=k(a-a0)^2
            # Equilibrium Angle
        else:
            # rotate u_N raound u_AB by 360 and compute the average
            u_N1 = np.zeros(3)
            max_idx = np.argsort(np.abs(u_AB))[-1]
            mid_idx = np.argsort(np.abs(u_AB))[-2]
            # u_N1 is a vector that perpendicular to u_AB
            u_N1[mid_idx] = u_AB[max_idx]
            u_N1[max_idx] = -1* u_AB[mid_idx]
            u_N1 = u_N1 / np.linalg.norm(u_N1)
            # u_N2 is a unit vector that perpendicular to both u_AB and u_N1
            u_N2 = np.cross(u_N1,u_AB)
            u_N2 = u_N2 / np.linalg.norm(u_N2)
            u_N = []
            k_theta_list = []
            for theta in np.arange(0,360,1):
                u_N = np.cos(np.radians(theta))*u_N1 + np.sin(np.radians(theta))*u_N2
                u_PA = np.cross(u_N, u_AB)
                u_PA = u_PA / np.linalg.norm(u_PA)
                u_PC = np.cross(u_N, u_CB)
                u_PC = u_PC / np.linalg.norm(u_PC)
                sum_first = eigenvalues_AB * np.abs(np.dot(u_PA,eigenvectors_AB))
                sum_second = eigenvalues_CB * np.abs(np.dot(u_PC,eigenvectors_CB))
                sum_first = np.sum(sum_first)
                sum_second = np.sum(sum_second)
                k_theta =  (1/(bond_length_AB**2*sum_first) + 1/(bond_length_CB**2*sum_second))**-1
                k_theta_list.append(abs(k_theta*0.5))
                # Equilibrium Angle
            k_theta = np.mean(k_theta_list)
            k_theta_mod = k_theta
        return k_theta,k_theta_mod

    def u_PA_from_angles(self,atom_A,atom_B,atom_C):
        # This gives the vector in the plane A,B,C and perpendicular to A to B
        # A B C are atom  serial.
        diff_AB = self.coords[atom_B-1] - self.coords[atom_A-1]
        u_AB = diff_AB / np.linalg.norm(diff_AB)
        diff_CB = self.coords[atom_B-1] - self.coords[atom_C-1]
        u_CB = diff_CB / np.linalg.norm(diff_CB)
        cross = np.cross(u_CB, u_AB)
        if np.linalg.norm(cross) > 0.0001:
            u_N = cross / np.linalg.norm(cross)
            u_PA = np.cross(u_N, u_AB)
            u_PA = u_PA/ np.linalg.norm(u_PA)
        else:
            u_N = cross
            u_PA = np.cross(u_N, u_AB)
        return u_PA

    def calc_distance(self,idx1,idx2):
        return np.linalg.norm(self.coords[idx1] - self.coords[idx2])

    def gen_equal_value(self):
        pass

    def modified_Seminario_method(self):
        '''Program to implement the Modified Seminario Method
           origin matlab version Written by Alice E. A. Allen, TCM, University of Cambridge
           Reference using AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018), doi:10.1021/acs.jctc.7b00785
           python version Written by Zhong Cheng, Wuhan university'''
        # Eigenvectors and eigenvalues calculated
        eigenvectors = np.zeros((3, 3, self.natom, self.natom),dtype = 'complex_')
        eigenvalues = np.zeros((self.natom,self.natom,3),dtype = 'complex_')
        for i in range(self.natom):
            for j in range(self.natom):
                A,B = np.linalg.eig(self.hessian[i*3:i*3+3,j*3:j*3+3])
                eigenvalues[i,j,:] = A
                eigenvectors[:,:,i,j] = B
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.bonds_calculated() # generate self.bonds
        self.angles_calculated() #generate self.angles
        self.dihedrals_calculated() #generate self.dihedrals
        self.impropers_calculated() #generate self.impropers

        # self.params['coords'] = self.coords
