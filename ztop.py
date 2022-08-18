#!/usr/bin/env python
'''todo
report more after paramtransfer
param average by structure match
set resname by atom match
combine frag with multiple dihedrals
change pair_nb when fit torsion
list res lib and pickle them for later use (done)
paramrefine do not need log file contain internal coords (done)
internal coords should be generated in coord magic (done)
build whole system
support libpargen and intermol
preview a torsion profile
partial param transfer
eq param transfer
none,read,XXX.chg options for -g q=xxx
template gjf input file
add total fragment charge in lib table 
do a small md to verify parameter stability
write command history
'''

from ztop import *
import os
import sys
import argparse
import coordmagic as cm
import shutil
from collections import defaultdict
import timeit


parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                               description='refine topology parameters by gaussian fchk and log file\n'
                               'combine small molecules to large ones')
parser.add_argument("-g",dest='gentop',metavar='gen_top',default='',type=str,
                    help='generate top from a structure file, by third party program \n'
                         'the format and some default value is:\n'
                         '-g "input.log;charge=0;spin=1;ff=gaff2;q=resp;ffp=amber;qp=multiwfn;charge_only"\n'
                         'default forcefield is gaff2 and default q is resp by multiwfn\n'
                         'recommanded file formats are: log, mol2, pdb\n'
                         'supported ff(force field) are:\n'
                         'gaff2, gaff, amber(requires antechamber)\n'
                         'opls-aa (requires libpargen, not available for now)\n'
                         'SMIRNOFF (requires openff, not available for now) \n'
                         'available q(charge methods) are: \n'
                         'resp resp2 am1-bcc gasteiger cm5 ddec3 ddec6 \n'
                         'ffp and qp are program for parameter generation and change generateion\n'
                         'Normally they will set automatically according to charge type and ff type\n')
parser.add_argument("-f",dest='fragment',metavar='fragment',action='append',default=[],
                    help='Define a fragment, the formats are:\n'
                         'D;p=ABC.top;x=ABC.gro;site=D1:2-3,D2:11-23;resname=DON;comment="any words"\n'
                         'The first letter D is the label of this residue used only in this program\n'
                         'p=,x= set the top and coord file of the residue\n'
                         'site= define cutting site of the residue. D1 and D2 are site label,\n'
                         'The second to last character of site label are used for site match\n'
                         'following the shortest match rule: e.g. D12 will match A1 or A12 but not A13\n'
                         'X542 will match Y5,Y54,Y542,Y5421 but not X6, X55, X543\n'gi
                         'D1:2-3 define site by breaking the bond 2-3 where atom 2 belongs to the residue.')
parser.add_argument("-b",dest='build',metavar='build',default='',type=str,
                    help = 'set a list of res label seperated to build the molecule. \n'
                    'res label in bracket followed by number n [x]n will be repeated n times. e.g.\n'
                    'H[AD]5T is a six mer\n'
                    'C[L]3[M]6[N]12 is a dendrimer of three generation\n'
                    'Note that you should first define residue by -r option')
parser.add_argument("-p",dest='top',metavar='initial_top',default='',type=str,
                    help='initital top file\n'
                         '"-p ABC.prmtop" or "-p ABC.top"')
parser.add_argument("-x",dest='coord',metavar='initial_coord',default='',type=str,
                    help='initial coord file\n'
                         'support format:\n'
                         'pbd, mol2, gro, inpcrd, gro')
parser.add_argument("-r",dest='refine',metavar='refine_param',default='',type=str,
                    help='refine parameters, the format is:\n'
                          '-r "e=ABC.log;k=ABC.fchk;c=ABC.chg;b=replace,k;a=replace,k,ms;d=add,e;i=omit,5" \n'
                          'e= to set structure file, needed for refine equilibrim paramter\n'
                          'k= to set fchk file with Hessian matrix, needed for force constant calculation\n'
                          'c= to set chg file, the last column of which will be read as charge\n'
                          'b=/a=/d=/i= to set action for bond/angle/dihedral/improper parameters, respectively\n'
                          'actions are: replace(all paramters), add(missing parameters only), omit(do nothing)\n'
                          'e(only change equlibrim parameter), k(change force constants if available)\n'
                          'ms/origin(modified/origin seminarios method for angle parameters)\n'
                          '5(improper dihedrals smaller than 5 degree will be added if action is not omit)\n'
                          '-r "e=ABC.log" only do equilibrium bond and angle refinement\n'
                          '-r "e;k;c" will refine params from default file\n')
parser.add_argument("-d",dest='dihedral',metavar='dihedral_fit',default='',type=str,
                    help='fit a torsion parameter, the format is:\n'
                         '-d "15-16;angle=e5,360,0;period=1,2,3,4;eth=10;opt=no;\n'
                         'ar=-30~30,150~180;nd=3;kth=0.02;regen=gjf;ene=g"\n'
                         '15-16 specify a bond to rotate and dihedral of which to fit\n'
                         'angle:e5,180,0 rotate dihedral by 0 to 180 with increment of 5\n'
                         'period:1,2,3,4 fit the dihedral with period of 1 2 3 and 4, default is auto\n'
                         'eth:60 fit the point with energy lower than 60kJ/mol\n'
                         'opt:no do not add constrain information to the end of generated gjf file\n'
                         'opt:fixone fix one dihedral, and opt:fixall fix all dihedrals\n'
                         'ar:-30~30 select angle range from -30 to 30 to fit\n'
                         'The purpose of angle range is to aviod large steric to intefere torsion param\n'
                         'nd:4,2 indicate fit first 4 dihedrals and then fit first 2 dihedrals\n'
                         'nd:1.3,2.4 indicate fit 1st and 3rd dihedrals and then fit first 2nd and 4th dihedrals\n'
                         'remember add opt(modredundant) to the keywords of gjf by --gjf options\n'
                         'kth:0.02 indicate that k value below this value will be ignored (unit: kCal/mol)\n'
                         'regen:gjf will regenerate gjf file\n' 
                         'regen:xyz will regenerate xyz file from log\n'
                         'ene=g/ex will read ground/excited state energy, default will read excited state energy if available\n'
                         'The program will first check a xyz file named ABC_DFit_15-16.xyz in current directory\n'
                         'In this file the comment line is the energy of each structure in kJ/mol\n'
                         'If the xyz file is not found,\n'
                         'The program will read log files in ABC_TSFit_15-16 and genererate the xyz file\n'
                         'If the directory is not found, the program will create it and the gjf files\n'
                         'The user should get the log file by themselves' )
parser.add_argument("-t",dest='transfer',metavar='transfer_param',action='append',default=[],type=str,
                    help = 'transfer parameters from a residue that define by -r\n'
                    'the format is:\n'
                    'd;m=12-45,73,9;a=40,43,9-12;p=be,ae,d,i\n'
                    'The D is the residue label\n'
                    'm= designate the atom serial in D that used to match substruct in main structure\n'
                    'a= designate the parameter of which atoms will be transfered.\n'
                    'p= b,a,d,i,c,v specifed which paramters will be transfers\n'
                    'c:charge,v:vdw,b:bond,a:angle,d:dihedral,i:improper\n'
                    'e or k could be appended to b/a/d/i to transfer only e or only k\n')
# parser.add_argument("-a",dest='average',metavar='average_param',default='',type=str,
#                     help = 'average parameters for substructrures\n'
#                        'the format is:\n'
#                        'm=12-45,73,9;a=40,43,9-12;p=d\n'
#                        'm= designate the atom serial used to match substructure\n'
#                        'a= designate the parameter of which parameters will be averaged.\n'
#                        'p= cbadv specifed which paramters will be transfers\n'
#                        'c:charge,v:vdw,b:bond,a:angle,d:dihedral')
parser.add_argument("-o",dest='output',metavar='output',default='',type=str,
                    help='set output filename and file extension. For example:\n '
                         '"-o ABC.top,ABC.gro" generate file for gromacs\n'
                         '"-o ABC.prmtop,ABC.inpcrd" generate file for amber\n')
# parser.add_argument("--resname",dest='resname',metavar='resname',default='',type=str,
#                     help='set the residue name of the structure\n'
#                          'seprate multiple names by comma')
parser.add_argument("--savelib",dest='savelib',action="store_true",
                    help='save the current fragment lib to directory FRAG_LIB\n')
parser.add_argument("--loadlib",dest='loadlib',action="store_true",
                    help='load the fragment lib from directory FRAG_LIB\n')
parser.add_argument("--gjf",dest='gjf',metavar='gjf_option',default='',type=str,
                    help='can set some gjf file options. For example:\n '
                         '"--gjf "nproc:8;mem:4GB;charge:0;spin:1;extra:eps=3.0;vdw:em(gd3bj)\n'
                         'method:b3lyp;basis:def2svp;solvent:none;addkey:addition keywords"\n')
parser.add_argument("--opt",dest='opt',action='store_true',
                    help='If this option is set, then the output geom will optimized by openmm\n')
parser.add_argument("--total_charge",dest='total_charge',type=int,default=-10000,
                    help='Set the total_charge of the system. The current atom charge will be adjusted evenly to reach this value\n')
parser.add_argument("--saveinfo",dest='saveinfo',type=str,default='',
                    help='which info to save when do param refine or param transfer\n'
                    'badictv for bond angle dihedral improper charge type vdw, respectively\n')
parser.add_argument("--noopenmm",dest='noopenmm',action='store_true',
                    help='not use openmm when do param refine or transfer\n')
parser.add_argument("--checkenv",dest='checkenv',action='store_true',
                    help='check the program and python package needed by this program\n')
parser.add_argument('--version', action='version', version='%(prog)s 1.1')
args=parser.parse_args()
# st=cm.read_structure('Dfrag.log')
# st.M.gen_molecules()
# D=st.M.gen_frag_by_breakbond(D1='18-36',D2='4-27')
# A0=st.M.gen_frag_by_breakbond(A2='27-4')
# sta = cm.read_structure('Afrag.log')
# sta.M.gen_molecules()
# A=sta.M.gen_frag_by_breakbond(A1='20-22',A2='14-4')
# D0=sta.M.gen_frag_by_breakbond(D2='22-20')
# frag1 = st.M.connect_frag(D,A,joinsite='D-A')
# # print(frag1)
# frag2 = st.M.connect_frag(frag1,D,joinsite='A-D')
# frag3 = st.M.connect_frag(frag2,A,joinsite='D-A')
# frag4 = st.M.connect_frag(frag3,D,joinsite='A-D')

pr = None
rc = None

if args.checkenv:
    print('check parmed...',end='')
    try:
        import parmed
    except:
        print('\nError!!! python package parmed not available')
    else:
        print('parmed version {:s} detected'.format(parmed.__version__))
    print('check networkx...',end='')
    try:
        import networkx
    except:
        print('\nError!!! python package networkx not available')
    else:
        print('networkx version {:s} detected'.format(networkx.__version__))
    print('check Multiwfn...',end='')
    multiwfn = shutil.which('Multiwfn')
    multiwfn_win = shutil.which('Multiwfn.exe')
    if not (multiwfn or multiwfn_win):
        print('\nError!!!  Multiwfn is not available')
    else:
        print('success!')
    print('check AmberTools...',end='')
    antechamber = shutil.which('antechamber')
    parmchk2 = shutil.which('parmchk2')
    tleap = shutil.which('tleap')
    p2p = {'antechamber': antechamber,
           'parmchk2': parmchk2,
           'tleap': tleap}
    if not (antechamber and parmchk2 and tleap):
        print('\nError! AmberTools program({:s}) not available'
              .format(','.join([i for i, v in p2p.items() if not v])))
    else:
        print('success!')
    print('check openmm...',end='')
    try:
        import openmm.testInstallation as TI
    except:
        print('Error!!! python package openmm not available')
    else:
        TI.main()

if args.loadlib:
    # print(args.residue)
    rc = ResidueComposer()
    rc.load_lib()
    rc.print_lib()

if len(args.fragment) > 0:
    # print(args.residue)
    rc = ResidueComposer()
    for r in args.fragment:
        rc.parse_residue(r)
    rc.print_lib()

if rc:
    if args.savelib:
        rc.save_lib()
    if args.build:
        mol=rc.compose_mol(args.build)
        final_top = mol['pmd_struct']
        pr = ParamRefine(pmd_struct=final_top)
        if args.total_charge != -10000:
            pr.set_charge(args.total_charge)
        # for d in pr.ps.dihedrals:
        #     if d.type.scee > 100:
        #         print(d)

if args.gentop:
    inpfile = args.gentop.split(';')[0]
    kwarg = {v.split('=')[0]:v.split('=')[1] for v in args.gentop.split(';') if len(v.split('='))==2}
    kwarg['gjf_options'] = args.gjf
    pg = ParamGen(inpfile,**kwarg)
    if "charge_only" in args.gentop:
        pg.gen_param(charge_only=True)
    else:
        top,crd,chg = pg.gen_param()
        pr = ParamRefine(top_file = top,saveinfo=args.saveinfo)
        pr.load_coord(crd)

if args.top:
    pr = ParamRefine(top_file = args.top,saveinfo=args.saveinfo)


if pr:
    if args.saveinfo:
        pr.saveinfo = args.saveinfo
    if args.coord:
        pr.load_coord(args.coord)
        st = cm.conver_structure(pr.ps,'parmed')
    if args.refine:
        rf = args.refine.split(';')
        rf_task = defaultdict(str,{i.split('=')[0]:'' for i in rf})
        rf_task.update({i.split('=')[0]:i.split('=')[1] for i in rf if len(i.split('=')) == 2})
        badi_conf = {k:v for k,v in rf_task.items() if k in ['b','a','d','i']}
        if 'c' in rf_task:
            if not rf_task['c']:
                rf_task['c'] = pr.basename + '.chg'
            if not os.path.isfile(rf_task['c']):
                print('Error! charge file {:s} not found in current directory'.format(rf_task['c']))
                sys.exit()
            print('Statistics on charge refinement:')
            pr.refine_charge(rf_task['c'])
        # find default structure file
        if 'e' in rf_task:
            if not rf_task['e']:
                available_ext = ['.log','.fchk','.fch']
                exist_file = [pr.basename + i for i in available_ext if os.path.isfile(pr.basename+i)]
                exist_file_sol = [pr.basename + '_sol' +  i for i in available_ext if os.path.isfile(pr.basename+'_sol'+i)]
                exist_file_gas = [pr.basename + '_gas' +  i for i in available_ext if os.path.isfile(pr.basename+'_gas'+i)]
                all_exist = exist_file_sol + exist_file + exist_file_gas
                if len(all_exist) >= 1:
                    rf_task['e'] = all_exist[0]
                else:
                    print('Error! {:s} file with basename {:s} or {:s} or {:s} not found in current directory'
                          .format('/'.join(available_ext),pr.basename, pr.basename+"_sol",pr.basename+"_gas"))
                    sys.exit()
            if not os.path.isfile(rf_task['e']):
                print('Error! Structure file {:s} not found in current directory'.format(rf_task['e']))
                sys.exit()
        # find default hessian file
        if 'k' in rf_task:
            if not rf_task['k']:
                available_ext = ['.fchk','.fch']
                exist_file = [pr.basename + i for i in available_ext if os.path.isfile(pr.basename+i)]

                if len(exist_file) >= 1:
                    rf_task['k'] = exist_file[0]
                else:
                    print('Error! {:s} file with basename {:s} not found in current directory'
                          .format('/'.join(available_ext),pr.basename))
                    sys.exit()
            if not os.path.isfile(rf_task['k']):
                print('Error! Hessian file {:s} not found in current directory'.format(rf_task['k']))
                sys.exit()
        if 'e' in rf_task:
            if 'k' in rf_task:
                print('Read equilibrium bond and angle from {:s}'.format(rf_task['e']))
                print('Read force constant from {:s}'.format(rf_task['k']))
                ms = ModSeminario(rf_task['e'], rf_task['k'])
            else:
                print('Read equilibrium bond and angle from {:s}'.format(rf_task['e']))
                ms = ModSeminario(rf_task['e'])
            if not pr.ps.positions:
                pr.load_coord(rf_task['e'])
            if not args.noopenmm:
                coord1,ene1=pr.minimize()
                report1=pr.compare_structure(ms.coords,coord1)
            pr.refine_badi(ms, **badi_conf)
            if not args.noopenmm:
                coord2,ene2=pr.minimize()
                report2=pr.compare_structure(ms.coords,coord2)
                print('Before parameter refine (openmm opt vs input):')
                print('Energy: {:.3f} kJ/mol'.format(ene1._value))
                print(report1)
                print('After parameter refine (openmm opt vs input):')
                print('Energy: {:.3f} kJ/mol'.format(ene2._value))
                print(report2)
    if len(args.transfer) > 0:
        for t in args.transfer:
            res = t.split(';')[0]
            param = {i.split('=')[0]:i.split('=')[1] for i in t.split(';')[1:]}
            if not args.noopenmm:
                s = timeit.default_timer()
                coord1, ene1 = pr.minimize()
                e = timeit.default_timer()
                print("first minimize took {:.3f} seconds".format(e-s))
            pr.transfer_param_from(rc.res_lib[res],**param)
            if not args.noopenmm:
               s = timeit.default_timer()
               coord2, ene2 = pr.minimize()
               e = timeit.default_timer()
               print("Second minimize took {:.3f} seconds".format(e-s))
               report = pr.compare_structure(coord1,coord2)
               print('Energy before transfer: {:f} kJ/mol; Energy after transfer {:f} kJ/mol'
                    .format(ene1._value,ene2._value))
               print('Min structure difference after parameter transfer:')
               print(report)
    if args.dihedral:
         bond = args.dihedral.split(';')[0]
         param = {i.split('=')[0]:i.split('=')[1] for i in args.dihedral.split(';')[1:]}
         if args.gjf:
             param.update({'gjf':args.gjf})
         TorsionFit(pr,bond,**param)
    if args.total_charge != -10000:
        pr.set_charge(args.total_charge)
    if args.opt:
         coords,ene = pr.minimize()
         pr.ps.positions = coords/10
         print('the mm energies after minimize is {:.3f} kJ/mol'.format(ene._value))
    if args.output:
        options = {}
        for f in args.output.split(','):
            pr.adj_dihedral(contract= 'all')
            if f.endswith('.gro') and not pr.ps.box: 
                options['nobox'] = True
            pr.ps.save(f,overwrite=True,**options)


