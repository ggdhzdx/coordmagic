#!/usr/bin/env python3
"""multiple file format converter"""

import argparse
import os

import coordmagic as cm

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description='various structure format converter')
parser.add_argument("-o", dest='outfmt', metavar='output_format', default='', type=str,
                    help='Set output format\n')
parser.add_argument("-i", dest='inpfmt', metavar='input_format', default='', type=str,
                    help='Set input format only if not inferred from file suffix\n')
parser.add_argument("-m", dest='mols', metavar='molecules', default='', type=str,
                    help='set the mol index to extract\n'
                         'separate multiple indexes by comma\n'
                         'If no index is set, i will list all mols with their index\n')
parser.add_argument("-O", dest='outnm', metavar='output_name', default='', type=str,
                    help='Set output file basename only if different from input file basename.\n')
parser.add_argument("-s", dest='split', metavar='split', default='last', type=str,
                    help='-s each : split input into multiple files if it contains multiple structures\n'
                         '-s last(default) : convert last frame\n'
                         '-s all : convert all frame to one multi-frame file,\n'
                         'resort to "-s each" if output format do not support multiframe\n')
parser.add_argument('-c', dest='combine', action='store_true',
                    help='convert multiple input file to one multi-frame file\n')
parser.add_argument('-B', dest='base', action='store_true',
                    help='convert multiple input file to one multi-frame file\n')
parser.add_argument('--bondth', dest='bondth', metavar="bond_threshold", default="",type=str,
                     help='set custom bond threshold between two elements\n'
                          'the format is like C-H=1.0,C-S=1.7\n')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
parser.add_argument('inputfile', nargs='+', help='the input structures can be gjf, mol, xyz, Gaussian log\n')
args = parser.parse_args()


all_struct = []
all_name = []
for f in args.inputfile:
    st = cm.read_structure(f, filetype=args.inpfmt)
    print("Reading: "+str(st))
    if args.mols == "list":
        st.graph.set_threshold(usrcovth=args.bondth)
        st.graph.gen_mol()
        print(st.mol_df[['sn_range','formula']])
    if not args.outnm:
        outnm = os.path.splitext(os.path.basename(f))[0]
    else:
        outnm = args.outnm
    all_name.append(outnm)
    all_struct.append(st)
if args.combine:
    st.concat_struct([all_struct],self_pos=None)
elif args.outfmt or args.outnm:
    for i,st in enumerate(all_struct):
        st.save(format=args.outfmt, name=all_name[i], frame=args.split)
