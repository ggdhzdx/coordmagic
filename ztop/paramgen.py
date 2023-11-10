from collections import defaultdict
import sys
import os
import shutil
import subprocess
from coordmagic.parameter import Parameter
sys.path.append('..')
import coordmagic as cm
import pandas as pd

class ParamGen:
    '''generate init parm by third party program'''

    def __init__(self, inputfile, charge="0", spin="1", inputfiletype='',
                       ff='', ffp='', q='', qp='', gjf_options=''):
        self.inputfile = inputfile
        self.inputfiletype = inputfiletype
        self.fftype = ff
        self.ffprogram = ffp
        self.qtype = q
        self.qprogram = qp
        self.charge = charge
        self.spin = spin
        self.ff2p = {
            'gaff': 'amber',
            'gaff2': 'amber',
            'amber': 'amber',
            'opls': 'libpargen',
            'smirnoff': 'openff'
        }
        self.q2p = {
            'resp': 'multiwfn',
            'resp2': 'multiwfn',
            'am1-bcc': 'amber',
            'cm5': 'multiwfn',
            'gasteiger': 'amber',
            'ddec3': 'chargemol',
            'ddec6': 'chargemol'
        }
        self.ext2file = {
            '.log': 'gau',
            '.out': 'orca',
            '.mol2': 'mol2',
            '.pdb': 'pdb',
            '.xyz': 'xyz',
            '.gjf': 'gjf'
        }
        self.check_program = {
            'amber': self.__check_amber,
            'multiwfn': self.__check_multiwfn
        }
        # 'chargemol':self.__check_chargemol,'openff':self.__check_openff,
        # 'libpargen':self.__check_libpargen}
        self.run_chg_program = {
            'amber': self.__run_amber,
            'multiwfn': self.__run_multiwfn
        }
        # 'chargemol': self.__run_chargemol}
        self.run_ff_program = {'amber': self.__run_amber}
        self.gjf_options = defaultdict(str, {v.split(':')[0]: v.split(':')[1]
                                             for v in gjf_options.split(';') if ':' in v})

        # ,'openff': self.__run_openff,
        #                         'libpargen': self.__run_libpargen}

    def gen_param(self, charge_only=False):
        '''first, initialized all parameters
        second, check if required program are available
        third, run program
        required input file are checked in the __run_program function
        '''
        self.charge_only = charge_only
        filename = os.path.basename(self.inputfile)
        self.basename, ext = os.path.splitext(filename)
        if not self.inputfiletype:
            self.inputfiletype = self.ext2file[ext]
        if not self.fftype:
            self.fftype = 'gaff2'
        if not self.ffprogram:
            self.ffprogram = self.ff2p[self.fftype]
        if not self.qtype:
            self.qtype = "resp"
            qtype = "resp"
        else:
            qtype_list = [i for i in self.q2p.keys() if i == self.qtype or i == self.qtype.split('-')[0]]
            if len(qtype_list) > 1:
                print("Warning! More than one qtype available: {:s}. I will use {:s} here."
                .format(" ".join(qtype_list), qtype_list[0]))
                qtype = qtype_list[0]
            elif len(qtype_list) == 1:
                qtype = qtype_list[0]
            else:
                print("Error! qtype {:s} not available".format(self.qtype))
                sys.exit()
        if not self.qprogram:
            self.qprogram = self.q2p[qtype]
        if charge_only:
            print("Use {:s} to generate {:s} charge file"
                  .format(self.qprogram, self.qtype))
            if not self.check_program[self.qprogram](qtype):
                sys.exit()
            if not self.run_chg_program[self.qprogram](self.fftype, self.qtype):
                sys.exit()
            else:
                return ["","",self.chgfile]
        else:
            print("Use {:s} to generate {:s} charge and use {:s} to generate {:s} forcefield parameter"
                  .format(self.qprogram, self.qtype, self.ffprogram, self.fftype))
            if not self.check_program[self.qprogram](qtype):
                sys.exit()
            if not self.check_program[self.ffprogram](self.fftype):
                sys.exit()
            if self.qprogram != self.ffprogram:
                if not self.run_chg_program[self.qprogram](self.fftype, self.qtype):
                    sys.exit()
            if not self.run_ff_program[self.ffprogram](self.fftype, self.qtype):
                sys.exit()
            else:
                return [self.topfile, self.crdfile, self.chgfile]

    def __check_multiwfn(self, qfftype):
        multiwfn = shutil.which('Multiwfn')
        multiwfn_win = shutil.which('Multiwfn.exe')
        if not (multiwfn or multiwfn_win):
            print('Error! {:s} requires Multiwfn, but it is not available'.
                  format(qfftype))
            return False
        else:
            return True

    def __check_amber(self, qfftype):
        antechamber = shutil.which('antechamber')
        parmchk2 = shutil.which('parmchk2')
        tleap = shutil.which('tleap')
        p2p = {
            'antechamber': antechamber,
            'parmchk2': parmchk2,
            'tleap': tleap
        }
        if not (antechamber and parmchk2 and tleap):
            print('Error! {:s} requires Amber({:s}), and it is not available'.
                  format(qfftype,
                         ','.join([i for i, v in p2p.items() if not v])))
            amber_avail = False
        else:
            amber_avail = True
        return amber_avail

    def __check_gau_log(self, filename):
        '''check gaussian log file '''
        try:
            gout = open(filename + '.log', 'r')
        except FileNotFoundError:
            return 0
        file_usable = 0
        for l in gout:
            if 'Electrostatic Properties' in l:
                file_usable = 1
                return 2
        if file_usable == 0:
            return 1

    def __check_gau_fchk(self, filename):
        '''check gaussian log file '''
        if os.path.isfile(filename + '.fchk'):
            return 1
        if os.path.isfile(filename + '.chk'):
            print('{:s} is not available but found {:s}, try to formchk it'.
                  format(filename + '.fchk', filename + '.chk'))
            return_code = subprocess.call(['formchk', filename + '.chk'])
            if return_code != 0:
                print('Error! formchk {:s} failed!'.format(filename + '.chk'))
                return 0
            else:
                return 1
        else:
            return 0
    def __gen_parmchk_data(self):
        '''generate additional atom type corresponding score file PARMCHK.DAT
        which will be used in parmchk2 -atc option
        '''
        P=Parameter()
        exist_elem = ['C','H','O','N','F','Cl','Br','I','S','P']
        f = open('PARMCHK.DAT','w')
        for k,v in P.elem2mass.items():
            if k not in exist_elem: 
                f.write('{:8s}{:8s} 0 0 {:f} 0 {:d}\n'.format('PARM',k,v,P.elem2an[k]))
        f.close()


    def __run_amber(self, fftype, qtype):
        '''first check required input file
         if not available, generate the input file for quantum chemistry program
         '''
        ''' if resp charge is requested then check if correct log file is available
        if not avail generate a gjf file
        '''
        cwd = os.getcwd()
        if qtype == 'resp' and self.qprogram == 'amber':
            gau_log = self.__check_gau_log(self.basename)
            if gau_log == 0:
                print('Gaussian log file {:s} not found in {:s}.\nI will generate {:s} file here.\n'
                      'The default level is B3LYP/def2SVP em=GD3BJ for opt and charge.\n'
                      'You may need to modify and submit it to Gaussian09/16 to get the log file'
                       .format(self.basename + '.log', cwd, self.basename + '.gjf'))
                self.gen_gjf(name=self.basename + '.gjf', profile='resp')
                return False
            elif gau_log == 1:
                print('Gaussian log file {:s} in {:s} do not contain Electrostatic information. '
                      'I will generate a gjf file {:s} here'
                      .format(self.basename + '.log', cwd, self.basename + '.gjf'))
                self.gen_gjf(name=self.basename + '.gjf', profile='resp')
                return False
            elif gau_log == 2:
                print('Gaussian log file {:s} with Electrostatic information found.\n'
                      'Antechamber will use this file to generate resp charge.'
                      .format(self.basename + '.log'))
            self.inputfile = self.basename + '.log'
            self.inputfiletype = 'gau'
        # generate charge command string for antechamber
        # use chg file from other program has higher priority
        qtype2comstr = {
            "am1-bcc": "-c bcc",
            "resp": "-c resp",
            "gasteiger": "-c gas"}
        chg_file = ''
        if os.path.isfile(self.basename + '_' + qtype + '.chg'):
            qcomstr = "-c rc -cf {:s}".format(self.basename + '_' + qtype + '.chg')
            chg_file = self.basename + '_' + qtype + '.chg'
        elif qtype in qtype2comstr:
            qcomstr = qtype2comstr[qtype]
        else:
            print("Error! charge file {:s} not found"
                  .format(self.basename + '_' + qtype + '.chg'))
            return False
        # generate input file command string
        ftype2comstr = {"gau": "-fi gout",
                        "mol2": "-fi mol2",
                        "pdb": "-fi pdb"}
        if self.inputfiletype in ftype2comstr:
            inpfcomstr = '-i {:s}'.format(self.inputfile) + " " + ftype2comstr[self.inputfiletype]
        else:
            print('File type of {:s} is not supported, I will try to convert it to pdb'
                  .format(self.inputfile))
            st = cm.read_structure(self.inputfile)
            cm.write_structure(st, name=self.basename + '.pdb')
            self.inputfile = self.basename + '.pdb'
            self.inputfiletype = 'pdb'
            inpfcomstr = "-i {:s} -fi pdb".format(self.inputfile + '.pdb')
        # generate other command string
        ffcomstr = "-at {:s}".format(fftype)
        outflnm = self.basename + '_amber'
        outfcomstr = "-fo mol2 -o {:s}.mol2 -rn {:s}".format(outflnm, self.basename[:3].upper())
        antechamber_com = "antechamber " + ' '.join([inpfcomstr, qcomstr, ffcomstr, outfcomstr]) + " -j 5 -pl 10 -dr no"
        addition_elem = "-atc PARMCHK.DAT"
        parmchk_com = 'parmchk2 -i {:s}.mol2 -f mol2 -o {:s}.frcmod -s gaff2 -a Y {:s}'.format(outflnm, outflnm,addition_elem)
        # copy file to wkdir and cd to wkdir
        wkdir = self.basename + '_AMBER'
        try:
            os.mkdir(wkdir)
        except FileExistsError:
            pass
        shutil.copy(self.inputfile, wkdir)
        if chg_file:
            shutil.copy(chg_file, wkdir)
        os.chdir(wkdir)
        # begin to run all the command
        antech_out = open('antechamber.out', 'w')
        return_code = subprocess.call(antechamber_com.split(), stdout=antech_out, stderr=antech_out)
        print('run following command in {:s}:'.format(wkdir))
        print(antechamber_com)
        if return_code != 0:
            print('Error! antechamber run failed! You should try the command:\n'
                  '{:s}\n'
                  'in {:s} to see what happens'.format(antechamber_com, wkdir))
            return False
        antech_out.close()
        if self.charge_only:
            chgflnm = self.basename + '_' + self.qtype + '.mol2'
            shutil.copy(outflnm + '.mol2', os.path.join(cwd, chgflnm))
            print('{:s} copied to current directory'.format(chgflnm))
            os.chdir(cwd)
            self.chgfile = chgflnm
            return
        self.__gen_parmchk_data()
        print(parmchk_com)
        parmchk_out = open('parmchk.out', 'w')
        return_code = subprocess.call(parmchk_com.split(), stdout=parmchk_out, stderr=parmchk_out)
        if return_code != 0:
            print('Error! parmchk2 run failed! You should try the command:\n'
                  '{:s}\n'
                  'in {:s} to see what happens'.format(parmchk_com, wkdir))
            return False
        parmchk_out.close()
        leapin = open(self.basename + '_leapin.in', 'w')
        leapin.write("loadamberparams {:s}.frcmod\n".format(outflnm))
        leapin.write("{:s}=loadmol2 {:s}.mol2\n".format(self.basename, outflnm))
        leapin.write("check {:s}\n".format(self.basename))
        leapin.write("saveamberparm {:s} {:s}.prmtop {:s}.inpcrd\n".format(self.basename, self.basename, self.basename))
        leapin.write("quit\n")
        leapin.close()
        tleap_com = "tleap -f {:s}_leapin.in".format(self.basename)
        print(tleap_com)
        tleap_out = open('tleap.out', 'w')
        return_code = subprocess.call(tleap_com.split(), stdout=tleap_out, stderr=tleap_out)
        if return_code != 0:
            print('Error! tleap run failed! You should try the command:\n'
                  '{:s}\n'
                  'in {:s} to see what happens and check the {:s}.in file'
                  .format(tleap_com, wkdir, '{:s}_leapin'.format(self.basename)))
            return False
        tleap_out.close()
        shutil.copy(self.basename + '.prmtop', cwd)
        print('{:s} copied to current directory'.format(self.basename + '.prmtop'))
        shutil.copy(self.basename + '.inpcrd', cwd)
        print('{:s} copied to current directory'.format(self.basename + '.inpcrd'))
        chgflnm = self.basename + '_' + self.qtype + '.mol2'
        shutil.copy(outflnm + '.mol2', os.path.join(cwd, chgflnm))
        print('{:s} copied to current directory'.format(chgflnm))
        os.chdir(cwd)
        self.topfile = self.basename + '.prmtop'
        self.crdfile = self.basename + '.inpcrd'
        self.chgfile = chgflnm
        return True

    def __run_multiwfn(self, fftype, qtype):
        # initialize variables
        cwd = os.getcwd()
        gen_resp = 0
        gen_resp2 = ''
        gen_cm5 = 0
        multiwfn_inp = []  # list of tuple,
        # first check charge type and check
        if qtype.startswith('resp2'):
            if self.basename.endswith('_sol'):
                self.basename = self.basename.replace('_sol','')
            if self.basename.endswith('_gas'):
                self.basename = self.basename.replace('_gas','')
        chgfile = self.basename + '_'+ qtype + '.chg'
        if os.path.isfile(chgfile):
            print("Find charge file {:s} in the current directory. Skip Multiwfn run"
                  .format(chgfile))
            return True
        if qtype == 'resp':
            if self.__check_gau_fchk(self.basename):
                if self.__check_gau_log(self.basename) == 2:
                    print('Multiwfn will read ESP from log file: {:s}'.format(self.basename+'.log'))
                    multiwfn_com = '7;18;8;;1;{:s};y;0;0;q'.format(self.basename + '.log')
                else:
                    print('Multiwfn will calculate ESP from fchk file: {:s}'.format(self.basename+'.fchk'))
                    multiwfn_com = '7;18;1;;y;0;0;q'
                inpfchk = self.basename + '.fchk'
                multiwfn_inp.append((inpfchk, multiwfn_com))
            elif self.__check_gau_fchk(self.basename + '_sol'):
                if self.__check_gau_log(self.basename + '_sol') == 2:
                    print('Multiwfn will read ESP from log file: {:s}'.format(self.basename+'_sol.log'))
                    multiwfn_com = '7;18;8;1;;{:s};y;0;0;q'.format(self.basename + '_sol.log')
                else:
                    print('Multiwfn will calculate ESP from fchk file: {:s}'.format(self.basename+'_sol.fchk'))
                    multiwfn_com = '7;18;1;;y;0;0;q'
                inpfchk = self.basename + '_sol.fchk'
                multiwfn_inp.append((inpfchk, multiwfn_com))
            else:
                gen_resp = 1
        if qtype.startswith('resp2'):
            if self.__check_gau_fchk(self.basename + '_gas'):
                if self.__check_gau_log(self.basename + '_gas') == 2:
                    multiwfn_com1 = '7;18;8;1;;{:s};y;0;0;q'.format(self.basename + '_gas.log')
                else:
                    multiwfn_com1 = '7;18;1;;y;0;0;q'
                inpfchk1 = self.basename + '_gas.fchk'
                multiwfn_inp.append((inpfchk1, multiwfn_com1))
            else:
                gen_resp2 = gen_resp2 + 'g'
            if self.__check_gau_fchk(self.basename + '_sol'):
                if self.__check_gau_log(self.basename + '_sol') == 2:
                    multiwfn_com2 = '7;18;8;1;;{:s};y;0;0;q'.format(self.basename + '_sol.log')
                else:
                    multiwfn_com2 = '7;18;1;;y;0;0;q'
                inpfchk2 = self.basename + '_sol.fchk'
                multiwfn_inp.append((inpfchk2, multiwfn_com2))
            else:
                gen_resp2 = gen_resp2 + 's'
        if qtype.startswith('cm5'):
            if self.__check_gau_fchk(self.basename):
                multiwfn_com = '7;-16;1;y;0;q'
                inpfchk = self.basename + '.fchk'
                multiwfn_inp.append((inpfchk, multiwfn_com))
            else:
                gen_cm5 = 1
        # second start to run multiwfn
        if not any([gen_resp, gen_resp2, gen_cm5]):
            # copy file to wkdir and cd to wkdir
            wkdir = self.basename + '_Multiwfn'
            try:
                os.mkdir(wkdir)
            except FileExistsError:
                pass
            print('run following command in {:s}:'.format(wkdir))
            for fchk, com in multiwfn_inp:
                shutil.copy(fchk, wkdir)
                log = [i for i in com.split(';') if '.log' in i]
                for f in log:
                    shutil.copy(f, wkdir)
                os.chdir(wkdir)
                com = com.replace(";", "\n")
                m = open('Multiwfn_com', 'w')
                m.write(com)
                m.close()
                Min = open('Multiwfn_com', 'r')
                mout = open('Multiwfn.out', 'w')
                try:
                    print('Multiwfn {:s} < Multiwfn_com > Multiwfn.out'.format(fchk))
                    p = subprocess.Popen(['Multiwfn', fchk], stdin=Min, stdout=subprocess.PIPE,
                                      stderr=mout, universal_newlines=True)
                except FileNotFoundError:
                    p = subprocess.Popen(['Multiwfn.exe', fchk], stdin=Min, stdout=subprocess.PIPE,
                                     stderr=mout, universal_newlines=True)
                newline = ''
                for line in p.stdout:
                    if 'Progress' in line:
                        line = line.strip('\n')
                        print(line, end='\r')
                        newline = ' '
                    else:
                        mout.write(line)
                if newline:
                    print(newline)
                p.wait()
                if p.returncode != 0:
                    print('Error! Multiwfn exit with error!\n'
                          'You need to check Multiwfn.out and run the Multiwfn manually to see what happens')
                    return 0
                Min.close()
                mout.close()
                os.chdir(cwd)
            # generate chg file for amber
            os.chdir(wkdir)
            out_chg = os.path.join(cwd, self.basename + '_' + qtype + '.chg')
            if qtype in ['cm5', 'resp']:
                chg_file = multiwfn_inp[0][0].replace('fchk', 'chg')
                cdf = pd.read_csv(chg_file, sep='\\s+', header=None)
                if qtype == 'cm5':
                    q = cdf[4] * 1.2
                else:
                    q = cdf[4]
                q.to_csv(out_chg, header=False, index=False)
            if qtype.startswith('resp2'):
                gas_file = [i.replace('fchk', 'chg') for i, j in multiwfn_inp if '_gas' in i]
                sol_file = [i.replace('fchk', 'chg') for i, j in multiwfn_inp if '_sol' in i]
                gas_df = pd.read_csv(gas_file[0], sep='\\s+', header=None)
                sol_df = pd.read_csv(sol_file[0], sep='\\s+', header=None)
                if len(qtype.split('-')) == 2:
                    delta = float(qtype.split('-')[1])
                else:
                    delta = 0.5
                resp2 = gas_df[4] * (1-delta) + sol_df[4] * delta
                resp2.to_csv(out_chg, header=False, index=False)
            print('{:s} generated'.format(out_chg))
            self.chgfile = self.basename+'_'+qtype+'.chg'
            os.chdir(cwd)
            # shutil.copy(self.ba+'.mol2', os.path.join(cwd, chgflnm))
            return True
        else:
        # third process output:
            if gen_resp == 1:
                print('Error! {:s} not found in {:s}.\nI will generate {:s} file here'
                      .format(self.basename + '.fchk', cwd, self.basename + '.gjf'))
                self.gen_gjf(name=self.basename + '.gjf', profile='resp')
            if 'g' in gen_resp2:
                print('Error! {:s} not found in {:s}.\nI will generate {:s} file here'
                .format(self.basename + '_gas.fchk', cwd, self.basename + '_gas.gjf'))
                self.gen_gjf(name=self.basename + '_gas.gjf', profile='resp_gas')
            if 's' in gen_resp2:
                print('Error! {:s} not found in {:s}.\nI will generate {:s} file here'
                .format(self.basename + '_sol.fchk', cwd, self.basename + '_sol.gjf'))
                self.gen_gjf(name=self.basename + '_sol.gjf', profile='resp')
            if gen_cm5 == 1:
                print('Error! {:s} not found in {:s}.\nI will generate {:s} file here'
                      .format(self.basename + '.fchk', cwd, self.basename + '.gjf'))
                self.gen_gjf(name=self.basename + '.gjf')
            print('The default level is B3LYP/def2SVP em=GD3BJ for opt and charge\n'
                  'You may need to modify and submit it to Gaussian09/16 to get the fchk file')
            return False

    def __check_libpargen(self):
        pass

    def __check_openff(self):
        pass

    def gen_gjf(self, name='', profile='', read=''):
        headtail_param = {'chk': name.replace('.gjf', '.chk'), 'nproc': '8', 'mem': '4GB', 'extra': ''}
        kw_param = {'method': 'b3lyp', 'basis': 'def2svp', 'solvent': 'water', 'addkey': '', 'vdw': 'em(gd3bj)'}
        sc_param = {'spin': self.spin, 'charge': self.charge}
        opt_flag = {'opt': '1'}
        opt_flag.update({k: v for k, v in self.gjf_options.items() if k in opt_flag.keys()})
        headtail_param.update({k: v for k, v in self.gjf_options.items() if k in headtail_param.keys()})
        kw_param.update({k: v for k, v in self.gjf_options.items() if k in kw_param.keys()})
        sc_param.update({k: v for k, v in self.gjf_options.items() if k in sc_param.keys()})
        gas_kw = ' '.join([kw_param['method'], kw_param['basis'], kw_param['vdw'], kw_param['addkey']])
        if kw_param['solvent'] and kw_param['solvent'] != 'none':
            sol_kw = gas_kw + 'scrf(pcm,solvent={:s})'.format(kw_param['solvent'])
        else:
            sol_kw = gas_kw
        if read:
            step1_kw = sol_kw + ' guess(read)'
            headtail_param.update({'oldchk': read})
        else:
            step1_kw = sol_kw
        combined_param = {}
        combined_param.update(headtail_param)
        combined_param.update(sc_param)
        link = []
        update_param = {}
        if not profile:
            if opt_flag['opt'] == '1':
                keywords = 'opt ' + step1_kw
            else:
                keywords = step1_kw + 'Geom(redundant)'
        elif profile == 'resp':
            if opt_flag['opt'] == '1':
                keywords = 'opt ' + step1_kw
                link = [{'keywords': 'Pop(mk) iop(6/33=2) iop(6/42=6) guess=read geom=allcheck ' + sol_kw}]
            else:
                keywords = 'Pop(mk) iop(6/33=2) iop(6/42=6) Geom(redundant)' + step1_kw
        elif profile == 'resp_gas':
            if opt_flag['opt'] == '1':
                keywords = 'opt ' + gas_kw
                link = [{'keywords': 'Pop(mk) iop(6/33=2) iop(6/42=6) guess=read geom=allcheck ' + gas_kw}]
            else:
                keywords = 'Pop(mk) iop(6/33=2) iop(6/42=6) Geom(redundant)' + gas_kw
        else:
            print("Error! unrecognized profile {:s} for function gen_gjf in class ParamGen"
                    .format(profile))
            sys.exit()
        combined_param.update(update_param)
        combined_param['keywords'] = keywords
        combined_param['link'] = link
        if not name:
            flnm = self.basename + '.gjf'
        else:
            flnm = name
        st = cm.read_structure(self.inputfile)
        cm.write_structure(st, name=flnm, **combined_param)

    def gen_orca_input(self):
        pass

    def ligpargen(self):
        pass
