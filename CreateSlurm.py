import sys
import numpy as np
import argparse as ap
from subprocess import run
from DustPymc3_Final import parse_args, make_prop_dict, label_creation

""" Create and run a batch job for the dust attenuation code using SLURM for a given set of conditions """

def main():
    # Get parameters of the run
    argv = sys.argv
    argv.remove('CreateSlurm.py')
    arg_str = ''
    for i in range(len(argv)-1):
        arg_str+=argv[i]+' '
    arg_str+=argv[-1]
    args = parse_args(argv)
    prop_dict, dep_dict = make_prop_dict()
    indep_name, indep_name2, _, _, _, _, _, dep_name, _, _ = label_creation(args,prop_dict,dep_dict)

    # Get the job/file name
    jobname = dep_name + '_'
    for nm in indep_name2: jobname += nm + '_'
    if not args.real:
        for nm in indep_name: jobname += nm + '_'
    jobname += str(args.size) + '_' + args.extratext

    # Get the approximate time and memory needed for the simulation
    if args.size==-1: size = 30000
    else: size = args.size
    # t_sec_tot = int(0.005*size*args.samples*args.degree2*len(indep_name2)**2 + 3000.0*np.sqrt(size/500.0))
    # if not args.n: t_sec_tot = 600000
    t_sec_tot = 600000
    t_day = t_sec_tot//86400; t_rem = t_sec_tot%86400
    t_hour = t_rem//3600; t_rem = t_rem%3600
    t_min = t_rem//60; t_sec = t_rem%60
    mem_MB = min(int(5.0e-4*size*args.samples*(args.degree2+1)**len(indep_name2)) + 10000, 500000) # Buffer of 10 GB

    f = open(jobname+'.sh','w')
    f.write('#!/bin/bash \n')
    f.write('#SBATCH --job-name '+jobname + '\n')
    f.write('#SBATCH -t %d-%d:%d:%d \n'%(t_day,t_hour,t_min,t_sec))
    f.write('#SBATCH -p genx -c 4 \n')
    f.write('#SBATCH --mem=%d \n'%(mem_MB))
    f.write('#SBATCH --mail-type=ALL \n')
    f.write('#SBATCH --mail-user=gnagaraj-visitor@flatironinstitute.org \n')
    f.write('cd ${HOME}/ceph/DustAttn/ \n')
    f.write('module load slurm gcc lib/openblas \n')
    f.write('source ${HOME}/ceph/env/bin/activate \n')
    f.write('python3 DustPymc3_Final.py %s \n'%(arg_str))
    f.write('deactivate')
    f.close()

    run('module load slurm gcc lib/openblas',shell=True)
    run("sbatch %s.sh"%(jobname),shell=True)

if __name__=='__main__':
    main()
