import numpy as np
import pickle
import pymc3 as pm 
import arviz as az
import theano.tensor as tt
from distutils.dir_util import mkpath
import os.path as op
from anchor_points import get_a_polynd, calc_poly, calc_poly_tt, polyfitnd, calc_poly_tt_vi
import glob
import argparse as ap
import sys
from astropy.table import Table

def parse_args(argv=None):
    """ Tool to parse arguments from the command line. The entries should be self-explanatory """
    parser = ap.ArgumentParser(description="DustPymc3Results",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-f','--input_file',help='Input file with values of independent variables; ascii format with possible column names logM (stellar mass), ssfr (specific star formation rate), logZ (metallicity), z (redshift), i (axis ratio), and d2 (diffuse dust optical depth)',default=None)
    parser.add_argument('-dg','--degree',help='Degree of polynomial (true)',type=int,default=2)
    parser.add_argument('-dir','--dir_orig',help='Parent directory for files',type=str,default='Pymc3Results')
    parser.add_argument('-n','--n',help='Whether or not to use dust index as dependent variable',action='count',default=0)
    parser.add_argument('-md','--mode',help='Which type of results to use: options are "hb" for hierarchical Bayesian and "lm" for least squares optimization using highest likelihood posteriors',type=str,default='hb')
    parser.add_argument('-ext','--extratext',help='Extra text to help distinguish particular run (optional)',type=str,default='')

    args = parser.parse_args(args=argv)

    args.mode = args.mode.lower()
    if args.input_file is None:
        sys.exit("You need to supply an input file with the values of the independent variables")
    else:
        dat = Table.read(args.input_file,format='ascii')
    cols = dat.colnames
    prop_dict, dep_dict = make_prop_dict()
    args.indep_name, args.indep_lab = [], []
    args.indep = []
    args.dataname = ''
    for name in prop_dict['names'].keys():
        if name in cols:
            args.indep_name.append(prop_dict['names'][name])
            args.indep_lab.append(prop_dict['labels'][name])
            args.indep.append(dat[name])
            args.dataname += name+'_'
    args.dataname += args.dep_name
    args.indep = np.array(args.indep)
    if args.n: 
        args.dep_name, args.dep_lab = dep_dict['names']['n'], dep_dict['labels']['n']
    else:
        args.dep_name, args.dep_lab = dep_dict['names']['tau2'], dep_dict['labels']['tau2']
    
    return args

def make_prop_dict():
    prop_dict, dep_dict = {}, {}
    prop_dict['names'] = {'logM': 'logM', 'ssfr': 'logsSFR', 'sfr': 'logSFR', 'logZ': 'logZ', 'z': 'z', 'i':'axis_ratio', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['names'] = {'tau2': 'dust2', 'n': 'n'}
    prop_dict['labels'] = {'logM': r'$\log M_*$', 'ssfr': r'$\log$ sSFR$_{\rm{100}}$', 'sfr': r'$\log$ SFR$_{\rm{100}}$', 'logZ': r'$\log (Z/Z_\odot)$', 'z': 'z', 'i':r'$b/a$', 'd1':r"$\hat{\tau}_{1}$", 'd2':r"$\hat{\tau}_{2}$"}
    dep_dict['labels'] = {'tau2': r"$\hat{\tau}_{2}$", 'n': 'n'}
    prop_dict['pickle_names'] = {'logM': 'stellar_mass', 'ssfr': 'ssfr_100', 'sfr': 'sfr_100', 'logZ': 'log_z_zsun', 'z': 'z', 'i':'inc', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['pickle_names'] = {'tau2': 'dust2', 'n': 'dust_index'}
    return prop_dict, dep_dict

def getFiles(args):
    dir_fin = op.join(args.dir_orig,args.deg)
    filelist = glob.glob(op.join(dir_fin,'*%s*'%(args.dataname)))
    if args.mode=='lm': 
        try:
            return [fl for fl in filelist if 'lmfit' in fl][0]
        except:
            sys.exit("Couldn't find lmfit best-fit parameter file; make sure you have it in the right directory or try a different configuration")
    else:
        try:
            return [fl for fl in filelist if 'HB' in fl][0] + [fl for fl in filelist if 'trace' in fl][0]
        except:
            sys.exit("Couldn't find either trace file or best-fit parameter file; make sure you have it in the right directory or try a different configuration")

def calc_best_fit_lmfit(args):
    fl = getFiles(args)
    dat = Table.read(fl,format='ascii')
    ndim = len(args.indep_name)
    sh = tuple([ndim]+[args.degree+1]*ndim)
    xx = np.empty((ndim,(args.degree+1)**ndim))
    med = np.empty(ndim)
    for i, name in enumerate(args.indep_name):
        xx[i] = dat[name]
        med[i] = dat[name+'_plus_med'][0]-dat[name][0]
    coefs = polyfitnd(xx.reshape(sh),dat['ngrid'])
    dep_lm = calc_poly(args.indep-med[:,None],coefs,args.degree)
    return dep_lm

def calc_pymc3(args,numsamp=-1):
    # Start with best fit values (median values of fitted parameters)
    fl = getFiles(args)
    dat = Table.read(fl[0],format='ascii')
    ndim = len(args.indep_name)
    sh = tuple([ndim]+[args.degree+1]*ndim)
    xx = np.empty((ndim,(args.degree+1)**ndim))
    med = np.empty(ndim)
    for i, name in enumerate(args.indep_name):
        xx[i] = dat[name]
        med[i] = dat[name+'_plus_med'][0]-dat[name][0]
    coefs_med = polyfitnd(xx.reshape(sh),dat['ngrid'])
    dep_med = calc_poly(args.indep-med[:,None],coefs_med,args.degree)
    # Create various samples of polynomial given the trace
    trace = az.from_netcdf(fl[1])
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,grid_name)), np.array(getattr(trace.posterior,width_name))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],sh[2]), log_width_0.reshape(sh[0]*sh[1])
    
    if numsamp==-1: numsamp = len(log_width)
    inds = np.random.choice(len(log_width),size=numsamp,replace=False)
    n_sim = np.empty((numsamp,indep[0].size))
    for i, ind in enumerate(inds):
        ngrid_mod = ngrid[ind]
        width_mod = np.exp(log_width[ind])
        coefs_mod = polyfitnd(xx,ngrid_mod)
        n_sim[i] = calc_poly(indep,coefs_mod,args.degree) + width_mod * np.random.randn(*n_sim[i].shape)
    return dep_med, n_sim

def main():
    args = parse_args()
    if args.mode=='lm': dep_med = calc_best_fit_lmfit(args)
    else: dep_med, n_sim = calc_pymc3(args)
    dict_for_pickle = {}
    for i, name in enumerate(args.indep_name):
        dict_for_pickle[name] = args.indep[i]
    dict_for_pickle['%s_med'%(args.dep_name)] = dep_med
    if args.mode=='hb': dict_for_pickle['%s_samples'%(args.dep_name)] = n_sim.T

    pickle.dump(dict_for_pickle,open('%s_%s%s.pickle'%(args.dataname,args.mode,args.extratext),'wb'),protocol=4)

if __name__=='__main__':
    main()