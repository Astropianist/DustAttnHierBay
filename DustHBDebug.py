import numpy as np 
import pymc3 as pm 
import theano.tensor as tt
import arviz as az
import pickle
import matplotlib.pyplot as plt 
import os.path as op
from distutils.dir_util import mkpath
from anchor_points import calc_poly, polyfitnd, calc_poly_tt, get_a_polynd
from DustPymc3_Final import make_prop_dict, plot_model_true, plot_color_map, plot1D, polyNDData, get_relevant_info_ND_Data, mass_completeness
from DustSimple import parse_args, label_creation, pymc3_simple, plot_pymc3
from copy import deepcopy
import seaborn as sns 
sns.set_context("paper") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               }) 

ssfrlim = -12.5

def getData():
    obj = pickle.load(open("3dhst_samples.pickle",'rb'))
    logM, ssfr, logZ, z, tau1, tau2, n = np.log10(obj['stellar_mass'])[:,0], np.log10(obj['ssfr_100'])[:,0], obj['log_z_zsun'][:,0], obj['z'], obj['dust1'][:,0], obj['dust2'][:,0], obj['dust_index'][:,0]
    inc = np.loadtxt('GalfitParamsProsp.dat',usecols=2,unpack=True)
    masscomp = mass_completeness(z)
    cond = np.logical_and.reduce((logM>=masscomp,ssfr>=ssfrlim))
    ind = np.where(cond)[0]
    return logM[ind], ssfr[ind], logZ[ind], z[ind], tau1[ind], tau2[ind], n[ind], inc[ind], ind

def getRealData(args,ind=None):
    obj = pickle.load(open("3dhst_resample_%d_inc.pickle"%(args.err_samp),'rb'))
    logM, ssfr, logZ, z, tau1, tau2, n, inc = np.log10(obj['stellar_mass']), np.log10(obj['ssfr_100']), obj['log_z_zsun'], obj['z'], obj['dust1'], obj['dust2'], obj['dust_index'], obj['inc']
    if ind is None:
        masscomp = mass_completeness(z)
        logMavg = np.average(logM,axis=1); logssfravg = np.average(ssfr,axis=1)
        cond = np.logical_and.reduce((logMavg>=masscomp,logssfravg>=ssfrlim))
        ind = np.where(cond)[0]
    return logM[ind], ssfr[ind], logZ[ind], z[ind], tau1[ind], tau2[ind], n[ind], inc[ind]

def getNecessaryData(args):
    logM, ssfr, logZ, z, tau1, tau2, n, inc, ind = getData()
    logMsamp, ssfrsamp, logZsamp, _, tau1samp, tau2samp, nsamp, incsamp = getRealData(args,ind)
    prop_dict, dep_dict = make_prop_dict()
    if args.i: cond = inc>=0
    else: cond = logM>=0
    ind = np.where(cond)[0]
    if args.size==-1: args.size = len(ind)
    indfin = np.random.choice(ind,size=args.size,replace=False)
    indep = np.empty((0,args.size,args.err_samp))
    indep_orig, indep_err = np.empty((0,args.size)), np.empty((0,args.size))
    indepsamp = np.empty((0,args.size,args.err_samp))
    med_mod = np.array([])
    indep_name, indep_lab, dep_name, dep_lab, nlim = label_creation(args,prop_dict,dep_dict)
    if args.logM: 
        md = np.median(logM[indfin])
        indep_orig = np.append(indep_orig, logM[indfin][None,:]-md,axis=0)
        indep_err = np.append(indep_err,args.err_mult*prop_dict['sigma']['logM']*np.ones((1,args.size)),axis=0)
        indep = np.append(indep,logM[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['logM']*np.random.randn(args.size,args.err_samp)[None,:,:],axis=0)
        indepsamp = np.append(indepsamp,logMsamp[indfin][None,:,:] - md, axis=0)
        med_mod = np.append(med_mod, md)
    if args.ssfr: 
        md = np.median(ssfr[indfin])
        indep_orig = np.append(indep_orig, ssfr[indfin][None,:]-md,axis=0)
        indep_err = np.append(indep_err,args.err_mult*prop_dict['sigma']['ssfr']*np.ones((1,args.size)),axis=0)
        indep = np.append(indep,ssfr[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['ssfr']*np.random.randn(args.size,args.err_samp)[None,:,:],axis=0)
        indepsamp = np.append(indepsamp,ssfrsamp[indfin][None,:,:] - md, axis=0)
        med_mod = np.append(med_mod, md)
    if args.logZ: 
        md = np.median(logZ[indfin])
        indep_orig = np.append(indep_orig, logZ[indfin][None,:]-md,axis=0)
        indep_err = np.append(indep_err,args.err_mult*prop_dict['sigma']['logZ']*np.ones((1,args.size)),axis=0)
        indep = np.append(indep,logZ[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['logZ']*np.random.randn(args.size,args.err_samp)[None,:,:],axis=0)
        indepsamp = np.append(indepsamp,logZsamp[indfin][None,:,:] - md, axis=0)
        med_mod = np.append(med_mod,md)
    if args.z: 
        md = np.median(z[indfin])
        indep_orig = np.append(indep_orig, z[indfin][None,:]-md,axis=0)
        indep_err = np.append(indep_err,args.err_mult*prop_dict['sigma']['z']*np.ones((1,args.size)),axis=0)
        indep = np.append(indep,z[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['z']*np.random.randn(args.size,args.err_samp)[None,:,:],axis=0)
        indepsamp = np.append(indepsamp,z[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['z']*np.random.randn(args.size,args.err_samp)[None,:,:], axis=0)
        med_mod = np.append(med_mod,md)
    if args.i:
        md = np.median(inc[indfin])
        indep_orig = np.append(indep_orig, inc[indfin][None,:]-md,axis=0)
        indep_err = np.append(indep_err,args.err_mult*prop_dict['sigma']['i']*np.ones((1,args.size)),axis=0)
        indep = np.append(indep,inc[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['i']*np.random.randn(args.size,args.err_samp)[None,:,:],axis=0)
        indepsamp = np.append(indepsamp,incsamp[indfin][None,:,:] - md, axis=0)
        med_mod = np.append(med_mod,md)
    if args.d1:
        md = np.median(tau1[indfin])
        indep_orig = np.append(indep_orig, tau1[indfin][None,:]-md,axis=0)
        indep_err = np.append(indep_err,args.err_mult*prop_dict['sigma']['d1']*np.ones((1,args.size)),axis=0)
        indep = np.append(indep,tau1[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['d1']*np.random.randn(args.size,args.err_samp)[None,:,:],axis=0)
        indepsamp = np.append(indepsamp,tau1samp[indfin][None,:,:] - md, axis=0)
        med_mod = np.append(med_mod,md)
    if args.d2:
        md = np.median(tau2[indfin])
        indep_orig = np.append(indep_orig, tau2[indfin][None,:]-md,axis=0)
        indep_err = np.append(indep_err,args.err_mult*prop_dict['sigma']['d2']*np.ones((1,args.size)),axis=0)
        indep = np.append(indep,tau2[indfin][None,:,None] - md + args.err_mult*prop_dict['sigma']['d2']*np.random.randn(args.size,args.err_samp)[None,:,:],axis=0)
        indepsamp = np.append(indepsamp,tau2samp[indfin][None,:,:] - md, axis=0)
        med_mod = np.append(med_mod,md)
    if args.n: 
        dep_orig = n[indfin]
        dep_err = args.err_mult*dep_dict['sigma']['n']*np.ones(args.size)
        dep = n[indfin][:,None] + args.err_mult*dep_dict['sigma']['n']*np.random.randn(args.size,args.err_samp)
        depsamp = nsamp[indfin]
        uniform = True
    else: 
        dep_orig = tau2[indfin]
        dep_err = args.err_mult*dep_dict['sigma']['tau2']*np.ones(args.size)
        dep = tau2[indfin][:,None] + args.err_mult*dep_dict['sigma']['tau2']*np.random.randn(args.size,args.err_samp)
        depsamp = tau2samp[indfin]
        uniform = False

    return indep, dep, indep_orig, indep_err, dep_orig, dep_err, indepsamp, depsamp, med_mod, z[indfin], uniform, indep_name, indep_lab, dep_name, dep_lab, nlim

def main():
    args = parse_args()
    img_dir_orig = op.join('DataTrue',args.dir_orig,'pm_dbg')
    
    indep, dep, indep_orig, indep_err, dep_orig, dep_err, indepsamp, depsamp, med_mod, z, uniform, indep_name, indep_lab, dep_name, dep_lab, nlim = getNecessaryData(args)

    trace, xx, map_estimate = pymc3_simple(indep_orig,dep_orig,img_dir_orig,degree=args.degree,mindep=nlim[0],maxdep=nlim[1],sampling=args.steps,tune=args.tune,uniform=uniform,extratext=args.extratext,plot=args.plot)

    plot_pymc3(trace,xx,indep_orig,indep_err,dep_orig,dep_err,indep_name,indep_lab,dep_name,dep_lab,med_mod,img_dir_orig=img_dir_orig,extratext=args.extratext,mindep=nlim[0],maxdep=nlim[1],map_estimate=map_estimate)

    trace, a_poly_T, xx = polyNDData(indep,dep,0.0,img_dir_orig,dep_lim=nlim,tune=args.tune,plot=args.plot,extratext=args.extratext,degree=args.degree,sampling=args.steps,uniform=uniform,var_inf=False)

    get_relevant_info_ND_Data(trace,a_poly_T,xx,indep,dep,med_mod,indep_name,indep_lab,dep_name,dep_lab,degree2=args.degree,extratext=args.extratext,img_dir_orig=img_dir_orig,var_inf=False)

    trace, a_poly_T, xx = polyNDData(indepsamp,depsamp,0.0,img_dir_orig,dep_lim=nlim,tune=args.tune,plot=args.plot,extratext=args.extratext+'_samp',degree=args.degree,sampling=args.steps,uniform=uniform,var_inf=False)

    get_relevant_info_ND_Data(trace,a_poly_T,xx,indepsamp,depsamp,med_mod,indep_name,indep_lab,dep_name,dep_lab,degree2=args.degree,extratext=args.extratext+'_samp',img_dir_orig=img_dir_orig,var_inf=False)

if __name__ == '__main__':
    main()