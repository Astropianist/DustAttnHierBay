""" Modeling dust attenuation curves (diffuse dust optical depth and slope of the curve) as a function of 1 or more of the following physical parameters: stellar mass, specific star formation rate, metallicity, redshift, inclination, diffuse dust optical depth (if not the dependent variable), birth cloud dust optical depth.

We use hierarchical models with importance sampling techniques (to be able to use posterior samples from Prospector, the code applied to individual galaxies) in the Pymc3 framework.

Author: Gautam Nagaraj """

import numpy as np
import pickle
import pymc3 as pm 
import arviz as az
import matplotlib.pyplot as plt 
import seaborn as sns
from time import time
import theano.tensor as tt
from distutils.dir_util import mkpath
import os.path as op
from anchor_points import get_a_polynd, calc_poly, calc_poly_tt, polyfitnd, calc_poly_tt_vi
from PlotColorMapProj import makeColorMapProj, getModelSamples
import DustAttnCurveModules as dac
import argparse as ap
from scipy.stats import norm, truncnorm
from scipy.integrate import trapz
from copy import copy, deepcopy
from scipy.interpolate import LinearNDInterpolator, interp1d
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool
from itertools import product
from astropy.table import Table
import corner
from sedpy.attenuation import noll

sns.set_context("paper") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

min_pdf = 1.0e-15
ssfr_lim_prior = -15.0
ssfr_lims_data = (-14.5,-7.0)
cores = 4
fl_fit_stats_univar = 'run_convergence_univar.txt'
fl_fit_stats_bivar = 'run_convergence_bivar_joint_pr_sig.txt'
nlab, taulab = 'n', r"$\hat{\tau}_{2}$"

def mass_completeness(zred):
    """used mass-completeness estimates from Tal+14, for FAST masses
    then applied M_PROSP / M_FAST to estimate Prospector completeness
    Credit: Joel Leja
    """

    zref = np.array([0.65,1,1.5,2.1,3.0])
    mcomp_prosp = np.array([8.71614882,9.07108637,9.63281923,9.79486727,10.15444536])
    mcomp = np.interp(zred, zref, mcomp_prosp)

    return mcomp

def get_dust_attn_curve_d2(wave,n=0.0,d2=1.0):
    Eb = 0.85 - 1.9*n
    return noll(wave,tau_v=d2,delta=n,c_r=0.0,Ebump=Eb)

def get_dust_attn_curve_d1(wave,d1=1.0):
    return d1*(wave/5500.0)**(-1)

def nvchev(tau_v):
    return 2.8/(1.0 + 3.0*np.sqrt(tau_v))

def bchev(tau_v):
    return 0.3 - 0.05*tau_v

def nlamchev(lam,tau_v):
    return nvchev(tau_v) + bchev(tau_v)*(lam-0.55)

def taulamchev(lam,tau_v):
    return tau_v * (lam/0.55)**(-nlamchev(lam,tau_v))

def getChevFunc():
    tau_v_arr = np.linspace(1.0e-5,5.0,10001)
    nvsS = getnvsS()
    Schev = taulamchev(0.15,tau_v_arr)/taulamchev(0.55,tau_v_arr)
    nchev = nvsS(Schev)
    return interp1d(tau_v_arr, nchev, bounds_error=None, fill_value='extrapolate')
    
def getnvsS():
    wvs = np.array([1500.,5500.])
    narr = np.linspace(-2.0,2.0,4001)
    Sarr = np.zeros_like(narr)
    for i, n in enumerate(narr):
        d21500, d25500 = get_dust_attn_curve_d2(wvs,n)
        Sarr[i] = d21500/d25500
    return interp1d(Sarr, narr)

def parse_args(argv=None):
    """ Tool to parse arguments from the command line. The entries should be self-explanatory """
    parser = ap.ArgumentParser(description="DustPymc3",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-e','--error_mult',help='Factor determining absolute error in simulated data',type=float, default=0.05)
    parser.add_argument('-en','--error_mult_n',help='Factor determining absolute error in true relation',type=float, default=0.1)
    parser.add_argument('-et','--error_mult_t',help='Factor determining absolute error in true relation second variable',type=float, default=0.1)
    parser.add_argument('-ec','--error_mult_cross',help='Factor determining absolute error in true relation',type=float, default=1.0)
    parser.add_argument('-si','--size',help='Number of mock galaxies',type=int,default=500)
    parser.add_argument('-sa','--samples',help='Number of posterior samples per galaxy',type=int,default=50)
    parser.add_argument('-pl','--plot',help='Whether or not to plot ArViz plot',action='count',default=0)
    parser.add_argument('-tu','--tune',help='Number of tuning steps',type=int,default=1000)
    parser.add_argument('-st','--steps',help='Number of desired steps included per chain',type=int,default=1000)
    parser.add_argument('-ext','--extratext',help='Extra text to help distinguish particular run',type=str,default='')
    parser.add_argument('-dg','--degree',help='Degree of polynomial (true)',type=int,default=2)
    parser.add_argument('-dg2','--degree2',help='Degree of polynomial (model)',type=int,default=2)
    parser.add_argument('-data','--data',help='To use data locations or not',action='count',default=0)
    parser.add_argument('-r','--real',help='Real data or simulation',action='count',default=0)
    parser.add_argument('-dir','--dir_orig',help='Parent directory for files',type=str,default='NewTests')

    parser.add_argument('-m','--logM',help='Whether or not to include stellar mass (true)',action='count',default=0)
    parser.add_argument('-s','--ssfr',help='Whether or not to include sSFR (true)',action='count',default=0)
    parser.add_argument('-sfr','--sfr',help='Whether or not to include SFR (true)',action='count',default=0)
    parser.add_argument('-logZ','--logZ',help='Whether or not to include metallicity (true)',action='count',default=0)
    parser.add_argument('-z','--z',help='Whether or not to include redshift (true)',action='count',default=0)
    parser.add_argument('-i','--i',help='Whether or not to include inclination in model',action='count',default=0)
    parser.add_argument('-d1','--d1',help='Whether or not to include birth cloud dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-d2','--d2',help='Whether or not to include diffuse dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-mm','--logM_mod',help='Whether or not to include stellar mass in model',action='count',default=0)
    parser.add_argument('-sm','--ssfr_mod',help='Whether or not to include sSFR in model',action='count',default=0)
    parser.add_argument('-sfrm','--sfr_mod',help='Whether or not to include SFR in model',action='count',default=0)
    parser.add_argument('-logZm','--logZ_mod',help='Whether or not to include metallicity in model',action='count',default=0)
    parser.add_argument('-zm','--z_mod',help='Whether or not to include redshift in model',action='count',default=0)
    parser.add_argument('-im','--i_mod',help='Whether or not to include inclination in model',action='count',default=0)
    parser.add_argument('-d1m','--d1_mod',help='Whether or not to include birth cloud dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-d2m','--d2_mod',help='Whether or not to include diffuse dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-n','--n',help='Whether or not to use dust index as dependent variable',action='count',default=0)
    parser.add_argument('-bv','--bivar',help='Whether or not to perform bivariate fitting',action='count',default=0)
    parser.add_argument('-rd','--random',help='Whether or not to add random numbers to color plots',action='count',default=0)
    parser.add_argument('-ad','--already_done',help='Whether fit was already finished',action='count',default=0)

    args = parser.parse_args(args=argv)
    if not args.n: args.d2_mod = 0
    if args.real: 
        if args.bivar: args.dir_orig = 'Bivar_Prior_Sig'
            # if args.size==-1: args.dir_orig = 'Bivar_Full'
            # else: args.dir_orig = 'Bivar_Sample'
        else:
            if args.size==-1: args.dir_orig = 'Full'
            else: args.dir_orig = 'NewTests'
    else: 
        if args.bivar: args.dir_orig = 'Bivar'
        else: args.dir_orig = 'PolyGen'

    return args

def polyND_bivar(img_dir_orig,limlist=None,size=500,samples=50,plot=False,extratext='',nlim=np.array([-1.0,0.4]),taulim=np.array([0.0,4.0]),degree=2,sampling=1000,tune=1000,errmult=1.0,errmultn=1.0,errmultt=1.0,errmultcross=1.0,indep_true=None,sigarr=None,sigN=0.19,sigT=0.12,n=None,tau=None):
    ''' Simulation of data where the dust index n is a 2-D polynomial function of stellar mass and sSFR.

    Parameters
    ----------
    img_dir: Str
        Parent directory in which folders will be created for the particular run
    size: Int
        Number of "galaxies" in the sample
    samples: Int
        Number of "posterior samples" available for each galaxy
    plot: Bool
        Whether or not an arviz plot should be made
    extratext: Str
        An addition to the plot name to distinguish it from previous runs
    mlim, ssfrlim, nlim: Two-element lists
        Lower and upper limits for all parameters in question
    degree: Int
        Polynomial degree in each variable
    sampling: Int
        Number of draws for sampling (not including burn-in steps)
    tune: Int
        Number of burn-in steps for each sampling chain
    errmult: Float
        Multiplicative factor that determines how large the simulated data errors are
    errmultn: Float
        Multiplicative factor that determines how large the intrinsic dispersion in the relation is
    logM, ssfr, n: 1-D Numpy Arrays
        If provided, this will supercede the creation of "true" values for log(M), log(sSFR), and n_grid
    Returns
    -------
    trace: Pymc3 trace object
        Result of pymc3 sampler
    a_poly_T: 2-D Numpy arrays
        Matrix whose inverse times n_grid gives polynomial coefficients
    xx: List of 2-D Numpy arrays
        Meshgrid where dust index is computed
    ngrid_true: 1-D Numpy array
        True values for dust index at (coarse) grid
    coefs_true: 1-D Numpy array
        True values for polynomial coefficients
    n_true: 1-D Numpy array
        True values for dust index at the true locations of the sample
    width_true: Float
        True intrinsic dispersion in relation
    indep_true: 2-D Numpy array
        Array with each row corresponding to true values of an independent variable (log(M) and log(sSFR))
    med_logM: Float
        Median log(M) value--for adding back to indep_true to get the actual log(M) values in plots 
    med_ssfr: Float
        Median log(sSFR) value
    
    The function runs the pymc3 sampler on this toy example and prints the results; an arviz plot of the results is also made if plot==True
    '''
    # Since there are six figures to create for one run of the code, we create a directory for each run, seperated by polynomial degree (and the run-specific text)
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)
    mkpath(img_dir)

    if limlist is None:
        limlist = []
        ndim = len(indep_true)
        for indep in indep_true:
            # limlist.append(np.percentile(indep,[2.5,97.5]))
            limlist.append(np.array([min(indep),max(indep)]))
    else:
        ndim = len(limlist)
        indep_true = np.empty((0,size))
        for lim in limlist:
            indep_true = np.append(indep_true,np.random.uniform(lim[0],lim[-1],(1,size)),axis=0)

    # Determining true parameters
    width_true = 0.1*errmultn #Intrinsic dispersion in relation param 1 (n)
    width_true2 = 0.1*errmultt #Intrinsic dispersion in relation param 2 (tau)
    rho_true = 0.3 #Cross intrinsic dispersion
    x = np.empty((0,degree+1)) # To set up grid on which dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    # rng = nlim[1]-nlim[0]
    # rng2 = taulim[1]-taulim[0]
    # avg = (nlim[1]+nlim[0])/2.0
    # avg2 = (taulim[1]+taulim[0])/2.0
    if n is None:
        np.random.seed(3890)
        # ngrid_true = 0.9*rng*np.random.rand(xx[0].size)-0.9*rng/2.0 + avg #True values of dust parameter at the grid
        ngrid_true = np.random.uniform(nlim[0],nlim[1],xx[0].size)
        taugrid_true = truncnorm.rvs(taulim[0],taulim[1],loc=0.3,scale=1.0,size=xx[0].size) #True values of 2nd dust parameter at the grid
    # When "data" inputs provided, use measured n to determine approximate grid values for n according to Prospector
    else:
        ngrid_true, taugrid_true = np.zeros(xx[0].size), np.zeros(xx[0].size)
        for index, i in enumerate(np.ndindex(xx[0].shape)):
            totarr = np.zeros_like(indep_true[0])
            for j, ind in enumerate(i):
                totarr+=abs(indep_true[j]-x[j,ind])
            ngrid_true[index] = n[np.argmin(totarr)]
            taugrid_true[index] = tau[np.argmin(totarr)]
    coefs_true = np.linalg.lstsq(a_poly_T,ngrid_true,rc)[0]
    coefs_true2 = np.linalg.lstsq(a_poly_T,taugrid_true,rc)[0] #Calculation of polynomial coefficients from the grid
    Z1 = np.random.randn(size) # Independent random variable for creating true parameters that follow a 2D Gaussian distribution
    Z2 = np.random.randn(size) # Independent random variable for creating true parameters that follow a 2D Gaussian distribution
    n_true = calc_poly(indep_true,coefs_true,degree) + width_true * Z1 #True dust index parameter n is polynomial function of n with some intrinsic scatter
    tau_true = calc_poly(indep_true,coefs_true2,degree) + width_true2 * (rho_true*Z1 + np.sqrt(1.0-rho_true**2)*Z2) #True dust index parameter tau is polynomial function of tau with some intrinsic scatter

    # Observed values of the galaxy parameters
    if sigarr is None:
        sigarr = np.repeat(0.15,len(indep_true))*errmult
    else:
        sigarr *= errmult
    sigN *= errmult; sigT *= errmult
    # First we need to "move" the mean of the posterior, then we generate posterior samples
    indep_samp = indep_true[:,:,None] + sigarr[:,None,None]*np.random.randn(ndim,size,1) + sigarr[:,None,None]*np.random.randn(ndim,size,samples)

    Z1 = np.random.randn(size,1) # Independent random variable for creating true parameters that follow a 2D Gaussian distribution
    Z2 = np.random.randn(size,1) # Independent random variable for creating true parameters that follow a 2D Gaussian distribution
    Z11 = np.random.randn(size,samples) # Independent random variable for creating true parameters that follow a 2D Gaussian distribution
    Z21 = np.random.randn(size,samples) # Independent random variable for creating true parameters that follow a 2D Gaussian distribution
    n_samp = n_true[:, None] + sigN * Z1 + sigN * Z11
    tau_samp = tau_true[:, None] + sigT * (rho_true*Z1 + np.sqrt(1.0-rho_true)*Z2) + sigT * (rho_true*Z11 + np.sqrt(1.0-rho_true)*Z21)
    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,degree)
    # breakpoint()
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        ngrid = pm.Uniform("ngrid",lower=nlim[0]-width_true,upper=nlim[1]+width_true,shape=ngrid_true.size,testval=np.random.uniform(nlim[0],nlim[1],ngrid_true.size))
        # 0.9*rng*np.random.rand(ngrid_true.size)-0.9*rng/2.0 + avg
        taugrid = pm.TruncatedNormal("taugrid",mu=0.3,sigma=1.0,lower=taulim[0]-width_true2,upper=taulim[1]+width_true2,shape=taugrid_true.size,testval=np.random.uniform(taulim[0],taulim[1],taugrid_true.size))
        # 0.9*rng2*np.random.rand(taugrid_true.size)-0.9*rng2/2.0 + avg2
        # log_width = pm.StudentT("log_width", nu=5, mu=np.log(width_true), lam=0.5, testval=-5.0)
        log_width = pm.Uniform("log_width",lower=np.log(width_true)-1.0,upper=np.log(width_true)+1.0,testval=np.log(width_true)+0.5*np.random.rand()-0.25)
        log_width2 = pm.Uniform("log_width2",lower=np.log(width_true2)-1.0,upper=np.log(width_true2)+1.0,testval=np.log(width_true2)+0.5*np.random.rand()-0.25)
        rho = pm.Uniform("rho",lower=-1.0,upper=1.0,testval=np.random.uniform(-0.5,0.5))

        # Compute the expected n at each sample
        coefs = tt.dot(aTinv,ngrid)
        coefs2 = tt.dot(aTinv,taugrid)
        mu = tt.tensordot(coefs,term,axes=1)
        mu2 = tt.tensordot(coefs2,term,axes=1)

        # The line has some width: we're calling it a Gaussian in n,tau
        zbiv = (n_samp-mu)**2 * pm.math.exp(-2*log_width) + (tau_samp-mu2)**2 * pm.math.exp(-2*log_width2) - 2 * rho * (n_samp-mu) * (tau_samp-mu2) * pm.math.exp(-log_width-log_width2)

        logp = -0.5 * zbiv/(1.0-rho**2) - log_width - log_width2 - pm.math.log(pm.math.sqrt(1-rho**2))

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=False, discard_tuned_samples=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
            trace_df = pm.trace_to_dataframe(trace)
            randinds = np.random.randint(0,len(ngrid_true),2)
            cols_to_keep = [f'ngrid__{randinds[0]}', f'taugrid__{randinds[1]}', 'log_width','log_width2','rho']
            fig = corner.corner(trace_df[cols_to_keep], truths= [ngrid_true[randinds[0]]] + [taugrid_true[randinds[1]]] + [np.log(width_true)] + [np.log(width_true2)] + [rho_true])
            fig.savefig(op.join(img_dir,"trace%s.png"%(extratext)))
        print(az.summary(trace,round_to=2))
        print("ngrid_true:"); print(ngrid_true)
        print("taugrid_true:"); print(taugrid_true)
        print("log_width_true:"); print(np.log(width_true))
        print("log_width_true2:"); print(np.log(width_true2))
        print("rho_true:"); print(rho_true)

    return trace[tune:], a_poly_T, xx, ngrid_true, taugrid_true, coefs_true, coefs_true2, n_true, tau_true, width_true, width_true2, rho_true, indep_true, indep_samp, n_samp, tau_samp

def polyNDgen(img_dir_orig,limlist=None,size=500,samples=50,plot=False,extratext='',nlim=np.array([-1.0,0.4]),degree=2,degree2=1,sampling=1000,tune=1000,errmult=1.0,errmultn=1.0,indep_true=None,sigarr=None,sigN=0.19,n=None,limlist2=None,indep_true2=None,sigarr2=None,indep_name=None,indep_name2=None,numtrials=30000):
    ''' Simulation of data where the dust index n is a 2-D polynomial function of stellar mass and sSFR.

    Parameters
    ----------
    img_dir: Str
        Parent directory in which folders will be created for the particular run
    size: Int
        Number of "galaxies" in the sample
    samples: Int
        Number of "posterior samples" available for each galaxy
    plot: Bool
        Whether or not an arviz plot should be made
    extratext: Str
        An addition to the plot name to distinguish it from previous runs
    mlim, ssfrlim, nlim: Two-element lists
        Lower and upper limits for all parameters in question
    degree: Int
        Polynomial degree in each variable
    sampling: Int
        Number of draws for sampling (not including burn-in steps)
    tune: Int
        Number of burn-in steps for each sampling chain
    errmult: Float
        Multiplicative factor that determines how large the simulated data errors are
    errmultn: Float
        Multiplicative factor that determines how large the intrinsic dispersion in the relation is
    logM, ssfr, n: 1-D Numpy Arrays
        If provided, this will supercede the creation of "true" values for log(M), log(sSFR), and n_grid
    Returns
    -------
    trace: Pymc3 trace object
        Result of pymc3 sampler
    a_poly_T: 2-D Numpy arrays
        Matrix whose inverse times n_grid gives polynomial coefficients
    xx: List of 2-D Numpy arrays
        Meshgrid where dust index is computed
    ngrid_true: 1-D Numpy array
        True values for dust index at (coarse) grid
    coefs_true: 1-D Numpy array
        True values for polynomial coefficients
    n_true: 1-D Numpy array
        True values for dust index at the true locations of the sample
    width_true: Float
        True intrinsic dispersion in relation
    indep_true: 2-D Numpy array
        Array with each row corresponding to true values of an independent variable (log(M) and log(sSFR))
    med_logM: Float
        Median log(M) value--for adding back to indep_true to get the actual log(M) values in plots 
    med_ssfr: Float
        Median log(sSFR) value
    
    The function runs the pymc3 sampler on this toy example and prints the results; an arviz plot of the results is also made if plot==True
    '''
    # Since there are six figures to create for one run of the code, we create a directory for each run, seperated by polynomial degree (and the run-specific text)
    img_dir = op.join(img_dir_orig,'deg_%d_deg2_%d'%(degree,degree2),extratext)
    mkpath(img_dir)
    ndim, ndim2 = len(indep_name), len(indep_name2)

    indtrue, indmod = np.zeros(ndim,dtype=int), np.zeros(ndim2,dtype=int)
    for i in range(ndim):
        try: indtrue[i] = indep_name2.index(indep_name[i])
        except: indtrue[i] = -99
    for i in range(ndim2):
        try: indmod[i] = indep_name.index(indep_name2[i])
        except: indmod[i] = -99

    if nlim[1]>1.0: uniform = False
    else: uniform = True

    if limlist is None:
        limlist, limlist2 = [], []
        for indep in indep_true:
            # limlist.append(np.percentile(indep,[2.5,97.5]))
            limlist.append(np.array([min(indep),max(indep)]))
        if indep_true2 is None: indep_true2 = copy(indep_true)
        for indep in indep_true2:
            limlist2.append(np.array([min(indep),max(indep)]))
    else:
        com, ind1orig, ind2orig = np.intersect1d(limlist,limlist2,return_indices=True)
        ind1, ind2 = np.unique(ind1orig//2), np.unique(ind2orig//2)
        rand_arr = np.random.rand(max(len(limlist),len(limlist2)),size)
        indep_true, indep_true2 = np.empty((0,size)), np.empty((0,size))
        induniq = np.setdiff1d(np.arange(len(limlist2)),ind2)
        for i,lim in enumerate(limlist):
            indep_true = np.append(indep_true,(lim[1]-lim[0])*rand_arr[i:i+1]+lim[0],axis=0)
        if limlist2 is None: limlist2 = copy(limlist)
        index = 0
        for i,lim in enumerate(limlist2):
            arr = np.nonzero(ind1==i)[0]
            if len(arr): indi = ind2[arr[0]]
            else: 
                indi = induniq[index]
                index+=1
            indep_true2 = np.append(indep_true2,(lim[1]-lim[0])*rand_arr[indi:indi+1]+lim[0],axis=0)

    # Determining true parameters
    width_true = 0.1*errmultn #Intrinsic dispersion in relation
    x = np.empty((0,degree+1)) # To set up grid on which true dust parameter n will be defined
    x2 = np.empty((0,degree2+1)) # To set up grid on which model dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    for lim in limlist2:
        x2 = np.append(x2,np.linspace(lim[0],lim[-1],degree2+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    xx2 = np.meshgrid(*x2) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation for true
    a_poly_T2 = get_a_polynd(xx2).T #Array related to grid that will be used in least-squares computation for model
    aTinv = np.linalg.inv(a_poly_T)
    aTinv2 = np.linalg.inv(a_poly_T2)
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    rng = nlim[1]-nlim[0]
    avg = (nlim[1]+nlim[0])/2.0
    if n is None:
        np.random.seed(3890)
        if uniform: ngrid_true = rng*np.random.rand(xx[0].size)-rng/2.0 + avg #True values of dust parameter at the grid
        else: ngrid_true = truncnorm.rvs(-0.3,3.7,loc=0.3,scale=1.0,size=xx[0].size)
    # When "data" inputs provided, use measured n to determine approximate grid values for n according to Prospector
    else:
        ngrid_true = np.zeros(xx[0].size)
        for index, i in enumerate(np.ndindex(xx[0].shape)):
            totarr = np.zeros_like(indep_true[0])
            for j, ind in enumerate(i):
                totarr+=abs(indep_true[j]-x[j,ind])
            ngrid_true[index] = n[np.argmin(totarr)]
    coefs_true = np.linalg.lstsq(a_poly_T,ngrid_true,rc)[0] #Calculation of polynomial coefficients from the grid
    n_true = calc_poly(indep_true,coefs_true,degree) + width_true * np.random.randn(size) #True dust index parameter n is polynomial function of n with some intrinsic scatter

    # Observed values of the galaxy parameters
    if sigarr is None:
        sigarr = np.repeat(0.15,len(indep_true))*errmult
        sigarr2 = np.repeat(0.15,len(indep_true2))*errmult
    else:
        sigarr *= errmult
        sigarr2 *= errmult
    sigN *= errmult
    # First we need to "move" the mean of the posterior, then we generate posterior samples
    randmean = np.random.randn(ndim,size,1) 
    randmean2 = np.random.randn(ndim2,size,1)
    randsamp = np.random.randn(ndim,size,samples)
    randsamp2 = np.random.randn(ndim2,size,samples)
    for i in range(ndim2):
        if indmod[i]>=0: 
            randmean2[i] = randmean[indmod[i]]
            randsamp2[i] = randsamp[indmod[i]]

    indep_samp = indep_true[:,:,None] + sigarr[:,None,None]*randmean + sigarr[:,None,None]*randsamp
    indep_samp2 = indep_true2[:,:,None] + sigarr2[:,None,None]*randmean2 + sigarr2[:,None,None]*randsamp2

    n_samp = n_true[:, None] + sigN * np.random.randn(size, 1) + sigN * np.random.randn(size, samples)
    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp2,degree2)
    # breakpoint()
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        if uniform: ngrid = pm.Uniform("ngrid",lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=xx2[0].size,testval=rng*np.random.rand(xx2[0].size)-rng/2.0 + avg)
        else: ngrid = pm.TruncatedNormal("ngrid",mu=0.3,sigma=1.0,lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=xx2[0].size,testval=rng*np.random.rand(xx2[0].size)-rng/2.0 + avg)
        log_width = pm.StudentT("log_width", nu=5, mu=np.log(width_true), lam=0.5, testval=-5.0)
        # log_width = pm.Uniform("log_width",lower=np.log(width_true)-5.0,upper=np.log(width_true)+5.0,testval=-5.0)

        # Compute the expected n at each sample
        coefs = tt.dot(aTinv2,ngrid)
        mu = tt.tensordot(coefs,term,axes=1)
        # The line has some width: we're calling it a Gaussian in n
        logp = -0.5 * (n_samp - mu) ** 2 * pm.math.exp(-2*log_width) - log_width

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)
        
        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=False, chains=4, discard_tuned_samples=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
            randinds = np.random.choice(len(ngrid_true),2,replace=False)
            cols_to_keep = [f'ngrid__{min(randinds)}', f'ngrid__{max(randinds)}', 'log_width']
            fig = corner.corner(trace_df[cols_to_keep], truths= [ngrid_true[min(randinds)]] + [ngrid_true[max(randinds)]] + [np.log(width_true)])
            fig.savefig(op.join(img_dir,"trace%s.png"%(extratext)))
        print(az.summary(trace,round_to=2))
        print("ngrid_true:"); print(ngrid_true)
        print("log_width_true:"); print(np.log(width_true))
    
    return trace[tune:], a_poly_T, a_poly_T2, xx, xx2, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_true2, indep_samp, indep_samp2, n_samp

def polyNDData(indep_samp,dep_samp,logp_prior,img_dir_orig,plot=False,extratext='',degree=2,sampling=1000,tune=1000,dep_lim=np.array([-1.0,0.4]),uniform=True):
    ''' Simulation of data where the dust index n is a 2-D polynomial function of stellar mass and sSFR.

    Parameters
    ----------
    indep_samp, dep_samp: 3-D, 2-D Numpy Array
        Prospector samples for the independent and dependent variables, respectively
    img_dir_orig: Str
        Parent directory in which folders will be created for the particular run
    plot: Bool
        Whether or not an arviz plot should be made
    extratext: Str
        An addition to the plot name to distinguish it from previous runs
    degree: Int
        Polynomial degree in each variable
    sampling: Int
        Number of draws for sampling (not including burn-in steps)
    tune: Int
        Number of burn-in steps for each sampling chain
    dep_lim: Two-element Numpy Array (or list)
        Limits for dependent variable
    uniform: Bool
        Whether or not prior distribution for dependent variable should be uniform (alternative is clipped Gaussian)
    Returns
    -------
    trace: Pymc3 trace object
        Result of pymc3 sampler
    a_poly_T: 2-D Numpy arrays
        Matrix whose inverse times n_grid gives polynomial coefficients
    xx: List of 2-D Numpy arrays
        Meshgrid where dust index is computed
    
    The function runs the pymc3 sampler on the Prospector data and prints the results; an arviz plot of the results is also made if plot==True
    '''
    # We create a directory for each run, seperated by polynomial degree (and the run-specific text)
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)
    mkpath(img_dir)
    ndim = len(indep_samp)
    limlist = []
    for indep in indep_samp:
        per = np.percentile(indep,[1.0,99.0])
        # limlist.append(np.array([np.amin(indep),np.amax(indep)]))
        limlist.append(per)
    # dep_per = np.percentile(dep_samp,[0.5,99.5])
    lower, upper = min(dep_lim[0],np.amin(dep_samp)), max(dep_lim[1],np.amax(dep_samp)) # Limits for dependent variable
    x = np.empty((0,degree+1)) # To set up grid on which true dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    # breakpoint()
    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,degree)
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        if uniform: ngrid = pm.Uniform("ngrid",lower=lower-1.0e-5,upper=upper+1.0e-5,shape=xx[0].size,testval=np.random.uniform(lower,upper,xx[0].size))
        else: ngrid = pm.TruncatedNormal("ngrid",mu=0.3,sigma=1.0,lower=lower-1.0e-5,upper=upper+1.0e-5,shape=xx[0].size,testval=np.random.uniform(lower,upper/2.0,xx[0].size))
        log_width = pm.StudentT("log_width", nu=5, mu=-4.5, sigma=0.5, testval=-5.0)
        # log_width = pm.Uniform("log_width",lower=-7.0,upper=-5.0,testval=-6.0)

        # Compute the expected n at each sample
        coefs = tt.dot(aTinv,ngrid)
        mu = tt.tensordot(coefs,term,axes=1)

        # The line has some width: we're calling it a Gaussian in n
        logp = -0.5 * (dep_samp - mu) ** 2 * pm.math.exp(-2*log_width) - log_width - logp_prior

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=True, chains=4, discard_tuned_samples=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
            trace_df2 = trace.to_dataframe(groups='posterior')
            trace_df = trace.to_dataframe(groups='warmup_posterior')
            trace_df = trace_df.append(trace_df2,ignore_index=True)
            randinds = np.random.choice(xx[0].size,2,replace=False)
            lr, hr = min(randinds), max(randinds)
            cols_to_keep = [(f'ngrid[{lr}]', lr), (f'ngrid[{hr}]', hr), 'log_width']
            fig = corner.corner(trace_df[cols_to_keep])
            fig.savefig(op.join(img_dir,"trace_with_tune%s.png"%(extratext)))
            fig = corner.corner(trace_df2[cols_to_keep])
            fig.savefig(op.join(img_dir,"trace%s.png"%(extratext)))
        print(az.summary(trace,round_to=2))

    return trace, a_poly_T, xx

def polyNDDataBivar(indep_samp,n_samp,tau_samp,logp_prior,img_dir_orig,plot=False,extratext='',degree=2,sampling=1000,tune=1000,nlim=np.array([-1.0,0.4]),taulim=np.array([0.0,4.0])):
    ''' Simulation of data where the dust index n is a 2-D polynomial function of stellar mass and sSFR.

    Parameters
    ----------
    indep_samp, dep_samp: 3-D, 2-D Numpy Array
        Prospector samples for the independent and dependent variables, respectively
    img_dir_orig: Str
        Parent directory in which folders will be created for the particular run
    plot: Bool
        Whether or not an arviz plot should be made
    extratext: Str
        An addition to the plot name to distinguish it from previous runs
    degree: Int
        Polynomial degree in each variable
    sampling: Int
        Number of draws for sampling (not including burn-in steps)
    tune: Int
        Number of burn-in steps for each sampling chain
    dep_lim: Two-element Numpy Array (or list)
        Limits for dependent variable
    uniform: Bool
        Whether or not prior distribution for dependent variable should be uniform (alternative is clipped Gaussian)
    Returns
    -------
    trace: Pymc3 trace object
        Result of pymc3 sampler
    a_poly_T: 2-D Numpy arrays
        Matrix whose inverse times n_grid gives polynomial coefficients
    xx: List of 2-D Numpy arrays
        Meshgrid where dust index is computed
    
    The function runs the pymc3 sampler on the Prospector data and prints the results; an arviz plot of the results is also made if plot==True
    '''
    # We create a directory for each run, seperated by polynomial degree (and the run-specific text)
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)
    mkpath(img_dir)
    ndim = len(indep_samp)
    limlist = []
    for indep in indep_samp:
        per = np.percentile(indep,[1.0,99.0])
        # limlist.append(np.array([np.amin(indep),np.amax(indep)]))
        limlist.append(per)
    x = np.empty((0,degree+1)) # To set up grid on which true dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    # nchev = getChevFunc()
    # nchev_tt = pm.distributions.dist_math.SplineWrapper(nchev)
    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,degree)
    # breakpoint()
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        a = pm.Beta("a",alpha=0.5,beta=0.5)
        # ngrid = pm.Uniform("ngrid",lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=xx[0].size,testval=np.random.uniform(nlim[0],nlim[1],xx[0].size))
        taugrid = pm.TruncatedNormal("taugrid",mu=0.3,sigma=1.0,lower=taulim[0]-1.0e-5,upper=taulim[1]+1.0e-5,shape=xx[0].size,testval=np.random.uniform(taulim[0],taulim[1]/2.0,xx[0].size))
        mu_n = a*(0.9487*tt.log10(tt.log10(2.6119*taugrid+1.0108))+0.3553)
        # sig_n = 3.0/(29.0*a+1)
        sig_n = pm.InverseGamma("sig_n",alpha=5,beta=0.5,testval=0.05)
        # mu_n = pm.Deterministic("mu_n",a*(0.9487*tt.log10(tt.log10(2.6119*taugrid+1.0108))+0.3553))
        # sig_n = pm.Deterministic("sig_n",3.0/(29.0*a+1))
        ngrid = pm.TruncatedNormal("ngrid",mu=mu_n,sigma=sig_n,lower=nlim[0],upper=nlim[1],shape=xx[0].size)
        log_width = pm.StudentT("log_width", nu=5, mu=-3.0, lam=0.5, testval=-3.5)
        log_width2 = pm.StudentT("log_width2", nu=5, mu=-2.9, lam=0.48, testval=-3.8)
        rho = pm.Uniform("rho",lower=-1.0,upper=1.0,testval=np.random.uniform(-0.5,0.5))

        # Compute the expected n at each sample
        coefs = tt.dot(aTinv,ngrid)
        coefs2 = tt.dot(aTinv,taugrid)
        mu = tt.tensordot(coefs,term,axes=1)
        mu2 = tt.tensordot(coefs2,term,axes=1)

        # The line has some width: we're calling it a Gaussian in n,tau
        zbiv = (n_samp-mu)**2 * pm.math.exp(-2*log_width) + (tau_samp-mu2)**2 * pm.math.exp(-2*log_width2) - 2 * rho * (n_samp-mu) * (tau_samp-mu2) * pm.math.exp(-log_width-log_width2)

        logp = -0.5 * zbiv/(1.0-rho**2) - log_width - log_width2 - pm.math.log(pm.math.sqrt(1-rho**2)) - logp_prior

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=True, chains=4, discard_tuned_samples=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
            trace_df2 = trace.to_dataframe(groups='posterior')
            trace_df = trace.to_dataframe(groups='warmup_posterior')
            trace_df.append(trace_df2,ignore_index=True)
            randinds = np.random.randint(0,xx[0].size,2)
            lr, hr = randinds[0], randinds[1]
            cols_to_keep = [(f'ngrid[{lr}]', lr), (f'taugrid[{hr}]', hr), 'log_width','log_width2','rho']
            fig = corner.corner(trace_df[cols_to_keep])
            fig.savefig(op.join(img_dir,"trace_with_tune%s.png"%(extratext)))
            fig = corner.corner(trace_df2[cols_to_keep])
            fig.savefig(op.join(img_dir,"trace%s.png"%(extratext)))
        print(az.summary(trace,round_to=2))

    return trace, a_poly_T, xx

def polyNDDataBivarOrig(indep_samp,n_samp,tau_samp,logp_prior,img_dir_orig,plot=False,extratext='',degree=2,sampling=1000,tune=1000,nlim=np.array([-1.0,0.4]),taulim=np.array([0.0,4.0])):
    ''' Simulation of data where the dust index n is a 2-D polynomial function of stellar mass and sSFR.

    Parameters
    ----------
    indep_samp, dep_samp: 3-D, 2-D Numpy Array
        Prospector samples for the independent and dependent variables, respectively
    img_dir_orig: Str
        Parent directory in which folders will be created for the particular run
    plot: Bool
        Whether or not an arviz plot should be made
    extratext: Str
        An addition to the plot name to distinguish it from previous runs
    degree: Int
        Polynomial degree in each variable
    sampling: Int
        Number of draws for sampling (not including burn-in steps)
    tune: Int
        Number of burn-in steps for each sampling chain
    dep_lim: Two-element Numpy Array (or list)
        Limits for dependent variable
    uniform: Bool
        Whether or not prior distribution for dependent variable should be uniform (alternative is clipped Gaussian)
    Returns
    -------
    trace: Pymc3 trace object
        Result of pymc3 sampler
    a_poly_T: 2-D Numpy arrays
        Matrix whose inverse times n_grid gives polynomial coefficients
    xx: List of 2-D Numpy arrays
        Meshgrid where dust index is computed
    
    The function runs the pymc3 sampler on the Prospector data and prints the results; an arviz plot of the results is also made if plot==True
    '''
    # We create a directory for each run, seperated by polynomial degree (and the run-specific text)
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)
    mkpath(img_dir)
    ndim = len(indep_samp)
    limlist = []
    for indep in indep_samp:
        per = np.percentile(indep,[1.0,99.0])
        # limlist.append(np.array([np.amin(indep),np.amax(indep)]))
        limlist.append(per)
    x = np.empty((0,degree+1)) # To set up grid on which true dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    # breakpoint()
    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,degree)
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        ngrid = pm.Uniform("ngrid",lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=xx[0].size,testval=np.random.uniform(nlim[0],nlim[1],xx[0].size))
        taugrid = pm.TruncatedNormal("taugrid",mu=0.3,sigma=1.0,lower=taulim[0]-1.0e-5,upper=taulim[1]+1.0e-5,shape=xx[0].size,testval=np.random.uniform(taulim[0],taulim[1]/2.0,xx[0].size))
        log_width = pm.StudentT("log_width", nu=5, mu=-3.0, lam=0.5, testval=-3.5)
        log_width2 = pm.StudentT("log_width2", nu=5, mu=-2.9, lam=0.48, testval=-3.8)
        rho = pm.Uniform("rho",lower=-1.0,upper=1.0,testval=np.random.uniform(-0.5,0.5))

        # Compute the expected n at each sample
        coefs = tt.dot(aTinv,ngrid)
        coefs2 = tt.dot(aTinv,taugrid)
        mu = tt.tensordot(coefs,term,axes=1)
        mu2 = tt.tensordot(coefs2,term,axes=1)

        # The line has some width: we're calling it a Gaussian in n,tau
        zbiv = (n_samp-mu)**2 * pm.math.exp(-2*log_width) + (tau_samp-mu2)**2 * pm.math.exp(-2*log_width2) - 2 * rho * (n_samp-mu) * (tau_samp-mu2) * pm.math.exp(-log_width-log_width2)

        logp = -0.5 * zbiv/(1.0-rho**2) - log_width - log_width2 - pm.math.log(pm.math.sqrt(1-rho**2)) - logp_prior

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=True, chains=4, discard_tuned_samples=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
            trace_df2 = trace.to_dataframe(groups='posterior')
            trace_df = trace.to_dataframe(groups='warmup_posterior')
            trace_df.append(trace_df2,ignore_index=True)
            randinds = np.random.randint(0,xx[0].size,2)
            lr, hr = randinds[0], randinds[1]
            cols_to_keep = [(f'ngrid[{lr}]', lr), (f'taugrid[{hr}]', hr), 'log_width','log_width2','rho']
            fig = corner.corner(trace_df[cols_to_keep])
            fig.savefig(op.join(img_dir,"trace_with_tune%s.png"%(extratext)))
            fig = corner.corner(trace_df2[cols_to_keep])
            fig.savefig(op.join(img_dir,"trace%s.png"%(extratext)))
        print(az.summary(trace,round_to=2))

    return trace, a_poly_T, xx

def plot1D(x,y,xx,yy,img_dir,name,xlab=r'$\log\ M_*$',ylab='Median model n',xerr=None,yerr=None,ngrid_err=None):
    fig, ax = plt.subplots()
    indsort = np.argsort(x)
    ax.plot(x[indsort],y[indsort],'b-.',markersize=1)
    ax.plot(xx,yy,'r^',markersize=2)
    if xerr is not None or yerr is not None: ax.errorbar(x,y,yerr=yerr,xerr=xerr,fmt='none',ecolor='b',alpha=np.sqrt(1.0/len(x)))
    if ngrid_err is not None: ax.errorbar(xx,yy,yerr=ngrid_err,fmt='none',ecolor='r',alpha=1.0)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    fig.savefig(op.join(img_dir,'%s.png'%(name)),bbox_inches='tight',dpi=200)

def plot_color_map(x,y,z,xdiv,ydiv,znum,xx,yy,img_dir,name,levels=10,xlab=r'$\log\ M_*$',ylab=r'$\log$ sSFR$_{\rm{100}}$',zlab='Median model n',cmap='viridis',xtrue=None,ytrue=None,xtrueerr=None,ytrueerr=None,minz=-100.0,maxz=100.0,width_mean=None,vmin=None,vmax=None):
    """ Plot color map of z(x,y) and over-plot contours showing the number of sources 
    
    Parameters:
    -----------
        x,y,z: 2-D Numpy Arrays
            x and y specify the grid (meshgrid works best); z specifies the value of the function at all points of the grid
        xdiv,ydiv,znum: 2-D Numpy Arrays
            xdiv and ydiv specify the (coarse) grid; z specifies the number of objects within each grid square
        img_dir: Str
            Image directory to create file
        name: Str
            Name for the image (without extension)
        levels: Int
            Number of filled contours for color plot
        xlab,ylab,zlab: Str
            xlabel, ylabel, and colorbar label
        cmap: Str
            Name of color map
    """
    fig, ax = plt.subplots()
    ax.plot(xx,yy,'ro',markersize=6)
    if cmap=='viridis': 
        if vmin is None or vmax is None: vmin, vmax = max(np.amin(z),minz), min(np.amax(z),maxz)
        # if width_mean is not None: vmax = max(vmax,width_mean)
        cf = ax.contourf(x,y,z,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax)
    else: 
        if vmin is None or vmax is None: vmin, vmax = -1.0*np.amax(abs(z)), np.amax(abs(z))
        cf = ax.contourf(x,y,z,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax)
    xmin, xmax = np.amin(xx),np.amax(xx)
    ymin, ymax = np.amin(yy),np.amax(yy)
    # if np.sum(znum)>2501:
    #     cnum = ax.contour(xdiv,ydiv,znum,levels=4,cmap='Greys')
    #     if xtrueerr is not None:
    #         if type(xtrueerr) is float: xe = [xtrueerr]
    #         else: xe = [np.mean(xtrueerr)]
    #         if type(ytrueerr) is float: ye = [ytrueerr]
    #         else: ye = [np.mean(ytrueerr)]
    #         plt.errorbar([xmin+0.15*(xmax-xmin)],[ymax-0.15*(ymax-ymin)],yerr=ye,xerr=xe,fmt='none',ecolor='k',elinewidth=1.5,capsize=2)
    # else:
    if xtrue is not None: 
        plt.plot(xtrue,ytrue,'k,',alpha=0.8)
    if xtrueerr is not None:
        plt.errorbar(xtrue,ytrue,yerr=ytrueerr,xerr=xtrueerr,fmt='none',ecolor='k',alpha=2.0/xtrue.size)
    # breakpoint()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    cb = fig.colorbar(cf,label=zlab)
    if width_mean is not None:
        ticks = cb.get_ticks()
        tick_labels = np.array([ "{:0.2f}".format(tk) for tk in ticks ])
        ind_wm = np.searchsorted(ticks,width_mean)
        ticks = np.insert(ticks,ind_wm,width_mean)
        tick_labels = np.insert(tick_labels,ind_wm,'IW')
        cb.set_ticks(ticks); cb.set_ticklabels(tick_labels)
    # if np.sum(znum)>2501:
    #     fig.colorbar(cnum,label='Number of Galaxies')
    fig.savefig(op.join(img_dir,'%s.png'%(name)),bbox_inches='tight',dpi=300)

def plot_hist_resid(resid_norm,img_dir,name,bins=30,alpha=0.8,lw=3):
    """ Plot the histogram of a given quantity resid_norm
    
    Parameters:
    -----------
    resid_norm: Numpy ndarray
        Array with given quantity
    img_dir: Str
        Directory in which to place image
    name: Str
        Name for image (without extension)
    bins: Int
        Number of bins in histogram
    alpha: Float
        Opacity of histogram bins
    """
    plt.figure()
    mu, sig = norm.fit(resid_norm)
    xplot = np.linspace(min(resid_norm),max(resid_norm),101)
    plt.hist(resid_norm,bins=bins,color='b',density=True,alpha=alpha,label='')
    plt.plot(xplot,norm.pdf(xplot,loc=mu,scale=sig),'r-',lw=lw,label=r'$\mu=%.3f$, $\sigma=%.3f$'%(mu,sig))
    plt.plot(xplot,norm.pdf(xplot,loc=0.0,scale=1.0),'k--',label='Standard Normal')
    plt.xlabel(r'$\frac{n_{\rm model}-n_{\rm true}}{\sigma_n}$')
    plt.ylabel("Density of Sources")
    plt.legend(loc='best',frameon=False)
    plt.savefig(op.join(img_dir,'%s.png'%(name)),bbox_inches='tight',dpi=200)

def plot_model_true(n_true,n_med,ngrid_true,ngrid_med,img_dir,name,n_err=None,width_true=0.1,all_std=None,ylab='Median model n',xlab='True n',ngrid_true_plot=True):
    """ Plot model vs true values

    Parameters:
    -----------
    n_true: 1-D Numpy array
        True values of dependent variable
    n_med: 1-D Numpy array
        Model values of dependent variable
    ngrid_true: 1-D Numpy array
        True values of dependent variable at certain grid points
    ngrid_med: 1-D Numpy array
        Model values of dependent variable at grid points (the actual parameters being fit)
    img_dir: Str
        Directory in which to place image
    name: Str
        Name for image (without extension)
    n_err: 1-D Numpy array (optional)
        Array of errors for model n
    width_true: Float or 1-D Numpy array (optional)
        Float/array of error(s) for true n
    all_std: 1-D Numpy array (optional)
        Array of errors for model n_grid
    ylab: Str
        Label for y-axis
    """
    plt.figure()
    if len(n_true)>=2500: marker, markersize = ',',None
    else: marker, markersize = '.', 2
    plt.plot(n_true,n_med,color='b',marker=marker,markersize=markersize,linestyle='none',label='Input values')
    if n_err is not None or width_true is not None: 
        if len(n_true)>=2500: plt.errorbar(n_true,n_med,yerr=n_err,xerr=width_true,fmt='none',ecolor='k',label='',alpha=0.5/np.sqrt(n_true.size))
        else: plt.errorbar(n_true,n_med,yerr=n_err,xerr=width_true,fmt='none',ecolor='b',label='',alpha=0.1)
    if ngrid_true_plot: plt.errorbar(ngrid_true,ngrid_med,yerr=all_std,fmt='r^',markersize=6,label='Grid')
    xmin, xmax = plt.gca().get_xlim()
    xplot = np.linspace(xmin,xmax,101)
    plt.plot(xplot,xplot,'k--',lw=2,label='1-1')
    plt.gca().set_xlim(xmin,xmax)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc='best',frameon=False)
    # plt.legend(loc='best',frameon=False)
    plt.savefig(op.join(img_dir,"%s.png"%(name)),bbox_inches='tight',dpi=150)

def plot_percentile(n_sim,n_true,img_dir,name,testper=29,endper=97.5):
    """ General method to compare distributions of model parameters and true parameters (in a percentile plot)

    Parameters:
    -----------
    n_sim: 2-D Numpy array
        Posterior samples of n at all data points; samples are along outer axis
    n_true: 1-D Numpy array
        True values of n at all data points
    img_dir: Str
        Directory in which to place image
    name: Str
        Name for image (without extension)
    """
    assert testper%2==1 # Want an odd number of percentiles so middle will be 50
    begper = 100.0-endper # Symmetric about median
    perarr = np.linspace(begper,endper,testper) # Array of percentiles
    nsim_per = np.percentile(n_sim,perarr,axis=0) #n_sim at the given percentiles
    ind_cent = testper//2 # Actual number of points for plot; also the index of the median in the percentiles array
    model_frac, true_frac = np.zeros(ind_cent), np.zeros(ind_cent)
    # Find the fraction of true sources
    for i in range(ind_cent):
        cond = np.logical_and(n_true>nsim_per[ind_cent-i-1],n_true<=nsim_per[ind_cent+i+1])
        model_frac[i] = (perarr[ind_cent+i+1]-perarr[ind_cent-i-1])/100.0
        true_frac[i] = float(len(n_true[cond]))/len(n_true)

    plt.figure()
    plt.plot(model_frac,true_frac,'b-o',label='')
    xmin, xmax = plt.gca().get_xlim()
    xplot = np.linspace(xmin,xmax,101)
    plt.plot(xplot,xplot,'r--',label='1-1')
    plt.gca().set_xlim(xmin,xmax)
    plt.legend(loc='best',frameon=False)
    plt.xlabel('Fraction of model around median')
    plt.ylabel('Fraction of true encapsulated')
    plt.savefig(op.join(img_dir,'%s.png'%(name)),bbox_inches='tight',dpi=200)

def get_relevant_info_ND_Gen(trace,a_poly_T2,xx_true,xx,ngrid_true,coefs_true,n_true,width_true,indep_true,indep_true2,indep_samp,indep_samp2,n_samp,med_arr_true,med_arr,indep_name_true,indep_name,indep_lab,dep_name='n',dep_lab='n',degree=2,degree2=1,numsamp=50,levels=10,extratext='',fine_grid=201,bins=20,img_dir_orig=op.join('DataSim','2DTests'),grid_name='ngrid',width_name='log_width',bivar=False):
    nlen = len(n_true)
    ndim, ndim2 = len(indep_true), len(indep_true2)
    if not bivar: img_dir = op.join(img_dir_orig,'deg_%d_deg2_%d'%(degree,degree2),extratext)
    else: img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)

    # Determine median model values for the dependent variable
    ngrid_med = np.median(getattr(trace,grid_name),axis=0)
    width_med = np.exp(np.median(getattr(trace,width_name)))
    width_mean = np.exp(np.mean(getattr(trace,width_name)))
    all_std = np.std(getattr(trace,grid_name),axis=0)
    coefs_med = np.linalg.lstsq(a_poly_T2,ngrid_med,None)[0]
    n_med = calc_poly(indep_true2,coefs_med,degree2) + width_med * np.random.randn(nlen)
    assert np.all(abs(calc_poly(xx_true,coefs_true,degree).ravel()-ngrid_true)<1.0e-6) # Make sure ngrid_true was properly computed

    # Print likelihoods of median model, best model, and true values
    log_width_true, log_width_med = np.log(width_true), np.log(width_med)
    mu = calc_poly(indep_samp,coefs_true,degree)
    mu_med = calc_poly(indep_samp2,coefs_med,degree2)
    logp = -0.5 * (n_samp - mu) ** 2 * np.exp(-2*log_width_true) - log_width_true
    logp_med = -0.5 * (n_samp - mu_med) ** 2 * np.exp(-2*log_width_med) - log_width_med
    max_logp, max_logp_med = np.max(logp, axis=1), np.max(logp_med, axis=1)
    marg_logp = max_logp + np.log(np.sum(np.exp(logp - max_logp[:, None]), axis=1))
    marg_logp_med = max_logp_med + np.log(np.sum(np.exp(logp_med - max_logp_med[:, None]), axis=1))

    print("Likelihood of true solution: %.4f"%(np.sum(marg_logp)))
    print("Likelihood of median NUTS solution: %.4f"%(np.sum(marg_logp_med)))
    print("Likelihood of best NUTS solution: %.4f"%(np.max(trace.model_logp)))

    # Evaluate the error/residuals at the input (data) points
    inds = np.random.choice(len(trace.log_width),size=numsamp,replace=False)
    coefs_mod_all = np.empty((numsamp,coefs_med.size))
    width_mod_all = np.empty(numsamp)
    n_sim = np.empty((numsamp,nlen))
    for i, ind in enumerate(inds):
        ngrid_mod = getattr(trace,grid_name)[ind]
        width_mod = np.exp(getattr(trace,width_name)[ind])
        coefs_mod = polyfitnd(xx,ngrid_mod)
        coefs_mod_all[i] = coefs_mod
        width_mod_all[i] = width_mod
        n_sim[i] = calc_poly(indep_true2,coefs_mod,degree2) + width_mod * np.random.randn(*n_sim[i].shape)
    n_mean, n_err = np.mean(n_sim,axis=0), np.std(n_sim,axis=0)
    n_quad_err = np.sqrt(n_err**2+width_true**2)
    # n_quad_err = np.sqrt(width_med**2+width_true**2)
    resid_norm = (n_med-n_true)/n_quad_err

    # Next plot compares model to the truth very simply
    if degree2==degree and indep_name_true==indep_name: ngrid_true_plot = True
    else: ngrid_true_plot = False
    plot_model_true(n_true,n_mean,ngrid_true,ngrid_med,img_dir,'PolyND_%s%s'%(dep_name,extratext),n_err=n_err,width_true=width_true,all_std=all_std,ylab='Mean posterior model %s'%(dep_lab),xlab='True %s'%(dep_lab),ngrid_true_plot=ngrid_true_plot)

    # Histogram of residuals/error and fit a normal distribution to them--ideal result is standard normal
    plot_hist_resid(resid_norm,img_dir,'hist_resid_sample_'+dep_name+extratext)

    # Plot general percentile comparison
    plot_percentile(n_sim,n_true,img_dir,'percentile_'+dep_name+extratext)

    if ndim2==1:
        x = np.linspace(min(indep_true2[0]),max(indep_true2[0]),fine_grid)
        # y = calc_poly(x[None,:],coefs_med,degree2)
        n_sim2 = np.zeros((numsamp,fine_grid))
        for i, ind in enumerate(inds):
            n_sim2[i] = calc_poly(x[None,:],coefs_mod_all[i],degree2) + width_mod_all[i] * np.random.randn(*n_sim2[i].shape)
        plot1D(x,np.mean(n_sim2,axis=0),xx[0].ravel(),ngrid_med,img_dir,'Sim_Model_%s_%s_%s'%(indep_name[0],dep_name,extratext),xlab=indep_lab[0],ylab='Mean posterior model '+dep_lab,xerr=None,yerr=np.std(n_sim2,axis=0),ngrid_err=all_std)
    if ndim==1: 
        x = np.linspace(min(indep_true[0]),max(indep_true[0]),fine_grid)
        y = calc_poly(x[None,:],coefs_true,degree)
        plot1D(x,y,xx_true[0].ravel(),ngrid_true,img_dir,'Sim_True_%s_%s_%s'%(indep_name_true[0],dep_name,extratext),xlab=indep_lab[0],ylab='True '+dep_lab)
        
    if ndim<2 or ndim2<2: return

    # 2-D (or 2+D-dependent) plots! First three are color maps of the model results for the dependent variable as a function of the independent variables and of the residuals (with and without division of the error)
    # Initialization of grids to use
    med_mod, med_true = np.median(indep_true2,axis=1), np.median(indep_true,axis=1)
    indep_div, indep_avg = np.ones((2,bins+1)), np.ones((2,bins))
    znum = np.zeros((bins,bins),dtype=int) # Will fill with numbers of galaxies
    for i in range(ndim2):
        namei = indep_name[i]
        for j in range(i+1,ndim2):
            namej = indep_name[j]
            if namei not in indep_name_true or namej not in indep_name_true: continue 
            indtruei, indtruej = indep_name_true.index(namei), indep_name_true.index(namej)

            indep_fine, indep_fine_true = med_mod[:,None]*np.ones((ndim2,fine_grid)), med_true[:,None]*np.ones((ndim,fine_grid))
            # Define grids of the independent variables: a fine one (xx_fine) and a coarse one (xx_div)
            index = 0
            for k,l in zip([i,j],[indtruei,indtruej]):
                indep_fine[k] = np.linspace(min(indep_true2[k]),max(indep_true2[k]),fine_grid)
                indep_fine_true[l] = indep_fine[k]
                indep_div[index] = np.linspace(min(indep_true2[k]),max(indep_true2[k])+1.0e-8,bins+1)
                indep_avg[index] = np.linspace((indep_div[index][0]+indep_div[index][1])/2.0,(indep_div[index][-1]+indep_div[index][-2])/2.0,bins)
                index+=1

            xx_fine = np.meshgrid(*indep_fine[(i,j),:])
            xx_fine_true = np.meshgrid(*indep_fine_true[(indtruei,indtruej),:])
            xx_div = np.meshgrid(*indep_avg)

            for k in range(i-1,-1,-1):
                medval = indep_fine[k,0]
                xx_fine.insert(0,medval*np.ones_like(xx_fine[0]))
            for k in range(i+1,j):
                medval = indep_fine[k,0]
                xx_fine.insert(-1,medval*np.ones_like(xx_fine[0]))
            for k in range(j+1,ndim2):
                medval = indep_fine[k,0]
                xx_fine.insert(k,medval*np.ones_like(xx_fine[0]))

            for k in range(indtruei-1,-1,-1):
                medval = indep_fine_true[k,0]
                xx_fine_true.insert(0,medval*np.ones_like(xx_fine_true[0]))
            for k in range(indtruei+1,indtruej):
                medval = indep_fine_true[k,0]
                xx_fine_true.insert(-1,medval*np.ones_like(xx_fine_true[0]))
            for k in range(indtruej+1,ndim):
                medval = indep_fine_true[k,0]
                xx_fine_true.insert(k,medval*np.ones_like(xx_fine_true[0]))
            # Evaluate the model and error/residuals at the fine grid
            n_quad_err_fine = np.sqrt(width_med**2+width_true**2)
            n_med_fine = calc_poly(xx_fine,coefs_med,degree2)
            n_true_fine = calc_poly(xx_fine_true,coefs_true,degree)
            # Determine number of galaxies in each coarse bin square
            for l in range(bins):
                condj = np.logical_and(indep_true2[j]>=indep_div[1][l],indep_true2[j]<indep_div[1][l+1])
                for k in range(bins):
                    condi = np.logical_and(indep_true2[i]>=indep_div[0][k],indep_true2[i]<indep_div[0][k+1])
                    cond = np.logical_and(condi,condj)
                    znum[l,k] = len(n_true[cond])

            n_sim2 = np.zeros((numsamp,fine_grid,fine_grid))
            for ii, ind in enumerate(inds):
                n_sim2[ii] = calc_poly(xx_fine,coefs_mod_all[ii],degree2) + width_mod_all[ii] * np.random.randn(*n_sim2[ii].shape)
            mmod, stdmod = np.mean(n_sim2,axis=0), np.std(n_sim2,axis=0)

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],mmod,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'MeanPost_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='Mean posterior model %s'%(dep_lab),xtrue=indep_true2[i]+med_arr[i],ytrue=indep_true2[j]+med_arr[j],xtrueerr=np.std(indep_samp2[i],axis=1),ytrueerr=np.std(indep_samp2[j],axis=1))

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_true_fine,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx_true[indtruei].ravel()+med_arr_true[indtruei],xx_true[indtruej].ravel()+med_arr_true[indtruej],img_dir,'True_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='True %s'%(dep_lab),xtrue=indep_true[indtruei]+med_arr_true[indtruei],ytrue=indep_true[indtruej]+med_arr_true[indtruej],xtrueerr=np.std(indep_samp[indtruei],axis=1),ytrueerr=np.std(indep_samp[indtruej],axis=1))

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],mmod-n_true_fine,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'Residv2_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],cmap='bwr',zlab=r'$%s_{\rm model}-%s_{\rm true}$'%(dep_name,dep_name),xtrue=indep_true2[i]+med_arr[i],ytrue=indep_true2[j]+med_arr[j],xtrueerr=np.std(indep_samp2[i],axis=1),ytrueerr=np.std(indep_samp2[j],axis=1))

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],stdmod/width_mean,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'ModErr_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='Model %s uncertainty / intrinsic width'%(dep_lab),xtrue=indep_true2[i]+med_arr[i],ytrue=indep_true2[j]+med_arr[j],xtrueerr=np.std(indep_samp2[i],axis=1),ytrueerr=np.std(indep_samp2[j],axis=1))

def get_relevant_info_ND_Data(trace,a_poly_T2,xx,indep_samp,n_samp,med_arr,indep_name,indep_lab,dep_name='n',dep_lab='n',degree2=1,levels=10,extratext='',fine_grid=201,bins=20,img_dir_orig=op.join('DataTrue','NewTests'),grid_name='ngrid',width_name='log_width',numsamp=50,fl_stats=fl_fit_stats_univar,append_to_file=True,correction=False,rand1=None,rand11=None,rand_add=True,bivar=False,tau_samp=None,indep_pickle_name=None):
    nlen, nwid = len(n_samp), len(n_samp[0])
    ndim2 = len(indep_samp)
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree2),extratext)

    # Determine median model values for the dependent variable
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,grid_name)), np.array(getattr(trace.posterior,width_name))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],sh[2]), log_width_0.reshape(sh[0]*sh[1])
    if bivar:
        taugrid_0, log_width2_0 = np.array(getattr(trace.posterior,'taugrid')), np.array(getattr(trace.posterior,width_name))
        taugrid, log_width2 = taugrid_0.reshape(sh[0]*sh[1],sh[2]), log_width2_0.reshape(sh[0]*sh[1])
        taugrid_med = np.median(taugrid,axis=0)
        rho_0 = np.array(getattr(trace.posterior,'rho'))
        rho = rho_0.reshape(sh[0]*sh[1])
        a_0, sig_n_0 = np.array(getattr(trace.posterior,'a')), np.array(getattr(trace.posterior,'sig_n'))
        a, sig_n = a_0.reshape(sh[0]*sh[1]), sig_n_0.reshape(sh[0]*sh[1])
    else:
        taugrid, log_width2, rho = None, None, None
    ngrid_med = np.median(ngrid,axis=0)
    width_med = np.exp(np.median(log_width))
    width_mean = np.exp(np.mean(log_width))
    all_std = np.std(ngrid,axis=0)
    if a_poly_T2 is not None: coefs_med = np.linalg.lstsq(a_poly_T2,ngrid_med,None)[0]
    else: coefs_med = polyfitnd(xx,ngrid_med)
    n_med = calc_poly(indep_samp,coefs_med,degree2)
    indep_avg = np.mean(indep_samp,axis=2)

    if append_to_file:
        dataarr = np.empty((ngrid_med.size,0))
        colnames = []
        for i in range(len(xx)):
            dataarr = np.append(dataarr,xx[i].reshape(ngrid_med.size,1),axis=1)
            dataarr = np.append(dataarr,xx[i].reshape(ngrid_med.size,1)+med_arr[i],axis=1)
            colnames.append(indep_name[i]); colnames.append(indep_name[i]+'_plus_med')
        dataname = ''
        for name in indep_name: dataname += name+'_'
        if 'univar' in fl_stats: 
            dataarr = np.append(dataarr,ngrid_med[:,None],axis=1); colnames.append(grid_name)
            dataname+=dep_name
        else: 
            dataarr = np.append(dataarr,ngrid_med[:,None],axis=1); colnames.append('ngrid')
            dataarr = np.append(dataarr,taugrid_med[:,None],axis=1); colnames.append('taugrid')
            dataname+='n_dust2'
        # np.savetxt(op.join(img_dir,'ngrid_%s%s_%s_vi%d.dat'%(dataname,dep_name,extratext,var_inf)),dataarr,header='%s  ngrid'%('  '.join(indep_name)),fmt='%.5f')
        t = Table(dataarr,names=colnames)
        t.write(op.join(img_dir,'ngrid_%s_%s_HB.dat'%(dataname,extratext)),overwrite=True,format='ascii')
        if not correction: trace.to_netcdf(op.join(img_dir,'trace_%s_%s.nc'%(dataname,extratext)))
        # pm.save_trace(trace,directory=img_dir)
        rhat_arr = []
        rhats = az.rhat(trace)
        rhat_keys = list(rhats.keys())
        for key in rhat_keys:
            val = rhats[key].values
            if val.ndim==0: rhat_arr+=[float(val)]
            else: rhat_arr+=[max(val)]
        num_div = len(trace.sample_stats.diverging.values.nonzero()[0])
        print("rhat_keys:", rhat_keys)
        rhat_argsort = np.argsort(rhat_keys)
        with open(fl_stats,'a') as fl:
            fl.write('%s_%s  '%(dataname, extratext))
            fl.write(f'{degree2}  {ndim2}  {nlen}  {nwid}  {len(log_width)}  {num_div}  ')
            for i_as in rhat_argsort: fl.write('%.4f  '%(rhat_arr[i_as]))
            if 'bivar' in fl_stats: fl.write('%.4f  %.4f  %.4f  %.4f  %.4f \n'%(np.log(width_med),np.median(log_width2),np.median(rho),np.median(a),np.median(sig_n)))
            else: fl.write('%.4f \n'%(np.log(width_med)))
    else: 
        dataname = ''
        for name in indep_name: dataname += name+'_'
        if bivar: dataname += 'n_dust2'
        else: dataname += dep_name

    n_sim, tau_sim = getModelSamples(xx, indep_avg, ngrid, log_width, taugrid, log_width2, rho, numsamp=numsamp, poly_only=not rand_add)
    n_mean, n_err = np.mean(n_sim,axis=0), np.std(n_sim,axis=0)
    plot_model_true(np.mean(n_samp,axis=1),n_mean,None,None,img_dir,'Real_%s_comp_%s'%(dep_name,extratext),n_err=n_err,width_true=np.std(n_samp,axis=1),ylab='Mean posterior model %s (at mean location)'%(dep_lab), xlab='Observed %s (Prospector posterior means)'%(dep_lab),ngrid_true_plot=False)
    if bivar:
        tau_mean, tau_err = np.mean(tau_sim,axis=0), np.std(tau_sim,axis=0)
        plot_model_true(np.mean(tau_samp,axis=1),tau_mean,None,None,img_dir,'Real_d2_comp_%s'%(extratext),n_err=n_err,width_true=np.std(tau_samp,axis=1),ylab='Mean posterior model %s (at mean location)'%(taulab), xlab='Observed %s (Prospector posterior means)'%(taulab),ngrid_true_plot=False)

    if ndim2==1:
        x = np.linspace(np.amin(xx[0]),np.amax(xx[0]),fine_grid)
        n_sim2, tau_sim2 = getModelSamples(xx,x[None,:],ngrid,log_width,taugrid,log_width2,rho,numsamp=numsamp,poly_only=not rand_add)
        
        n_mean2, n_err2 = np.mean(n_sim2,axis=0), np.std(n_sim2,axis=0)
        plot1D(x+med_arr[0],n_mean2,xx[0].ravel()+med_arr[0],ngrid_med,img_dir,'Real_Model_%s_%s_%s_rd_%d'%(indep_name[0],dep_name,extratext,rand_add),xlab=indep_lab[0],ylab='Mean posterior model '+dep_lab,yerr=n_err2)
        if bivar:
            tau_mean2, tau_err2 = np.mean(tau_sim2,axis=0), np.std(tau_sim2,axis=0)
            plot1D(x+med_arr[0],tau_mean2,xx[0].ravel()+med_arr[0],taugrid_med,img_dir,'Real_Model_%s_d2_%s_rd_%d'%(indep_name[0],extratext,rand_add),xlab=indep_lab[0],ylab='Mean posterior model '+taulab,yerr=tau_err2)
        return

    makeColorMapProj(trace, xx, med_arr, indep_lab, indep_name, indep_pickle_name, dataname, img_dir, bivar=bivar, poly_only=not rand_add, numsamp=numsamp, sliced=False, fine_grid=fine_grid, numslices=5, numsamples=nwid, numgal=None, coef_direct=False, dg=degree2,indep=indep_samp,dep_name=dep_name)
    return

def create_prior(proplist,samplearr,zarr):
    proparr = ['ssfr', 'mwa', 'stmass', 'dust1', 'dust2', 'met']
    zlist = np.linspace(0.49999,3.00001,6)
    # propneed = list(set(proparr)&set(proplist))
    propneed = []
    for prop in proparr: 
        if prop in proplist: propneed.append(prop)
    filelist, dimlist, indlist = [], [], []
    n = len(propneed)
    if n==1:
        filelist.append('prior_hist_%s'%(propneed[0]))
        dimlist.append(1)
        indlist.append(proplist.index(propneed[0]))
    elif n==2:
        filelist.append('prior_hist_%s_%s'%(propneed[0],propneed[1]))
        dimlist.append(2)
        indlist.append((proplist.index(propneed[0]),proplist.index(propneed[1])))
    elif n==3:
        if 'stmass' in propneed:
            proprem = np.setdiff1d(propneed,['stmass'],assume_unique=True)
            filelist.append('prior_hist_%s_%s'%(proprem[0],proprem[1]))
            filelist.append('prior_hist_stmass')
            indlist.append((proplist.index(proprem[0]),proplist.index(proprem[1])))
            indlist.append(proplist.index('stmass'))
        else:
            filelist.append('prior_hist_%s_%s'%(propneed[0],propneed[1]))
            filelist.append('prior_hist_%s'%(propneed[2]))
            indlist.append((proplist.index(propneed[0]),proplist.index(propneed[1])))
            indlist.append((proplist.index(propneed[2])))
        dimlist.append(2); dimlist.append(1)
    elif n==4:
        if 'stmass' in propneed and 'met' in propneed:
            proprem = np.setdiff1d(propneed,['stmass','met'],assume_unique=True)
            filelist.append('prior_hist_%s_%s'%(proprem[0],proprem[1]))
            filelist.append('prior_hist_stmass_met')
            indlist.append((proplist.index(proprem[0]),proplist.index(proprem[1])))
            indlist.append((proplist.index('stmass'),proplist.index('met')))
        else:
            filelist.append('prior_hist_%s_%s'%(propneed[0],propneed[1]))
            filelist.append('prior_hist_%s_%s'%(propneed[2],propneed[3]))
            indlist.append((proplist.index(propneed[0]),proplist.index(propneed[1])))
            indlist.append((proplist.index(propneed[2]),proplist.index(propneed[3])))
        dimlist.append(2); dimlist.append(2)
    else:
        for index in range(0,len(propneed),2):
            if index+1 < len(propneed): 
                filelist.append('prior_hist_%s_%s'%(propneed[index],propneed[index+1]))
                dimlist.append(2)
                indlist.append((proplist.index(propneed[index]),proplist.index(propneed[index+1])))
            else: 
                filelist.append('prior_hist_%s'%(propneed[index]))
                dimlist.append(1)
                indlist.append(proplist.index(propneed[index]))
    logprior = np.zeros_like(samplearr)
    sh = list(samplearr.shape)[1:]
    sh.insert(0,len(zlist))
    priordiffz = np.zeros(tuple(sh))
    for fi,dim,ind in zip(filelist,dimlist,indlist):
        if dim==1:
            for i,z in enumerate(zlist):
                x, pdf = np.loadtxt(op.join('Prior','Hists',fi+'_%.2f.dat'%(z)),unpack=True)
                priordiffz[i] += np.log(np.interp(samplearr[ind],x,pdf,left=min_pdf, right=min_pdf))
        else:
            for i,z in enumerate(zlist):
                every = np.loadtxt(op.join('Prior','Hists',fi+'_%.2f.dat'%(z)))
                xx, yy = np.meshgrid(every[:,0],every[:,1])
                pdf = LinearNDInterpolator(np.column_stack((xx.ravel(),yy.ravel())),every[:,2:].ravel(),fill_value=min_pdf)
                priordiffz[i] += np.log(pdf(samplearr[ind[0]],samplearr[ind[1]]))
    indarr = np.searchsorted(zlist,zarr)
    priordiff = np.zeros_like(samplearr[0])
    for i,ind in enumerate(indarr):
        priordiff[i] = (zlist[ind]-zarr[i])*priordiffz[ind-1,i,:] + (zarr[i]-zlist[ind-1])*priordiffz[ind,i,:]
    return priordiff

def KDEApproach(data):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-2, 1, 40)}
    grid = GridSearchCV(KernelDensity(algorithm='auto', kernel='epanechnikov', metric='euclidean', atol=1.0e-10, rtol=1.0e-10, breadth_first=True, leaf_size=40), params)
    grid.fit(data)
    kde = grid.best_estimator_
    print("best bandwidth: {0}".format(kde.bandwidth))

    # Plot the thing!
    med = np.median(data,axis=0)
    plot_len = 201
    for i in range(len(data[0])):
        x_grid = np.repeat(med[None,:],plot_len,axis=0)
        x_grid[:,i] = np.linspace(min(data[:,i]),max(data[:,i]),plot_len)
        fig, ax = plt.subplots()
        pdf = np.exp(kde.score_samples(x_grid))
        ax.plot(x_grid[:,i], pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
        ax.hist(data[:,i], 30, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
        # ax.legend(loc='upper left')
        plt.show()

    # use the best estimator to compute the kernel density estimate
    return kde

def scoring(arr,kde):
    return kde.score_samples(arr)

def getKDE(proplist,samplearr,zarr):
    print("Starting KDE Module")
    zlist = np.linspace(0.49999,3.00001,6)
    sh = list(samplearr.shape)[1:]
    sh.insert(0,len(zlist))
    priordiffz = np.zeros(tuple(sh))
    for j, zred in enumerate(zlist):
        print("Starting redshift %.2f KDE prior run"%(zred))
        fn = 'Prior/samples_z{0:.2f}.pickle'.format(zred)
        with (open(fn,'rb')) as openfile:
            obj = pickle.load(openfile)
        ssfr = np.array(obj['ssfr'])
        cond = ssfr>=ssfr_lim_prior
        data = np.empty((len(ssfr[cond]),len(proplist)))
        for i,prop in enumerate(proplist):
            if prop=='sfr': data[:,i] = np.array(obj['ssfr'])[cond] + np.array(obj['stmass'])[cond]
            else: data[:,i] = np.array(obj[prop])[cond]
        print("About to run KDE code on data, with shape",data.shape)
        # kde = KDEApproach(data)
        kde = KernelDensity(bandwidth=0.2*len(proplist)-0.1,algorithm='auto', kernel='gaussian', metric='euclidean', atol=1.0e-5, rtol=1.0e-5, breadth_first=True, leaf_size=40)
        kde.fit(data)
        print("Finished running KDE code")

        arr_score = np.transpose(samplearr,axes=(1,2,0)).reshape(samplearr.shape[1]*samplearr.shape[2],samplearr.shape[0])
        arr_split = np.array_split(arr_score,cores,axis=0)

        with Pool(processes=cores) as pool:
            results = pool.starmap(scoring,product(arr_split,[kde]))

        priordiffz[j] = np.concatenate(results).reshape(samplearr[0].shape)
        # print("Finished getting priordiffz using map")
        # priordiffz_orig = kde.score_samples(arr_score).reshape(samplearr[0].shape)
        print("Finished this round's probability calculation")
    indarr = np.searchsorted(zlist,zarr)
    priordiff = np.zeros_like(samplearr[0])
    for i,ind in enumerate(indarr):
        dz = zlist[ind]-zlist[ind-1]
        priordiff[i] = (zlist[ind]-zarr[i])/dz*priordiffz[ind-1,i,:] + (zarr[i]-zlist[ind-1])/dz*priordiffz[ind,i,:]
    print("Finished KDE module")
    return priordiff

def getKDEsimple(proplist,samplearr,proplab=None,extratext='allz_gauss'):
    print("Starting KDE Module")
    fn = 'Prior/samples_allz.pickle'
    with (open(fn,'rb')) as openfile:
        obj = pickle.load(openfile)
    ssfr = np.array(obj['ssfr'])
    cond = ssfr>=ssfr_lim_prior
    data = np.empty((len(ssfr[cond]),len(proplist)))
    for i,prop in enumerate(proplist):
        data[:,i] = np.array(obj[prop])[cond]
    print("About to run KDE code on data, with shape",data.shape)
    # kde = KDEApproach(data)
    kde = KernelDensity(bandwidth=0.2*len(proplist)-0.1,algorithm='auto', kernel='gaussian', metric='euclidean', atol=1.0e-5, rtol=1.0e-5, breadth_first=True, leaf_size=40)
    kde.fit(data)
    print("Finished running KDE code")
    # if len(samplearr)==1: plotPrior1D(data[:,0],kde,proplist[0],proplab[0],extratext=extratext)
    # elif len(samplearr)==2: plotPrior2D(data,kde,proplist,proplab,extratext=extratext)
    # else: pass

    arr_score = np.transpose(samplearr,axes=(1,2,0)).reshape(samplearr.shape[1]*samplearr.shape[2],samplearr.shape[0])
    arr_split = np.array_split(arr_score,cores,axis=0)

    with Pool(processes=cores) as pool:
        results = pool.starmap(scoring,product(arr_split,[kde]))

    priordiff = np.concatenate(results).reshape(samplearr[0].shape)
    print("Finished KDE module")
    return priordiff

def getKDEPrior(proplist,proplab):
    zlist = np.linspace(0.5,3.0,6)
    for j, zred in enumerate(zlist):
        fn = 'Prior/samples_z{0:.2f}.pickle'.format(zred)
        with (open(fn,'rb')) as openfile:
            obj = pickle.load(openfile)
        ssfr = np.array(obj['ssfr'])
        cond = ssfr>=ssfr_lim_prior
        data = np.empty((len(ssfr[cond]),len(proplist)))
        for i,prop in enumerate(proplist):
            data[:,i] = np.array(obj[prop])[cond]
        print("About to run KDE code on data, with shape",data.shape)
        # kde = KDEApproach(data)
        kde = KernelDensity(bandwidth=0.2*len(proplist)-0.1,algorithm='auto', kernel='epanechnikov', metric='euclidean', atol=1.0e-5, rtol=1.0e-5, breadth_first=True, leaf_size=40)
        kde.fit(data)
        print("Finished running KDE code")
        if len(proplist)==1: plotPrior1D(data[:,0],kde,proplist[0],proplab[0],extratext='z%.2f'%(zred))
        elif len(proplist)==2: plotPrior2D(data,kde,proplist,proplab,extratext='z%.2f'%(zred),gridding=100)
        else: pass

def plotPrior1D(samplearr,kde,propname,proplab,extratext='',gridding=1001):
    mkpath('Prior/KDE/%s'%(propname))
    fig, ax = plt.subplots()
    ax.hist(samplearr.ravel(),bins=min(50,samplearr.size//100),color='b',density=True,log=True,label='Normalized Histogram',alpha=0.75)
    # inds = np.argsort(samplearr.ravel())
    per = np.percentile(samplearr,[1.0,99.0])
    xgrid = np.linspace(per[0],per[1],gridding)
    # ax.plot(samplearr.ravel()[inds],np.exp(priordiff.ravel()[inds]),'r-',lw=2,label='KDE-based PDF')
    print("About to run score samples in 1-D case")
    prob = np.exp(kde.score_samples(xgrid[:,None]))
    print("Finished scoring in 1-D case")
    integ = trapz(prob,xgrid)
    print("Total integral over the considered parameter space in this graph:",integ)
    ax.plot(xgrid,prob,'r-',lw=2,label='KDE-based PDF')
    ax.set_xlabel(proplab)
    ax.set_ylabel("Prior Probability")
    ax.set_yscale('log') # Just in case it wasn't already
    ax.set_xlim(per[0],per[1])
    # ax.legend(loc='best',frameon=False,fontsize='small')
    fig.savefig("Prior/KDE/%s/%s_PriorPDF%s.png"%(propname,propname,extratext),bbox_inches='tight',dpi=200)

def plotPrior2D(samplearr,kde,propnames,proplabs,gridding=201,extratext=''):
    mkpath('Prior/KDE/%s_%s'%(propnames[0],propnames[1]))
    per_matrix = np.percentile(samplearr,[1.0,99.0],axis=0)
    gridx = np.linspace(per_matrix[0][0],per_matrix[1][0],gridding)
    gridy = np.linspace(per_matrix[0][1],per_matrix[1][1],gridding)
    xx, yy = np.meshgrid(gridx,gridy)
    data = np.column_stack((xx.ravel(),yy.ravel()))
    lnprob = kde.score_samples(data).reshape(xx.shape)
    prob = np.exp(lnprob)
    integ = trapz(trapz(prob,gridx),gridy)
    print("Total integral over the considered parameter space in this graph:",integ)
    fig, ax = plt.subplots()
    cf = ax.contourf(gridx,gridy,lnprob,levels=15,cmap='viridis')
    ax.set_xlabel(proplabs[0]); ax.set_ylabel(proplabs[1])
    plt.colorbar(cf,label='ln probability')
    fig.savefig("Prior/KDE/%s_%s/%s_%s_PriorPDF%s.png"%(propnames[0],propnames[1],propnames[0],propnames[1],extratext),bbox_inches='tight',dpi=300)

def make_prop_dict():
    prop_dict, dep_dict = {}, {}
    prop_dict['names'] = {'logM': 'logM', 'ssfr': 'logsSFR', 'sfr': 'logSFR', 'logZ': 'logZ', 'z': 'z', 'i':'axis_ratio', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['names'] = {'tau2': 'dust2', 'n': 'n'}
    prop_dict['labels'] = {'logM': r'$\log M_*$', 'ssfr': r'$\log$ sSFR$_{\rm{100}}$', 'sfr': r'$\log$ SFR$_{\rm{100}}$', 'logZ': r'$\log (Z/Z_\odot)$', 'z': 'z', 'i':r'$b/a$', 'd1':r"$\hat{\tau}_{1}$", 'd2':r"$\hat{\tau}_{2}$"}
    dep_dict['labels'] = {'tau2': r"$\hat{\tau}_{2}$", 'n': 'n'}
    prop_dict['pickle_names'] = {'logM': 'stellar_mass', 'ssfr': 'ssfr_100', 'sfr': 'sfr_100', 'logZ': 'log_z_zsun', 'z': 'z', 'i':'inc', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['pickle_names'] = {'tau2': 'dust2', 'n': 'dust_index'}
    prop_dict['sigma'] = {'logM': 0.0732, 'ssfr': 0.214, 'sfr': 0.2, 'logZ': 0.189, 'z': 0.005, 'i': 0.06, 'd1': 0.136, 'd2': 0.116}
    dep_dict['sigma'] = {'tau2': 0.116, 'n': 0.191}
    prop_dict['med'] = {'logM': 9.91, 'ssfr': -9.30, 'sfr': 0.61, 'logZ': -0.26, 'z': 1.10, 'i': 0.485, 'd1': 0.221, 'd2': 0.239}
    prop_dict['min'] = {'logM': -1.193, 'ssfr': -2.697, 'sfr': -3.0, 'logZ': -1.710, 'z': -0.605, 'i': 0, 'd1': 0, 'd2': 0}
    prop_dict['max'] = {'logM': 1.907, 'ssfr': 1.393, 'sfr': 2.5, 'logZ': 0.446, 'z': 1.894, 'i': 1.0, 'd1': 4.8, 'd2': 4.0}
    return prop_dict, dep_dict

def data_simulation(args,prop_dict):
    if args.data:
        with (open("3dhst_samples_10_inc.pickle",'rb')) as openfile:
            obj = pickle.load(openfile)
        logM, ssfr = np.log10(obj['stellar_mass']), np.log10(obj['ssfr_100'])
        logZ = obj['log_z_zsun']
        logMavg = np.average(logM,axis=1); logssfravg = np.average(ssfr,axis=1)
        logZavg = np.average(logZ,axis=1)
        z = obj['z']
        n = obj['dust_index']
        tau2 = obj['dust2']
        navg = np.average(n,axis=1)
        tau2avg = np.average(tau2,axis=1)
        masscomp = mass_completeness(z)
        cond = np.logical_and.reduce((logMavg>=masscomp,z<3.0,logssfravg>ssfr_lims_data[0],logssfravg<ssfr_lims_data[1]))
        if args.size==-1: args.size = len(z[cond])
        ind = np.where(cond)[0]
        indfin = np.random.choice(ind,size=args.size,replace=False)
        indep_true, indep_true2 = np.empty((0,args.size)), np.empty((0,args.size))
        med = np.array([])
        if args.logM: 
            indep_true = np.append(indep_true,logMavg[indfin][None,:]-np.median(logMavg[indfin]),axis=0)
            med = np.append(med,np.median(logMavg[indfin]))
        if args.ssfr: 
            indep_true = np.append(indep_true,logssfravg[indfin][None,:]-np.median(logssfravg[indfin]),axis=0)
            med = np.append(med,np.median(logssfravg[indfin]))
        if args.logZ: 
            indep_true = np.append(indep_true,logZavg[indfin][None,:]-np.median(logZavg[indfin]),axis=0)
            med = np.append(med,np.median(logZavg[indfin]))
        if args.z: 
            indep_true = np.append(indep_true,z[indfin][None,:]-np.median(z[indfin]),axis=0)
            med = np.append(med,np.median(z[indfin]))

        med_mod = np.array([])
        if args.logM_mod: 
            indep_true2 = np.append(indep_true2,logMavg[indfin][None,:]-np.median(logMavg[indfin]),axis=0)
            med_mod = np.append(med_mod,np.median(logMavg[indfin]))
        if args.ssfr_mod: 
            indep_true2 = np.append(indep_true2,logssfravg[indfin][None,:]-np.median(logssfravg[indfin]),axis=0)
            med_mod = np.append(med_mod,np.median(logssfravg[indfin]))
        if args.logZ_mod: 
            indep_true2 = np.append(indep_true2,logZavg[indfin][None,:]-np.median(logZavg[indfin]),axis=0)
            med_mod = np.append(med_mod,np.median(logZavg[indfin]))
        if args.z_mod: 
            indep_true2 = np.append(indep_true2,z[indfin][None,:]-np.median(z[indfin]),axis=0)
            med_mod = np.append(med_mod,np.median(z[indfin]))

        if args.n: n = navg[indfin]
        else: n = tau2avg[indfin]
        limlist = None; limlist2 = None
    else:
        limlist, limlist2 = [], []
        med, med_mod = np.array([]), np.array([])
        for name in prop_dict['names'].keys():
            if getattr(args,name):
                limlist.append(np.array([prop_dict['min'][name],prop_dict['max'][name]]))
                med = np.append(med,prop_dict['med'][name])
            if getattr(args,name+'_mod'):
                limlist2.append(np.array([prop_dict['min'][name],prop_dict['max'][name]]))
                med_mod = np.append(med_mod,prop_dict['med'][name])
        indep_true, indep_true2, n = None, None, None
    return indep_true, med, n, limlist, indep_true2, med_mod, limlist2

def data_true(args):
    with (open("3dhst_samples_10_inc.pickle",'rb')) as openfile:
        obj = pickle.load(openfile)
    logM, ssfr = np.log10(obj['stellar_mass']), np.log10(obj['ssfr_100'])
    samples = len(logM[0])
    logZ = obj['log_z_zsun']
    logMavg = np.average(logM,axis=1); logssfravg = np.average(ssfr,axis=1); logZavg = np.average(logZ,axis=1)
    z = obj['z']
    n = obj['dust_index']
    tau1 = obj['dust1']
    tau2 = obj['dust2']
    inc = obj['inc']
    sfr = np.log10(obj['sfr_100'])
    # comb = np.array([ssfr,logZ])
    per = np.percentile(ssfr,[1.0,99.0])
    masscomp = mass_completeness(z)
    # cond = np.logical_and.reduce((logMavg>=masscomp,z<3.0,np.amin(ssfr,axis=1)>ssfr_lims_data[0],np.amax(ssfr,axis=1)<ssfr_lims_data[1],logssfravg>per[0][0],logssfravg<per[1][0],logZavg>per[1][0],logZavg<per[1][1]))
    cond = np.logical_and.reduce((logMavg>=masscomp,logssfravg>per[0]))
    
    if args.i_mod: cond = np.logical_and.reduce((cond,np.median(inc,axis=1)>=0.0,np.median(inc,axis=1)<=1.0))

    lenzcond = len(z[cond])
    print("Length of z cond: ", lenzcond)
    if args.size == -1 or args.size>lenzcond: args.size = lenzcond

    ind = np.where(cond)[0]
    indfin = np.random.choice(ind,size=args.size,replace=False)
    indep_samp, prior_samp = np.empty((0,len(z[indfin]),len(n[0]))), np.empty((0,len(z[indfin]),len(n[0])))
    med_mod, prior_prop, prior_lab = np.array([]), [], []
    if args.logM_mod: 
        md = np.median(logM[indfin])
        indep_samp = np.append(indep_samp,logM[indfin][None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
        prior_samp = np.append(prior_samp,logM[indfin][None,:,:],axis=0)
        prior_prop.append('stmass')
        prior_lab.append(r'$\log M_*$')
    if args.ssfr_mod: 
        md = np.median(ssfr[indfin])
        indep_samp = np.append(indep_samp,ssfr[indfin][None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
        prior_samp = np.append(prior_samp,ssfr[indfin][None,:,:],axis=0)
        prior_prop.append('ssfr')
        prior_lab.append(r'$\log$ sSFR$_{\rm{100}}$')
    if args.sfr_mod: 
        md = np.median(sfr[indfin])
        indep_samp = np.append(indep_samp,sfr[indfin][None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
        prior_samp = np.append(prior_samp,sfr[indfin][None,:,:],axis=0)
        prior_prop.append('sfr')
        prior_lab.append(r'$\log$ SFR$_{\rm{100}}$')
    if args.logZ_mod: 
        md = np.median(logZ[indfin])
        indep_samp = np.append(indep_samp,logZ[indfin][None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
        prior_samp = np.append(prior_samp,logZ[indfin][None,:,:],axis=0)
        prior_prop.append('met')
        prior_lab.append(r'$\log (Z/Z_\odot)$')
    if args.z_mod: 
        md = np.median(z[indfin])
        zrep = np.repeat(z[indfin][:,None],samples,axis=1)
        indep_samp = np.append(indep_samp,zrep[None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
    if args.i_mod:
        md = np.median(inc[indfin])
        indep_samp = np.append(indep_samp,inc[indfin][None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
    if args.d1_mod:
        md = np.median(tau1[indfin])
        indep_samp = np.append(indep_samp,tau1[indfin][None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
        prior_samp = np.append(prior_samp,tau1[indfin][None,:,:],axis=0)
        prior_prop.append('dust1')
        prior_lab.append(r"$\hat{\tau}_{1}$")
    if args.d2_mod:
        md = np.median(tau2[indfin])
        indep_samp = np.append(indep_samp,tau2[indfin][None,:,:]-md,axis=0)
        med_mod = np.append(med_mod,md)
        prior_samp = np.append(prior_samp,tau2[indfin][None,:,:],axis=0)
        prior_prop.append('dust2')
        prior_lab.append(r"$\hat{\tau}_{2}$")

    if args.bivar: 
        dep_samp = [n[indfin], tau2[indfin]]
        prior_samp = np.append(prior_samp,tau2[indfin][None,:,:],axis=0)
        prior_prop.append('dust2')
        prior_lab.append(r"$\hat{\tau}_{2}$")
        uniform = False
    else:
        if args.n and not args.bivar: 
            dep_samp = n[indfin]
            uniform = True
        else: 
            dep_samp = tau2[indfin]
            prior_samp = np.append(prior_samp,dep_samp[None,:,:],axis=0)
            prior_prop.append('dust2')
            prior_lab.append(r"$\hat{\tau}_{2}$")
            uniform = False

    return indep_samp, dep_samp, prior_samp, med_mod, z[indfin], prior_prop, prior_lab, uniform

def label_creation(args,prop_dict,dep_dict):
    indep_pickle_name, indep_name, indep_lab, sigarr = [], [], [], np.array([])
    indep_pickle_name2, indep_name2, indep_lab2, sigarr2 = [], [], [], np.array([])
    for name in prop_dict['names'].keys():
        if getattr(args,name):
            indep_pickle_name.append(prop_dict['pickle_names'][name])
            indep_name.append(prop_dict['names'][name])
            indep_lab.append(prop_dict['labels'][name])
            sigarr = np.append(sigarr,prop_dict['sigma'][name])
        if getattr(args,name+'_mod'):
            indep_pickle_name2.append(prop_dict['pickle_names'][name])
            indep_name2.append(prop_dict['names'][name])
            indep_lab2.append(prop_dict['labels'][name])
            sigarr2 = np.append(sigarr2,prop_dict['sigma'][name])
    if args.n: 
        sigN, dep_name, dep_lab, nlim = dep_dict['sigma']['n'], dep_dict['names']['n'], dep_dict['labels']['n'], np.array([-1.0,0.4])
    else:
        sigN, dep_name, dep_lab, nlim = dep_dict['sigma']['tau2'], dep_dict['names']['tau2'], dep_dict['labels']['tau2'], np.array([0.0,4.0])
    return indep_name, indep_name2, indep_lab, indep_lab2, indep_pickle_name, indep_pickle_name2, sigarr, sigarr2, sigN, dep_name, dep_lab, nlim

def main(args=None):
    RANDOM_SEED = 8929
    np.random.seed(RANDOM_SEED)

    prop_dict, dep_dict = make_prop_dict()
    if args==None: args = parse_args()
    print(args)

    indep_name, indep_name2, indep_lab, indep_lab2, indep_pickle_name, indep_pickle_name2, sigarr, sigarr2, sigN, dep_name, dep_lab, nlim = label_creation(args,prop_dict,dep_dict)
    args.indep_name = indep_name2

    if args.real: 
        img_dir_orig = op.join('DataTrue',args.dir_orig)
        indep_samp, dep_samp, prior_samp, med_mod, zarr, prior_prop, prior_lab, uniform = data_true(args)
        print("Finished getting data")
        if args.already_done:
            img_dir_orig, args.dataname = op.join('DataFinal',args.dir_orig), args.extratext
            trace, xx, _ = dac.getPostModelData(args,img_dir_orig=img_dir_orig)
            app_to_file, a_poly_T = False, None
        else:
            app_to_file=True
            if len(prior_prop)>=1: 
                # logp_prior_orig = create_prior(prior_prop,prior_samp,zarr)
                # print("Finished creating original 1-D/2-D prior combination")
                # getKDEPrior(prior_prop,prior_lab)
                logp_prior = getKDE(prior_prop,prior_samp,zarr)
                # logp_prior_simp = getKDEsimple(prior_prop,prior_samp,proplab=prior_lab)
                print("Finished all KDE Calculations")
                print("Nans/infs in logp_prior:",np.isnan(logp_prior).any() or np.isinf(logp_prior).any())
                # logp_prior = 0.0
            else: logp_prior = 0.0
            # print(logp_prior_orig-np.amin(logp_prior_orig),logp_prior-np.amin(logp_prior))
        if args.bivar:
            if not args.already_done: trace, a_poly_T, xx = polyNDDataBivar(indep_samp,dep_samp[0],dep_samp[1],logp_prior,img_dir_orig,plot=args.plot,extratext=args.extratext,degree=args.degree2,sampling=args.steps,tune=args.tune)

            get_relevant_info_ND_Data(trace,a_poly_T,xx,indep_samp,dep_samp[0],med_mod,indep_name2,indep_lab2,degree2=args.degree2,extratext=args.extratext,img_dir_orig=img_dir_orig,fl_stats=fl_fit_stats_bivar,bivar=True,rand_add=args.random,tau_samp=dep_samp[1],indep_pickle_name=indep_pickle_name2,append_to_file=app_to_file)

        else:
            if not args.already_done: trace, a_poly_T, xx = polyNDData(indep_samp,dep_samp,logp_prior,img_dir_orig,dep_lim=nlim,tune=args.tune,plot=args.plot,extratext=args.extratext,degree=args.degree2,sampling=args.steps,uniform=uniform)

            get_relevant_info_ND_Data(trace,a_poly_T,xx,indep_samp,dep_samp,med_mod,indep_name2,indep_lab2,dep_name=dep_name,dep_lab=dep_lab,degree2=args.degree2,extratext=args.extratext,img_dir_orig=img_dir_orig,fl_stats=fl_fit_stats_univar,append_to_file=app_to_file,rand_add=args.random,bivar=False,indep_pickle_name=indep_pickle_name2)

    else: 
        img_dir_orig = op.join('DataSim',args.dir_orig)
        indep_true, med, n, limlist, indep_true2, med_mod, limlist2 = data_simulation(args,prop_dict)

        if args.bivar:
            trace, a_poly_T, xx, ngrid_true, taugrid_true, coefs_true, coefs_true2, n_true, tau_true, width_true, width_true2, rho_true, indep_true, indep_samp, n_samp, tau_samp = polyND_bivar(img_dir_orig, limlist=limlist, size=args.size,samples=args.samples,tune=args.tune,errmult=args.error_mult,errmultn=args.error_mult_n,errmultt=args.error_mult_t,errmultcross=args.error_mult_cross,plot=args.plot,extratext=args.extratext,degree=args.degree,sampling=args.steps,sigarr=sigarr,sigN=dep_dict['sigma']['n'],sigT=dep_dict['sigma']['tau2'],indep_true=indep_true,n=None,tau=None)
    
            get_relevant_info_ND_Gen(trace, a_poly_T, xx, xx, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_true, indep_samp, indep_samp, n_samp, med, med, indep_name, indep_name, indep_lab, dep_name=dep_dict['names']['n'], dep_lab=dep_dict['labels']['n'], degree=args.degree, degree2=args.degree, extratext=args.extratext, numsamp=500, img_dir_orig=img_dir_orig, bivar=args.bivar)

            get_relevant_info_ND_Gen(trace, a_poly_T, xx, xx, taugrid_true, coefs_true2, tau_true, width_true2, indep_true, indep_true, indep_samp, indep_samp, tau_samp, med, med, indep_name, indep_name, indep_lab, dep_name=dep_dict['names']['tau2'], dep_lab=dep_dict['labels']['tau2'], degree=args.degree, degree2=args.degree, extratext=args.extratext,numsamp=500,img_dir_orig=img_dir_orig, grid_name='taugrid', width_name='log_width2', bivar=args.bivar)

        else:
            trace, a_poly_T, a_poly_T2, xx, xx2, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_true2, indep_samp, indep_samp2, n_samp = polyNDgen(img_dir_orig, limlist=limlist, nlim=nlim, size=args.size,samples=args.samples,tune=args.tune,errmult=args.error_mult,errmultn=args.error_mult_n,plot=args.plot,extratext=args.extratext,degree=args.degree,sampling=args.steps,sigarr=sigarr,sigN=sigN,indep_true=indep_true,n=n,degree2=args.degree2,limlist2=limlist2,indep_true2=indep_true2,sigarr2=sigarr2,indep_name=indep_name,indep_name2=indep_name2,numtrials=20000)

            get_relevant_info_ND_Gen(trace, a_poly_T2, xx, xx2, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_true2, indep_samp, indep_samp2, n_samp, med, med_mod, indep_name, indep_name2, indep_lab2, dep_name=dep_name, dep_lab=dep_lab, degree=args.degree,degree2=args.degree2,extratext=args.extratext,numsamp=500,img_dir_orig=img_dir_orig)

if __name__=='__main__':
    main()