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
from anchor_points import get_a_polynd, calc_poly, calc_poly_tt, polyfitnd
import argparse as ap
from scipy.stats import norm, truncnorm
from copy import copy, deepcopy
import corner

def mass_completeness(zred):
    """used mass-completeness estimates from Tal+14, for FAST masses
    then applied M_PROSP / M_FAST to estimate Prospector completeness
    """

    zref = np.array([0.65,1,1.5,2.1,3.0])
    mcomp_prosp = np.array([8.71614882,9.07108637,9.63281923,9.79486727,10.15444536])
    mcomp = np.interp(zred, zref, mcomp_prosp)

    return mcomp

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
    parser.add_argument('-dir','--dir_orig',help='Parent directory for files',type=str,default='NewTests')

    parser.add_argument('-m','--logM',help='Whether or not to include stellar mass (true)',action='count',default=0)
    parser.add_argument('-s','--ssfr',help='Whether or not to include sSFR (true)',action='count',default=0)
    parser.add_argument('-logZ','--logZ',help='Whether or not to include metallicity (true)',action='count',default=0)
    parser.add_argument('-z','--z',help='Whether or not to include redshift (true)',action='count',default=0)
    parser.add_argument('-mm','--logM_mod',help='Whether or not to include stellar mass in model',action='count',default=0)
    parser.add_argument('-sm','--ssfr_mod',help='Whether or not to include sSFR in model',action='count',default=0)
    parser.add_argument('-logZm','--logZ_mod',help='Whether or not to include metallicity in model',action='count',default=0)
    parser.add_argument('-zm','--z_mod',help='Whether or not to include redshift in model',action='count',default=0)
    parser.add_argument('-n','--n',help='Whether or not to use dust index as dependent variable',action='count',default=0)

    return parser.parse_args(args=argv)

def polyND(img_dir_orig,limlist=None,size=500,samples=50,plot=False,extratext='',nlim=np.array([-1.0,0.4]),degree=2,sampling=1000,tune=1000,errmult=1.0,errmultn=1.0,indep_true=None,sigarr=None,sigN=0.19,n=None):
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
    if nlim[1]>1.0: uniform = False
    else: uniform = True

    if limlist is None:
        limlist = []
        for indep in indep_true:
            # limlist.append(np.percentile(indep,[2.5,97.5]))
            limlist.append(np.array([min(indep),max(indep)]))
    else:
        indep_true = np.empty((0,size))
        for lim in limlist:
            indep_true = np.append(indep_true,np.random.uniform(lim[0],lim[-1],(1,size)),axis=0)
    ndim = len(limlist)
    # Determining true parameters
    width_true = 0.1*errmultn #Intrinsic dispersion in relation
    x = np.empty((0,degree+1)) # To set up grid on which dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    rng = nlim[1]-nlim[0]
    avg = (nlim[1]+nlim[0])/2.0
    if n is None:
        np.random.seed(3890)
        if uniform: ngrid_true = rng*np.random.rand(xx[0].size)-rng/2.0 + avg #True values of dust parameter at the grid
        else: 
            ngrid_true = truncnorm.rvs(-0.3,3.7,loc=0.3,scale=1.0,size=xx[0].size)
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
    else:
        sigarr *= errmult
    sigN *= errmult
    # First we need to "move" the mean of the posterior, then we generate posterior samples
    # indep_samp = indep_true[:,:,None] + np.tensordot(sigarr,np.random.randn(size,1),axes=0) + np.tensordot(sigarr,np.random.randn(size,samples),axes=0)

    indep_samp = indep_true[:,:,None] + sigarr[:,None,None]*np.random.randn(ndim,size,1) + sigarr[:,None,None]*np.random.randn(ndim,size,samples)

    n_samp = n_true[:, None] + sigN * np.random.randn(size, 1) + sigN * np.random.randn(size, samples)
    # breakpoint()
    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,degree)
    
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        if uniform: ngrid = pm.Uniform("ngrid",lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=ngrid_true.size,testval=rng*np.random.rand(ngrid_true.size)-rng/2.0 + avg)
        else: ngrid = pm.TruncatedNormal("ngrid",mu=0.3,sigma=1.0,lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=ngrid_true.size,testval=rng*np.random.rand(ngrid_true.size)-rng/2.0 + avg)
        log_width = pm.StudentT("log_width", nu=5, mu=np.log(width_true), lam=0.5, testval=-5.0)
        # log_width = pm.Uniform("log_width",lower=np.log(width_true)-5.0,upper=np.log(width_true)+5.0,testval=-5.0)

        # Compute the expected n at each sample
        coefs = tt.dot(aTinv,ngrid)
        mu = tt.tensordot(coefs,term,axes=1)

        # The line has some width: we're calling it a Gaussian in n
        logp = -0.5 * (n_samp - mu) ** 2 * pm.math.exp(-2*log_width) - log_width

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
        print(az.summary(trace,round_to=2))
        print("ngrid_true:"); print(ngrid_true)
        print("log_width_true:"); print(np.log(width_true))

    return trace, a_poly_T, xx, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_samp, n_samp

def polyNDgen(img_dir_orig,limlist=None,size=500,samples=50,plot=False,extratext='',nlim=np.array([-1.0,0.4]),degree=2,degree2=1,sampling=1000,tune=1000,errmult=1.0,errmultn=1.0,indep_true=None,sigarr=None,sigN=0.19,n=None,limlist2=None,indep_true2=None,sigarr2=None,indep_name=None,indep_name2=None):
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
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
        print(az.summary(trace,round_to=2))
        print("ngrid_true:"); print(ngrid_true)
        print("log_width_true:"); print(np.log(width_true))
    
    return trace, a_poly_T, a_poly_T2, xx, xx2, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_true2, indep_samp, indep_samp2, n_samp

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
    rng = nlim[1]-nlim[0]
    rng2 = taulim[1]-taulim[0]
    avg = (nlim[1]+nlim[0])/2.0
    avg2 = (taulim[1]+taulim[0])/2.0
    if n is None:
        np.random.seed(3890)
        ngrid_true = 0.9*rng*np.random.rand(xx[0].size)-0.9*rng/2.0 + avg #True values of dust parameter at the grid
        taugrid_true = truncnorm.rvs(-0.3,3.7,loc=0.3,scale=1.0,size=xx[0].size) #True values of 2nd dust parameter at the grid
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
        ngrid = pm.Uniform("ngrid",lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=ngrid_true.size,testval=0.9*rng*np.random.rand(ngrid_true.size)-0.9*rng/2.0 + avg)
        taugrid = pm.TruncatedNormal("taugrid",mu=0.3,sigma=1.0,lower=taulim[0]-1.0e-5,upper=taulim[1]+1.0e-5,shape=taugrid_true.size,testval=0.9*rng2*np.random.rand(taugrid_true.size)-0.9*rng2/2.0 + avg2)
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
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=False)

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

    return trace, a_poly_T, xx, ngrid_true, taugrid_true, coefs_true, coefs_true2, n_true, tau_true, width_true, width_true2, rho_true, indep_true, indep_samp, n_samp, tau_samp

def plot_color_map(x,y,z,xdiv,ydiv,znum,xx,yy,img_dir,name,levels=10,xlab=r'$\log\ M_*$',ylab=r'$\log$ sSFR$_{\rm{100}}$',zlab='Median model n',cmap='viridis',xtrue=None,ytrue=None):
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
    if cmap=='viridis': cf = ax.contourf(x,y,z,levels=levels,cmap=cmap)
    else: cf = ax.contourf(x,y,z,levels=levels,cmap=cmap,vmin=-1.0*np.amax(abs(z)),vmax=np.amax(abs(z)))
    if np.sum(znum)>2501:
        cnum = ax.contour(xdiv,ydiv,znum,levels=4,cmap='Greys')
    else:
        if xtrue is not None: plt.plot(xtrue,ytrue,'k.',markersize=1,alpha=0.3)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(np.amin(xx),np.amax(xx))
    ax.set_ylim(np.amin(yy),np.amax(yy))
    fig.colorbar(cf,label=zlab)
    if np.sum(znum)>2501:
        fig.colorbar(cnum,label='Number of Galaxies')
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
    plt.errorbar(n_true,n_med,yerr=n_err,xerr=width_true,fmt='b.',markersize=2,label='Input values',alpha=0.1)
    if ngrid_true_plot: plt.errorbar(ngrid_true,ngrid_med,yerr=all_std,fmt='r^',markersize=6,label='Grid')
    xmin, xmax = plt.gca().get_xlim()
    xplot = np.linspace(xmin,xmax,101)
    plt.plot(xplot,xplot,'k--',lw=2,label='1-1')
    plt.gca().set_xlim(xmin,xmax)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
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

def get_relevant_info_ND(trace,a_poly_T,xx,ngrid_true,coefs_true,n_true,width_true,indep_true,indep_samp,n_samp,med_arr,indep_name,indep_lab,dep_name='n',dep_lab='n',degree=2,numsamp=50,levels=10,extratext='',fine_grid=201,bins=20,img_dir_orig=op.join('DataSim','2DTests'),grid_name='ngrid',width_name='log_width'):
    """ Perform various checks for a simulated run to determine how well the test has performed

    Parameters:
    -----------
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
    degree: Int
        Polynomial degree
    numsamp: Int
        Number of samples to consider from trace for determining model error
    levels: Int
        Number of filled contours for color map
    extratext: Str
        String to help distinguish the particular run
    fine_grid: Int
        Number of entries in each dimension of the fine grid for color maps
    bins: Int
        Number of entries in each dimension of a coarser grid for counting galaxies
    img_dir_orig: Str
        Parent directory in which directory for current run was created
    """
    nlen = len(n_true)
    ndim = len(indep_true)
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)

    # Determine median model values for the dependent variable
    ngrid_med = np.median(getattr(trace,grid_name),axis=0)
    width_med = np.exp(np.median(getattr(trace,width_name)))
    all_std = np.std(getattr(trace,grid_name),axis=0)
    coefs_med = np.linalg.lstsq(a_poly_T,ngrid_med,None)[0]
    n_med = calc_poly(indep_true,coefs_med,degree) + width_med * np.random.randn(nlen)
    assert np.all(abs(calc_poly(xx,coefs_true,degree).ravel()-ngrid_true)<1.0e-6) # Make sure ngrid_true was properly computed

    # Print likelihoods of median model, best model, and true values
    log_width_true, log_width_med = np.log(width_true), np.log(width_med)
    mu = calc_poly(indep_samp,coefs_true,degree)
    mu_med = calc_poly(indep_samp,coefs_med,degree)
    logp = -0.5 * (n_samp - mu) ** 2 * np.exp(-2*log_width_true) - log_width_true
    logp_med = -0.5 * (n_samp - mu_med) ** 2 * np.exp(-2*log_width_med) - log_width_med
    max_logp, max_logp_med = np.max(logp, axis=1), np.max(logp_med, axis=1)
    marg_logp = max_logp + np.log(np.sum(np.exp(logp - max_logp[:, None]), axis=1))
    marg_logp_med = max_logp_med + np.log(np.sum(np.exp(logp_med - max_logp_med[:, None]), axis=1))

    print("Likelihood of true solution: %.4f"%(np.sum(marg_logp)))
    print("Likelihood of median NUTS solution: %.4f"%(np.sum(marg_logp_med)))
    print("Likelihood of best NUTS solution: %.4f"%(np.max(trace.model_logp)))

    # Evaluate the error/residuals at the input (data) points
    inds = np.random.choice(len(trace.model_logp),size=numsamp,replace=True)
    n_sim = np.zeros((numsamp,nlen))
    for i, ind in enumerate(inds):
        ngrid_mod = getattr(trace,grid_name)[ind]
        width_mod = np.exp(getattr(trace,width_name)[ind])
        coefs_mod = polyfitnd(xx,ngrid_mod)
        n_sim[i] = calc_poly(indep_true,coefs_mod,degree)+ width_mod * np.random.randn(*n_sim[i].shape)
    # n_err = np.std(n_sim,axis=0)
    # n_quad_err = np.sqrt(n_err**2+width_true**2)
    n_quad_err = np.sqrt(width_med**2+width_true**2)
    resid_norm = (n_med-n_true)/n_quad_err

    # Next plot compares model to the truth very simply
    plot_model_true(n_true,n_med,ngrid_true,ngrid_med,img_dir,'PolyND_%s%s'%(dep_name,extratext),n_err=width_med,width_true=width_true,all_std=all_std,ylab='Median model %s'%(dep_lab),xlab='True %s'%(dep_lab))

    # Histogram of residuals/error and fit a normal distribution to them--ideal result is standard normal
    plot_hist_resid(resid_norm,img_dir,'hist_resid_sample_'+dep_name+extratext)

    # Plot general percentile comparison
    plot_percentile(n_sim,n_true,img_dir,'percentile_'+dep_name+extratext)

    if ndim<2: 
        return
    
    # 2-D (or 2+D-dependent) plots! First three are color maps of the model results for the dependent variable as a function of the independent variables and of the residuals (with and without division of the error)
    # Initialization of grids to use
    indep_fine = np.zeros((2,fine_grid))
    indep_div = np.zeros((2,bins+1))
    indep_avg = np.zeros((2,bins))
    znum = np.zeros((bins,bins),dtype=int) # Will fill with numbers of galaxies
    for i in range(ndim):
        for j in range(i+1,ndim):
            # Define grids of the independent variables: a fine one (xx_fine) and a coarse one (xx_div)
            for ii,k in enumerate(list((i,j))):
                indep_fine[ii] = np.linspace(min(indep_true[k]),max(indep_true[k]),fine_grid)
                indep_div[ii] = np.linspace(min(indep_true[k]),max(indep_true[k])+1.0e-8,bins+1)
                indep_avg[ii] = np.linspace((indep_div[ii][0]+indep_div[ii][1])/2.0,(indep_div[ii][-1]+indep_div[ii][-2])/2.0,bins)

            xx_fine = np.meshgrid(*indep_fine)
            xx_div = np.meshgrid(*indep_avg)
            for k in range(i-1,-1,-1):
                medval = np.median(indep_true[k])
                xx_fine.insert(0,medval*np.ones_like(xx_fine[0]))
                xx_div.insert(0,medval*np.ones_like(xx_div[0]))
            for k in range(i+1,j):
                medval = np.median(indep_true[k])
                xx_fine.insert(-1,medval*np.ones_like(xx_fine[0]))
                xx_div.insert(-1,medval*np.ones_like(xx_div[0]))
            for k in range(j+1,ndim):
                medval = np.median(indep_true[k])
                xx_fine.insert(k,medval*np.ones_like(xx_fine[0]))
                xx_div.insert(k,medval*np.ones_like(xx_div[0]))
            # Evaluate the model and error/residuals at the fine grid
            # shlist = list(xx_fine[0].shape)
            # shlist.insert(0,numsamp)
            # n_sim_fine = np.zeros(tuple(shlist))
            # for ii, ind in enumerate(inds):
            #     ngrid_mod = getattr(trace,grid_name)[ind]
            #     width_mod = np.exp(getattr(trace,width_name)[ind])
            #     coefs_mod = polyfitnd(xx,ngrid_mod)
            #     n_sim_fine[ii] = calc_poly(xx_fine,coefs_mod,degree) + width_mod * np.random.randn(*n_sim_fine[ii].shape)
            # n_err_fine = np.std(n_sim_fine,axis=0)
            n_quad_err_fine = np.sqrt(width_med**2+width_true**2)
            n_med_fine = calc_poly(xx_fine,coefs_med,degree) # + width_med * np.random.randn(*xx_fine[0].shape)
            n_true_fine = calc_poly(xx_fine,coefs_true,degree) # + width_true * np.random.randn(*xx_fine[0].shape)
            # resid_norm_fine = (n_med_fine-n_true_fine)/n_quad_err_fine
            # Determine number of galaxies in each coarse bin square
            for l in range(bins):
                condj = np.logical_and(indep_true[j]>=indep_div[1][l],indep_true[j]<indep_div[1][l+1])
                for k in range(bins):
                    condi = np.logical_and(indep_true[i]>=indep_div[0][k],indep_true[i]<indep_div[0][k+1])
                    cond = np.logical_and(condi,condj)
                    znum[l,k] = len(n_true[cond])

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_med_fine,xx_div[i]+med_arr[i],xx_div[j]+med_arr[j],znum,xx[i]+med_arr[i],xx[j]+med_arr[j],img_dir,'Medianv2_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='Median model %s'%(dep_lab),xtrue=indep_true[i]+med_arr[i],ytrue=indep_true[j]+med_arr[j])
            # plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],resid_norm_fine,xx_div[i]+med_arr[i],xx_div[j]+med_arr[j],znum,img_dir,'Resid_scale_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],cmap='bwr',zlab=r'$\frac{%s_{\rm model}-%s_{\rm true}}{\sigma_{%s}}$'%(dep_name,dep_name,dep_name))
            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_med_fine-n_true_fine,xx_div[i]+med_arr[i],xx_div[j]+med_arr[j],znum,xx[i]+med_arr[i],xx[j]+med_arr[j],img_dir,'Residv2_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],cmap='bwr',zlab=r'$%s_{\rm model}-%s_{\rm true}$'%(dep_name,dep_name),xtrue=indep_true[i]+med_arr[i],ytrue=indep_true[j]+med_arr[j])

            # Histogram of residuals/error and fit a normal distribution to them--ideal result is standard normal
            # plot_hist_resid(resid_norm_fine.ravel(),img_dir,'hist_resid_fine'+extratext+'_'+indep_name[i]+indep_name[j]+dep_name)

def get_relevant_info_ND_Gen(trace,a_poly_T2,xx_true,xx,ngrid_true,coefs_true,n_true,width_true,indep_true,indep_true2,indep_samp,indep_samp2,n_samp,med_arr_true,med_arr,indep_name_true,indep_name,indep_lab,dep_name='n',dep_lab='n',degree=2,degree2=1,numsamp=50,levels=10,extratext='',fine_grid=201,bins=20,img_dir_orig=op.join('DataSim','2DTests'),grid_name='ngrid',width_name='log_width'):
    nlen = len(n_true)
    ndim, ndim2 = len(indep_true), len(indep_true2)
    img_dir = op.join(img_dir_orig,'deg_%d_deg2_%d'%(degree,degree2),extratext)

    # Determine median model values for the dependent variable
    ngrid_med = np.median(getattr(trace,grid_name),axis=0)
    width_med = np.exp(np.median(getattr(trace,width_name)))
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
    inds = np.random.choice(len(trace.model_logp),size=numsamp,replace=True)
    n_sim = np.zeros((numsamp,nlen))
    for i, ind in enumerate(inds):
        ngrid_mod = getattr(trace,grid_name)[ind]
        width_mod = np.exp(getattr(trace,width_name)[ind])
        coefs_mod = polyfitnd(xx,ngrid_mod)
        n_sim[i] = calc_poly(indep_true2,coefs_mod,degree2) + width_mod * np.random.randn(*n_sim[i].shape)
    # n_err = np.std(n_sim,axis=0)
    # n_quad_err = np.sqrt(n_err**2+width_true**2)
    n_quad_err = np.sqrt(width_med**2+width_true**2)
    resid_norm = (n_med-n_true)/n_quad_err

    # Next plot compares model to the truth very simply
    if degree2==degree and indep_name_true==indep_name: ngrid_true_plot = True
    else: ngrid_true_plot = False
    plot_model_true(n_true,n_med,ngrid_true,ngrid_med,img_dir,'PolyND_%s%s'%(dep_name,extratext),n_err=width_med,width_true=width_true,all_std=all_std,ylab='Median model %s'%(dep_lab),xlab='True %s'%(dep_lab),ngrid_true_plot=ngrid_true_plot)

    # Histogram of residuals/error and fit a normal distribution to them--ideal result is standard normal
    plot_hist_resid(resid_norm,img_dir,'hist_resid_sample_'+dep_name+extratext)

    # Plot general percentile comparison
    plot_percentile(n_sim,n_true,img_dir,'percentile_'+dep_name+extratext)

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

            indep_fine, indep_fine_true = med_arr[:,None]*np.ones((ndim2,fine_grid)), med_arr_true[:,None]*np.ones((ndim,fine_grid))
            # Define grids of the independent variables: a fine one (xx_fine) and a coarse one (xx_div)
            index = 0
            for k,l in zip([i,j],[indtruei,indtruej]):
                indep_fine[k] = np.linspace(min(indep_true2[k]),max(indep_true2[k]),fine_grid)
                indep_fine_true[l] = indep_fine[k]
                indep_div[index] = np.linspace(min(indep_true2[k]),max(indep_true2[k])+1.0e-8,bins+1)
                indep_avg[index] = np.linspace((indep_div[k][0]+indep_div[k][1])/2.0,(indep_div[k][-1]+indep_div[k][-2])/2.0,bins)
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

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_med_fine,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'Medianv2_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='Median model %s'%(dep_lab),xtrue=indep_true2[i]+med_arr[i],ytrue=indep_true2[j]+med_arr[j])

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_true_fine,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx_true[indtruei].ravel()+med_arr_true[indtruei],xx_true[indtruej].ravel()+med_arr_true[indtruej],img_dir,'True_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='True %s'%(dep_lab),xtrue=indep_true[indtruei]+med_arr_true[indtruei],ytrue=indep_true[indtruej]+med_arr_true[indtruej])

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_med_fine-n_true_fine,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'Residv2_%s_%s_%s'%(dep_name,indep_name[i],indep_name[j])+extratext,levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],cmap='bwr',zlab=r'$%s_{\rm model}-%s_{\rm true}$'%(dep_name,dep_name),xtrue=indep_true2[i]+med_arr[i],ytrue=indep_true2[j]+med_arr[j])

def main():
    RANDOM_SEED = 8929
    np.random.seed(RANDOM_SEED)

    prop_dict, dep_dict = {}, {}
    prop_dict['names'] = {'logM': 'logM', 'ssfr': 'logsSFR', 'logZ': 'logZ', 'z': 'z'}
    dep_dict['names'] = {'tau2': 'dust2', 'n': 'n'}
    prop_dict['labels'] = {'logM': r'$\log M_*$', 'ssfr': r'$\log$ sSFR$_{\rm{100}}$', 'logZ': r'$\log (Z/Z_\odot)$', 'z': 'z'}
    dep_dict['labels'] = {'tau2': r"$\hat{\tau}_{\lambda,2}$", 'n': 'n'}
    prop_dict['sigma'] = {'logM': 0.0732, 'ssfr': 0.214, 'logZ': 0.189, 'z': 0.005}
    dep_dict['sigma'] = {'tau2': 0.116, 'n': 0.191}
    prop_dict['med'] = {'logM': 9.91, 'ssfr': -9.30, 'logZ': -0.26, 'z': 1.10}
    prop_dict['min'] = {'logM': -1.193, 'ssfr': -2.697, 'logZ': -1.710, 'z': -0.605}
    prop_dict['max'] = {'logM': 1.907, 'ssfr': 1.393, 'logZ': 0.446, 'z': 1.894}
    args = parse_args()
    print(args)
    img_dir_orig = op.join('DataSim',args.dir_orig)

    if args.data:
        with (open("3dhst_resample_10.pickle",'rb')) as openfile:
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
        cond = np.logical_and.reduce((logMavg>=masscomp,z<3.0,logssfravg>-14.0))
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
            med_mod = np.append(med,np.median(logMavg[indfin]))
        if args.ssfr_mod: 
            indep_true2 = np.append(indep_true2,logssfravg[indfin][None,:]-np.median(logssfravg[indfin]),axis=0)
            med_mod = np.append(med,np.median(logssfravg[indfin]))
        if args.logZ_mod: 
            indep_true2 = np.append(indep_true2,logZavg[indfin][None,:]-np.median(logZavg[indfin]),axis=0)
            med_mod = np.append(med,np.median(logZavg[indfin]))
        if args.z_mod: 
            indep_true2 = np.append(indep_true2,z[indfin][None,:]-np.median(z[indfin]),axis=0)
            med_mod = np.append(med,np.median(z[indfin]))

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

    indep_name, indep_lab, sigarr = [], [], np.array([])
    indep_name2, indep_lab2, sigarr2 = [], [], np.array([])
    for name in prop_dict['names'].keys():
        if getattr(args,name):
            indep_name.append(prop_dict['names'][name])
            indep_lab.append(prop_dict['labels'][name])
            sigarr = np.append(sigarr,prop_dict['sigma'][name])
        if getattr(args,name+'_mod'):
            indep_name2.append(prop_dict['names'][name])
            indep_lab2.append(prop_dict['labels'][name])
            sigarr2 = np.append(sigarr2,prop_dict['sigma'][name])
    if args.n: 
        sigN, dep_name, dep_lab, nlim = dep_dict['sigma']['n'], dep_dict['names']['n'], dep_dict['labels']['n'], np.array([-1.0,0.4])
    else:
        sigN, dep_name, dep_lab, nlim = dep_dict['sigma']['tau2'], dep_dict['names']['tau2'], dep_dict['labels']['tau2'], np.array([0.0,4.0])

    # trace, a_poly_T, xx, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_samp, n_samp = polyND(img_dir_orig, limlist=limlist, nlim=nlim, size=args.size,samples=args.samples,tune=args.tune,errmult=args.error_mult,errmultn=args.error_mult_n,plot=args.plot,extratext=args.extratext,degree=args.degree,sampling=args.steps,sigarr=sigarr,sigN=sigN,indep_true=indep_true,n=n)

    # trace, a_poly_T, a_poly_T2, xx, xx2, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_true2, indep_samp, indep_samp2, n_samp = polyNDgen(img_dir_orig, limlist=limlist, nlim=nlim, size=args.size,samples=args.samples,tune=args.tune,errmult=args.error_mult,errmultn=args.error_mult_n,plot=args.plot,extratext=args.extratext,degree=args.degree,sampling=args.steps,sigarr=sigarr,sigN=sigN,indep_true=indep_true,n=n,degree2=args.degree2,limlist2=limlist2,indep_true2=indep_true2,sigarr2=sigarr2,indep_name=indep_name,indep_name2=indep_name2)

    trace, a_poly_T, xx, ngrid_true, taugrid_true, coefs_true, coefs_true2, n_true, tau_true, width_true, width_true2, rho_true, indep_true, indep_samp, n_samp, tau_samp = polyND_bivar(img_dir_orig, limlist=limlist, nlim=nlim, size=args.size,samples=args.samples,tune=args.tune,errmult=args.error_mult,errmultn=args.error_mult_n,errmultt=args.error_mult_t,errmultcross=args.error_mult_cross,plot=args.plot,extratext=args.extratext,degree=args.degree,sampling=args.steps,sigarr=sigarr,sigN=sigN,sigT=dep_dict['sigma']['tau2'],indep_true=indep_true,n=n,tau=None)
    
    get_relevant_info_ND(trace, a_poly_T, xx, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_samp, n_samp, med, indep_name, indep_lab, dep_name=dep_name, dep_lab=dep_lab, degree=args.degree,extratext=args.extratext,numsamp=500,img_dir_orig=img_dir_orig)

    get_relevant_info_ND(trace, a_poly_T, xx, taugrid_true, coefs_true2, tau_true, width_true2, indep_true, indep_samp, tau_samp, med, indep_name, indep_lab, dep_name=dep_dict['names']['tau2'], dep_lab=dep_dict['labels']['tau2'], degree=args.degree,extratext=args.extratext,numsamp=500,img_dir_orig=img_dir_orig,grid_name='taugrid',width_name='log_width2')

    # get_relevant_info_ND_Gen(trace, a_poly_T2, xx, xx2, ngrid_true, coefs_true, n_true, width_true, indep_true, indep_true2, indep_samp, indep_samp2, n_samp, med, med_mod, indep_name, indep_name2, indep_lab2, dep_name=dep_name, dep_lab=dep_lab, degree=args.degree,degree2=args.degree2,extratext=args.extratext,numsamp=500,img_dir_orig=img_dir_orig)

if __name__=='__main__':
    main()