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
from PlotColorMapProj import getModelSamplesI, makeColorMapProjI
from RegularLinearInterp import regular_grid_interp, regular_grid_interp_scipy
import DustAttnCurveModules as dac
import argparse as ap
from scipy.stats import norm, truncnorm
from scipy.integrate import trapz
from copy import copy, deepcopy
from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool
from itertools import product
from astropy.table import Table
import corner

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
fl_fit_stats_univar = 'run_convergence_univar_interp.txt'
fl_fit_stats_bivar = 'run_convergence_bivar_interp.txt'
fig_size = (12,5)
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

def parse_args(argv=None):
    """ Tool to parse arguments from the command line. The entries should be self-explanatory """
    parser = ap.ArgumentParser(description="DustPymc3",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-si','--size',help='Number of mock galaxies',type=int,default=500)
    parser.add_argument('-sa','--samples',help='Number of posterior samples per galaxy',type=int,default=50)
    parser.add_argument('-pl','--plot',help='Whether or not to plot ArViz plot',action='count',default=0)
    parser.add_argument('-tu','--tune',help='Number of tuning steps',type=int,default=1000)
    parser.add_argument('-st','--steps',help='Number of desired steps included per chain',type=int,default=1000)
    parser.add_argument('-gp','--grid_pts',help='Number of grid points (approx) for interpolation',type=int,default=-1)
    parser.add_argument('-ext','--extratext',help='Extra text to help distinguish particular run',type=str,default='')
    parser.add_argument('-dir','--dir_orig',help='Parent directory for files',type=str,default='NewTests')

    parser.add_argument('-m','--logM',help='Whether or not to include stellar mass (true)',action='count',default=0)
    parser.add_argument('-s','--ssfr',help='Whether or not to include sSFR (true)',action='count',default=0)
    parser.add_argument('-logZ','--logZ',help='Whether or not to include metallicity (true)',action='count',default=0)
    parser.add_argument('-z','--z',help='Whether or not to include redshift (true)',action='count',default=0)
    parser.add_argument('-i','--i',help='Whether or not to include inclination in model',action='count',default=0)
    parser.add_argument('-d1','--d1',help='Whether or not to include birth cloud dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-d2','--d2',help='Whether or not to include diffuse dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-n','--n',help='Whether or not to use dust index as dependent variable',action='count',default=0)
    parser.add_argument('-bv','--bivar',help='Whether or not to perform bivariate fitting',action='count',default=0)
    parser.add_argument('-ad','--already_done',help='Whether fit was already finished',action='count',default=0)

    args = parser.parse_args(args=argv)
    if not args.n: args.d2_mod = 0
    if args.bivar: args.dir_orig = op.join('Interp','Bivar')
    else: args.dir_orig = op.join('Interp','Univar')

    return args

def polyNDDataI(indep_samp,dep_samp,logp_prior,img_dir_orig,plot=False,extratext='',sampling=1000,tune=1000,dep_lim=np.array([-1.0,0.4]),uniform=True,grid_pts=1000):
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
    if uniform: dep_name = 'n'
    else: dep_name = 'd2'
    img_dir = op.join(img_dir_orig,extratext)
    mkpath(img_dir)
    ndim = len(indep_samp)
    nlen, nwid = len(dep_samp), len(dep_samp[0])
    gridlen = int(np.round(grid_pts**(1/ndim)))
    grid_pts_actual = gridlen**ndim
    per = np.linspace(1.0,99.0,gridlen)
    x = np.empty((ndim,gridlen))
    for i in range(ndim):
        arr = indep_samp[i].ravel()
        # cond = np.logical_and(arr>np.amin(arr),arr<np.amax(arr))
        # x[i,1:-1] = np.sort(np.random.choice(arr[cond],size=gridlen-2,replace=False))
        # x[i,0] = np.amin(arr); x[i,-1] = np.amax(arr)
        x[i] = np.percentile(arr,per)
    xtup = tuple(x) # For interpolation calculation
    indepI, depI = indep_samp.reshape(ndim,nlen*nwid).T, dep_samp.reshape(nlen*nwid).T
    sh = tuple([gridlen]*ndim)
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        ngrid = pm.Uniform("ngrid",lower=dep_lim[0]-1.0e-5,upper=dep_lim[1]+1.0e-5,shape=sh,testval=np.random.uniform(dep_lim[0],dep_lim[1],sh))
        log_width = pm.StudentT("log_width", nu=5, mu=-3.0, sigma=0.5, testval=-3.5)

        # Compute the expected n at each sample
        mu = regular_grid_interp(xtup,ngrid,indepI)

        # The line has some width: we're calling it a Gaussian in n
        logp_flat = -0.5 * (depI - mu) ** 2 * pm.math.exp(-2*log_width) - log_width - logp_prior
        logp = tt.reshape(logp_flat,(nlen,nwid))

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=True, chains=4, discard_tuned_samples=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"interp%s_%d_trace.png"%(extratext,grid_pts_actual)),bbox_inches='tight',dpi=300)
            trace_df2 = trace.to_dataframe(groups='posterior')
            trace_df = trace.to_dataframe(groups='warmup_posterior')
            trace_df = trace_df.append(trace_df2,ignore_index=True)
            try:
                randinds = np.random.choice(gridlen,2,replace=False)
                lr, hr = min(randinds), max(randinds)
                cols_to_keep = [(f'ngrid[{lr}]', lr), (f'ngrid[{hr}]', hr), 'log_width']
                fig = corner.corner(trace_df[cols_to_keep])
                fig.savefig(op.join(img_dir,"trace_with_tune%s_%d.png"%(extratext,grid_pts_actual)))
                fig = corner.corner(trace_df2[cols_to_keep])
                fig.savefig(op.join(img_dir,"trace%s_%d.png"%(extratext,grid_pts_actual)))
            except: pass
        print(az.summary(trace,round_to=2))

    return trace, xtup

def polyNDDataBivarI(indep_samp,n_samp,tau_samp,logp_prior,img_dir_orig,plot=False,extratext='',sampling=1000,tune=1000,nlim=np.array([-1.0,0.4]),taulim=np.array([0.0,4.0]),grid_pts=1000):
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
    img_dir = op.join(img_dir_orig,extratext)
    mkpath(img_dir)
    ndim = len(indep_samp)
    nlen, nwid = len(n_samp), len(n_samp[0])
    gridlen = int(np.round(grid_pts**(1/ndim)))
    grid_pts_actual = gridlen**ndim
    per = np.linspace(1.0,99.0,gridlen)
    x = np.empty((ndim,gridlen))
    for i in range(ndim):
        arr = indep_samp[i].ravel()
        x[i] = np.percentile(arr,per)
    xtup = tuple(x) # For interpolation calculation
    indepI, nI, tauI = indep_samp.reshape(ndim,nlen*nwid).T, n_samp.reshape(nlen*nwid).T, tau_samp.reshape(nlen*nwid).T
    sh = tuple([gridlen]*ndim)
    
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        ngrid = pm.Uniform("ngrid",lower=nlim[0]-1.0e-5,upper=nlim[1]+1.0e-5,shape=sh,testval=np.random.uniform(nlim[0],nlim[1],sh))
        taugrid = pm.Uniform("taugrid",lower=taulim[0]-1.0e-5,upper=taulim[1]+1.0e-5,shape=sh,testval=np.random.uniform(taulim[0],taulim[1],sh))
        log_width = pm.StudentT("log_width", nu=5, mu=-3.0, lam=0.5, testval=-3.5)
        log_width2 = pm.StudentT("log_width2", nu=5, mu=-2.9, lam=0.48, testval=-3.8)
        rho = pm.Uniform("rho",lower=-1.0,upper=1.0,testval=np.random.uniform(-0.5,0.5))

        # Compute the expected n at each sample
        mu = regular_grid_interp(xtup,ngrid,indepI)
        mu2 = regular_grid_interp(xtup,taugrid,indepI)

        # The line has some width: we're calling it a Gaussian in n,tau
        zbiv = (nI-mu)**2 * pm.math.exp(-2*log_width) + (tauI-mu2)**2 * pm.math.exp(-2*log_width2) - 2 * rho * (nI-mu) * (tauI-mu2) * pm.math.exp(-log_width-log_width2)

        logp_flat = -0.5 * zbiv/(1.0-rho**2) - log_width - log_width2 - pm.math.log(pm.math.sqrt(1-rho**2)) - logp_prior
        logp = tt.reshape(logp_flat,(nlen,nwid))

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=True, chains=4, discard_tuned_samples=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"interpBV%s_%d_trace.png"%(extratext,grid_pts_actual)),bbox_inches='tight',dpi=300)
            try:
                trace_df2 = trace.to_dataframe(groups='posterior')
                trace_df = trace.to_dataframe(groups='warmup_posterior')
                trace_df.append(trace_df2,ignore_index=True)
                randinds = np.random.randint(0,gridlen,2)
                lr, hr = randinds[0], randinds[1]
                cols_to_keep = [(f'ngrid[{lr}]', lr), (f'taugrid[{hr}]', hr), 'log_width','log_width2','rho']
                fig = corner.corner(trace_df[cols_to_keep])
                fig.savefig(op.join(img_dir,"trace_with_tune%s_%d.png"%(extratext,grid_pts_actual)))
                fig = corner.corner(trace_df2[cols_to_keep])
                fig.savefig(op.join(img_dir,"trace%s_%d.png"%(extratext,grid_pts_actual)))
            except: pass
        print(az.summary(trace,round_to=2))

    return trace, xtup

def plot1D(x,y,xx,yy,img_dir,name,xlab=r'$\log\ M_*$',ylab='Median model n',xerr=None,yerr=None,ngrid_err=None):
    fig, ax = plt.subplots()
    indsort = np.argsort(x)
    ax.plot(x[indsort],y[indsort],'b-.',markersize=1)
    if yy is not None: ax.plot(xx,yy,'r^',markersize=2)
    if xerr is not None or yerr is not None: ax.errorbar(x,y,yerr=yerr,xerr=xerr,fmt='none',ecolor='b',alpha=np.sqrt(1.0/len(x)))
    if ngrid_err is not None and yy is not None: ax.errorbar(xx,yy,yerr=ngrid_err,fmt='none',ecolor='r',alpha=1.0)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    fig.savefig(op.join(img_dir,'%s.png'%(name)),bbox_inches='tight',dpi=200)

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
    
def get_relevant_info_ND_Data_I(trace,xtup,indep_samp,n_samp,indep_name,indep_lab,dep_name='n',dep_lab='n',levels=10,extratext='',fine_grid=201,bins=20,img_dir_orig=op.join('DataTrue','NewTests'),grid_name='ngrid',width_name='log_width',numsamp=50,fl_stats=fl_fit_stats_univar,append_to_file=False,correction=False,rand1=None,rand11=None,rand_add=True,bivar=False,obj=None,indfin=None,sliced=False,indep_pickle_name=None,tau_samp=None):
    nlen, nwid = len(n_samp), len(n_samp[0])
    ndim = len(indep_samp)
    indepI = indep_samp.reshape(ndim,nlen*nwid).T
    img_dir = op.join(img_dir_orig,extratext)
    # Determine median model values for the dependent variable
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,grid_name)), np.array(getattr(trace.posterior,width_name))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],*sh[2:]), log_width_0.reshape(sh[0]*sh[1])
    grid_pts_actual = ngrid[0].size
    if bivar:
        taugrid_0, log_width2_0 = np.array(getattr(trace.posterior,'taugrid')), np.array(getattr(trace.posterior,width_name))
        taugrid, log_width2 = taugrid_0.reshape(sh[0]*sh[1],*sh[2:]), log_width2_0.reshape(sh[0]*sh[1])
        taugrid_med = np.median(taugrid,axis=0)
        rho_0 = np.array(getattr(trace.posterior,'rho'))
        rho = rho_0.reshape(sh[0]*sh[1])
    else:
        taugrid, log_width2, rho = None, None, None
    ngrid_med = np.median(ngrid,axis=0)
    sh_nm = ngrid_med.shape
    width_med = np.exp(np.median(log_width))
    width_mean = np.exp(np.mean(log_width))
    indep_avg = np.mean(indep_samp,axis=2)

    n_sim, tau_sim = getModelSamplesI(xtup,indep_avg,ngrid,log_width,taugrid,log_width2,rho,numsamp=numsamp,poly_only=not rand_add)
    n_mean, n_err = np.mean(n_sim,axis=0), np.std(n_sim,axis=0)

    if append_to_file:
        dataarr = np.empty((ngrid_med.size,0))
        xx_tup = np.meshgrid(*xtup)
        colnames = []
        for i in range(len(xtup)):
            dataarr = np.append(dataarr,xx_tup[i].reshape(ngrid_med.size,1),axis=1)
            colnames.append(indep_name[i])
        dataname = ''
        for name in indep_name: dataname += name+'_'
        if 'univar' in fl_stats: 
            dataarr = np.append(dataarr,ngrid_med.reshape(np.prod(sh_nm),1),axis=1); colnames.append(grid_name)
            dataname+=dep_name
        else: 
            dataarr = np.append(dataarr,ngrid_med.reshape(np.prod(sh_nm),1),axis=1); colnames.append('ngrid')
            dataarr = np.append(dataarr,taugrid_med.reshape(np.prod(sh_nm),1),axis=1); colnames.append('taugrid')
            dataname+='n_dust2'
        # np.savetxt(op.join(img_dir,'ngrid_%s%s_%s_vi%d.dat'%(dataname,dep_name,extratext,var_inf)),dataarr,header='%s  ngrid'%('  '.join(indep_name)),fmt='%.5f')
        t = Table(dataarr,names=colnames)
        t.write(op.join(img_dir,'Interp_%s_%s_%d_HB.dat'%(dataname,extratext,grid_pts_actual)),overwrite=True,format='ascii')
        if not correction: trace.to_netcdf(op.join(img_dir,'trace_int_%s_%s_%d.nc'%(dataname,extratext,grid_pts_actual)))
        rhat_arr = []
        rhats = az.rhat(trace)
        rhat_keys = list(rhats.keys())
        for key in rhat_keys:
            val = rhats[key].values
            if val.ndim==0: rhat_arr+=[float(val)]
            else: rhat_arr+=[np.amax(val)]
        num_div = len(trace.sample_stats.diverging.values.nonzero()[0])
        print("rhat_keys:", rhat_keys)
        rhat_argsort = np.argsort(rhat_keys)
        with open(fl_stats,'a') as fl:
            fl.write('%s_%s  '%(dataname, extratext))
            fl.write(f'{ndim}  {xtup[0].size}  {nlen}  {nwid}  {len(log_width)}  {num_div}  ')
            for i_as in rhat_argsort: fl.write('%.4f  '%(rhat_arr[i_as]))
            if 'bivar' in fl_stats: fl.write('%.4f  %.4f  %.4f \n'%(np.log(width_med),np.median(log_width2),np.median(rho)))
            else: fl.write('%.4f \n'%(np.log(width_med)))

    plot_model_true(np.mean(n_samp,axis=1),n_mean,None,None,img_dir,'Real_%s_comp_%s'%(dep_name,extratext),n_err=n_err,width_true=np.std(n_samp,axis=1),ylab='Mean posterior model %s (at mean location)'%(dep_lab), xlab='Observed %s (Prospector posterior means)'%(dep_lab),ngrid_true_plot=False)

    if bivar: 
        tau_mean, tau_err = np.mean(tau_sim,axis=0), np.std(tau_sim,axis=0)
        plot_model_true(np.mean(tau_samp,axis=1),tau_mean,None,None,img_dir,'Real_d2_comp_%s'%(extratext),n_err=tau_err,width_true=np.std(tau_samp,axis=1),ylab='Mean posterior model %s (at mean location)'%(taulab), xlab='Observed %s (Prospector posterior means)'%(taulab),ngrid_true_plot=False)

    if ndim==1:
        x = np.linspace(np.amin(indep_samp[0]),np.amax(indep_samp[0]),fine_grid)
        n_sim2, tau_sim2 = getModelSamplesI(xtup,x[None,:],ngrid,log_width,taugrid,log_width2,rho,numsamp=numsamp,poly_only=not rand_add)
        n_mean2, n_err2 = np.mean(n_sim2,axis=0), np.std(n_sim2,axis=0)
        plot1D(x,n_mean2,None,None,img_dir,'Real_Model_%s_%s_%s_rd_%d_gp_%d'%(indep_name[0],dep_name,extratext,rand_add,grid_pts_actual),xlab=indep_lab[0],ylab='Mean posterior model '+dep_lab,yerr=n_err2)
        if bivar:
            tau_mean2, tau_err2 = np.mean(tau_sim2,axis=0), np.std(tau_sim2,axis=0)
            plot1D(x,tau_mean2,None,None,img_dir,'Real_Model_%s_d2_%s_rd_%d_gp_%d'%(indep_name[0],extratext,rand_add,grid_pts_actual),xlab=indep_lab[0],ylab='Mean posterior model '+taulab,yerr=tau_err2)
        return

    makeColorMapProjI(trace,xtup,indep_samp,indep_lab,indep_name,indep_pickle_name,extratext,img_dir,bivar=bivar,poly_only=not rand_add,numsamp=numsamp,sliced=False,fine_grid=fine_grid,numslices=5,numsamples=nwid,numgal=None,dep_name=dep_name,extrawords=f'gp_{grid_pts_actual}')  
    return

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
            data[:,i] = np.array(obj[prop])[cond]
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

def make_prop_dict():
    prop_dict, dep_dict = {}, {}
    prop_dict['names'] = {'logM': 'logM', 'ssfr': 'logsSFR', 'logZ': 'logZ', 'z': 'z', 'i':'axis_ratio', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['names'] = {'tau2': 'dust2', 'n': 'n'}
    prop_dict['labels'] = {'logM': r'$\log M_*$', 'ssfr': r'$\log$ sSFR$_{\rm{100}}$', 'logZ': r'$\log (Z/Z_\odot)$', 'z': 'z', 'i':r'$b/a$', 'd1':r"$\hat{\tau}_{1}$", 'd2':r"$\hat{\tau}_{2}$"}
    dep_dict['labels'] = {'tau2': r"$\hat{\tau}_{2}$", 'n': 'n'}
    prop_dict['pickle_names'] = {'logM': 'stellar_mass', 'ssfr': 'ssfr_100', 'logZ': 'log_z_zsun', 'z': 'z', 'i':'inc', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['pickle_names'] = {'tau2': 'dust2', 'n': 'dust_index'}
    prop_dict['sigma'] = {'logM': 0.0732, 'ssfr': 0.214, 'logZ': 0.189, 'z': 0.005, 'i': 0.06, 'd1': 0.136, 'd2': 0.116}
    dep_dict['sigma'] = {'tau2': 0.116, 'n': 0.191}
    prop_dict['med'] = {'logM': 9.91, 'ssfr': -9.30, 'logZ': -0.26, 'z': 1.10, 'i': 0.485, 'd1': 0.221, 'd2': 0.239}
    prop_dict['min'] = {'logM': -1.193, 'ssfr': -2.697, 'logZ': -1.710, 'z': -0.605, 'i': 0, 'd1': 0, 'd2': 0}
    prop_dict['max'] = {'logM': 1.907, 'ssfr': 1.393, 'logZ': 0.446, 'z': 1.894, 'i': 1.0, 'd1': 4.8, 'd2': 4.0}
    return prop_dict, dep_dict

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
    # comb = np.array([ssfr,logZ])
    per = np.percentile(ssfr,[1.0,99.0])
    masscomp = mass_completeness(z)
    # cond = np.logical_and.reduce((logMavg>=masscomp,z<3.0,np.amin(ssfr,axis=1)>ssfr_lims_data[0],np.amax(ssfr,axis=1)<ssfr_lims_data[1],logssfravg>per[0][0],logssfravg<per[1][0],logZavg>per[1][0],logZavg<per[1][1]))
    cond = np.logical_and.reduce((logMavg>=masscomp,logssfravg>per[0]))
    
    if args.i: cond = np.logical_and.reduce((cond,np.median(inc,axis=1)>=0.0,np.median(inc,axis=1)<=1.0))

    lenzcond = len(z[cond])
    print("Length of z cond: ", lenzcond)
    if args.size == -1 or args.size>lenzcond: args.size = lenzcond

    ind = np.where(cond)[0]
    indfin = np.random.choice(ind,size=args.size,replace=False)
    indep_samp, prior_samp = np.empty((0,len(z[indfin]),len(n[0]))), np.empty((0,len(z[indfin]),len(n[0])))
    prior_prop, prior_lab = [], []
    if args.logM: 
        indep_samp = np.append(indep_samp,logM[indfin][None,:,:],axis=0)
        prior_samp = np.append(prior_samp,logM[indfin][None,:,:],axis=0)
        prior_prop.append('stmass')
        prior_lab.append(r'$\log M_*$')
    if args.ssfr: 
        indep_samp = np.append(indep_samp,ssfr[indfin][None,:,:],axis=0)
        prior_samp = np.append(prior_samp,ssfr[indfin][None,:,:],axis=0)
        prior_prop.append('ssfr')
        prior_lab.append(r'$\log$ sSFR$_{\rm{100}}$')
    if args.logZ: 
        indep_samp = np.append(indep_samp,logZ[indfin][None,:,:],axis=0)
        prior_samp = np.append(prior_samp,logZ[indfin][None,:,:],axis=0)
        prior_prop.append('met')
        prior_lab.append(r'$\log (Z/Z_\odot)$')
    if args.z: 
        zrep = np.repeat(z[indfin][:,None],samples,axis=1)
        indep_samp = np.append(indep_samp,zrep[None,:,:],axis=0)
    if args.i:
        indep_samp = np.append(indep_samp,inc[indfin][None,:,:],axis=0)
    if args.d1:
        indep_samp = np.append(indep_samp,tau1[indfin][None,:,:],axis=0)
        prior_samp = np.append(prior_samp,tau1[indfin][None,:,:],axis=0)
        prior_prop.append('dust1')
        prior_lab.append(r"$\hat{\tau}_{1}$")
    if args.d2:
        indep_samp = np.append(indep_samp,tau2[indfin][None,:,:],axis=0)
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

    return indep_samp, dep_samp, prior_samp, z[indfin], prior_prop, prior_lab, uniform

def label_creation(args,prop_dict,dep_dict):
    indep_pickle_name, indep_name, indep_lab = [], [], []
    for name in prop_dict['names'].keys():
        if getattr(args,name):
            indep_pickle_name.append(prop_dict['pickle_names'][name])
            indep_name.append(prop_dict['names'][name])
            indep_lab.append(prop_dict['labels'][name])
    if args.n: 
        dep_name, dep_lab, nlim = dep_dict['names']['n'], dep_dict['labels']['n'], np.array([-1.0,0.4])
    else:
        dep_name, dep_lab, nlim = dep_dict['names']['tau2'], dep_dict['labels']['tau2'], np.array([0.0,4.0])
    return indep_name, indep_lab, indep_pickle_name, dep_name, dep_lab, nlim

def main(args=None):
    RANDOM_SEED = 8929
    np.random.seed(RANDOM_SEED)

    prop_dict, dep_dict = make_prop_dict()
    if args==None: args = parse_args()
    print(args)

    indep_name, indep_lab, indep_pickle_name, dep_name, dep_lab, nlim = label_creation(args,prop_dict,dep_dict)
    args.indep_name = indep_name
    indep_samp, dep_samp, prior_samp, zarr, prior_prop, prior_lab, uniform = data_true(args)
    ndim = len(indep_name)
    if args.grid_pts==-1: grid_pts = int(min(50 * 2**ndim, (args.size/20) * ndim))
    else: grid_pts = args.grid_pts
    print("Finished getting data")
    if args.already_done:
        img_dir_orig = op.join('DataFinal',args.dir_orig)
        trace, xtup = dac.getPostModelData(args,img_dir_orig=img_dir_orig,run_type='I')
        app_to_file=False
    else:
        img_dir_orig = op.join('DataTrue',args.dir_orig)
        app_to_file=True
        if len(prior_prop)>=1: 
            logp_prior = getKDE(prior_prop,prior_samp,zarr).flatten()
            print("Finished all KDE Calculations")
            print("Nans/infs in logp_prior:",np.isnan(logp_prior).any() or np.isinf(logp_prior).any())
            # logp_prior = 0.0
        else: logp_prior = 0.0
    if args.bivar:
        if not args.already_done:
            trace, xtup = polyNDDataBivarI(indep_samp,dep_samp[0],dep_samp[1],logp_prior,img_dir_orig,plot=args.plot,extratext=args.extratext,sampling=args.steps,tune=args.tune,grid_pts=grid_pts)

        get_relevant_info_ND_Data_I(trace,xtup,indep_samp,dep_samp[0],indep_name,indep_lab,extratext=args.extratext,img_dir_orig=img_dir_orig,fl_stats=fl_fit_stats_bivar,append_to_file=app_to_file,bivar=True,indep_pickle_name=indep_pickle_name,tau_samp=dep_samp[1],rand_add=True,sliced=False)
    else:
        if not args.already_done:
            trace, xtup = polyNDDataI(indep_samp,dep_samp,logp_prior,img_dir_orig,dep_lim=nlim,tune=args.tune,plot=args.plot,extratext=args.extratext,sampling=args.steps,uniform=uniform,grid_pts=grid_pts)

        get_relevant_info_ND_Data_I(trace,xtup,indep_samp,dep_samp,indep_name,indep_lab,dep_name=dep_name,dep_lab=dep_lab,extratext=args.extratext,sliced=False,img_dir_orig=img_dir_orig,indep_pickle_name=indep_pickle_name,fl_stats=fl_fit_stats_univar,append_to_file=app_to_file,rand_add=True,bivar=False)
        

if __name__=='__main__':
    main()