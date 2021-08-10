import numpy as np
import pickle
import pymc3 as pm 
import arviz as az
import matplotlib.pyplot as plt 
import seaborn as sns
from HierBay import plot2DDustMean, mass_completeness
from time import time
# import theano
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value='warn'
import theano.tensor as tt
import os.path as op
from anchor_points import get_a_polynd, calc_poly, calc_poly_tt, polyfitnd
import argparse as ap

RANDOM_SEED = 8929
np.random.seed(RANDOM_SEED)

def parse_args(argv=None):
    parser = ap.ArgumentParser(description="DustPymc3",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-e','--error_mult',help='Factor determining absolute error in simulated data',type=float, default=1.0)
    parser.add_argument('-si','--size',help='Number of mock galaxies',type=int,default=500)
    parser.add_argument('-sa','--samples',help='Number of posterior samples per galaxy',type=int,default=50)
    parser.add_argument('-pl','--plot',help='Whether or not to plot ArViz plot',action='count',default=0)
    parser.add_argument('-tu','--tune',help='Number of tuning steps',type=int,default=1000)
    parser.add_argument('-st','--steps',help='Number of desired steps included per chain',type=int,default=1000)
    parser.add_argument('-ci','--complex_indep',help='Whether or not to sample independent variables from complex distribution',action='count',default=0)
    parser.add_argument('-ext','--extratext',help='Extra text to help distinguish particular run',type=str,default='')
    parser.add_argument('-dg','--degree',help='Degree of polynomial',type=int,default=2)

    args = parser.parse_args(args=argv)

    return args

def quadconst(y1,y2,y3,z1=0.2,z2=1.6,z3=3.0):
    """Compute quadratic coefficients of polynomial given three points."""
    a = (((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) /(z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)))
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a, b, c

def quadnoconst(y1,y2,x1,x2): 
    """ Compute quadratic coefficients of polynomial ax^2+bx given two points """
    b = (x2**2*y1-x1**2*y2)/(x1*x2**2 - x1**2*x2)
    a = y1/x1**2 - b/x1
    return a, b

def lin_mass_ssfr(size=10000,samples=500):
    # True values for parameters; sigM and sigS don't get modeled--they are just the sigmas for a normal distribution from which samples are pulled
    # Other parameters are the same as in lin_simple; sigma is now called sigN
    sigM, sigS = 0.4, 0.6
    alpha, sigN = -0.2, 1.0
    beta = [0.3,0.35]
    logM = 3.0*np.random.rand(size)+8.0
    ssfr = 4.0*np.random.rand(size)-12.0
    # As mentioned earlier, the samples are taken from normal distributions around the uniformly sampled log(M) and log(sSFR) values from before
    logMsamp = sigM*np.random.randn(size*samples).reshape(size,samples)+logM.reshape(size,1)
    ssfrsamp = sigS*np.random.randn(size*samples).reshape(size,samples)+ssfr.reshape(size,1)
    logMavg = np.average(logMsamp,axis=1); ssfravg = np.average(ssfrsamp,axis=1)
    # Simulated dust slope values based on the linear model; same as in lin_simple
    n_sim = alpha + beta[0]*logMsamp + beta[1]*ssfrsamp + np.random.randn(size*samples).reshape(size,samples)*sigN
    navg = np.average(n_sim,axis=1)
    plot2DDustMean(logMavg,ssfravg,navg,r'$\log M_*$','logM',r'$\log$ sSFR$_{\rm{100}}$','ssfr100','n','n',levels=20,extratext='linlogMssfr')
    return n_sim

def quad_every_data(logM,ssfr,logZ,z):
    size, samples = len(logM), len(logM[0])
    logMavg = np.average(logM,axis=1); ssfravg = np.average(ssfr,axis=1)
    logZavg = np.average(logZ,axis=1)
    alpha = 0.0
    sigN = 0.1*(logMavg-ssfravg)+0.05*(z-min(z))
    m1,m2 = 9.0, 11.0
    ssfr1,ssfr2 = -12.0,-9.0
    logZ1,logZ2 = -1.0,-0.1
    z1,z2 = 1.0,2.0
    nm1,nm2 = -0.12,-0.36
    nssfr1,nssfr2 = -0.1,0.2
    nlogZ1,nlogZ2 = -0.2,0.1
    nz1,nz2 = 0.0,0.05
    am,bm = quadnoconst(nm1,nm2,m1,m2)
    assfr,bssfr = quadnoconst(nssfr1,nssfr2,ssfr1,ssfr2)
    alz,blz = quadnoconst(nlogZ1,nlogZ2,logZ1,logZ2)
    az,bz = quadnoconst(nz1,nz2,z1,z2)
    n_sim = alpha + 0.25*(am*logM**2+bm*logM + assfr*ssfr**2+bssfr*ssfr + alz*logZ**2+blz*logZ + az*z[:,None]**2+bz*z[:,None]) + np.random.randn(size*samples).reshape(size,samples)*sigN[:,None]
    navg = np.average(n_sim,axis=1)
    cond = np.logical_and(z>=0.5,z<3.0)
    plot2DDustMean(logMavg[cond],ssfravg[cond],navg[cond],r'$\log M_*$','logM',r'$\log$ sSFR$_{\rm{100}}$','ssfr100','n','n',levels=20,extratext='quadall_full_z')
    plot2DDustMean(logMavg,logZavg,navg,r'$\log M_*$','logM',r'$\log (Z/Z_\odot)$','logZ','n','n',levels=20,extratext='quadall_full_z')
    return n_sim

def lin_data(logM,ssfr,logZ):
    size, samples = len(logM), len(logM[0])
    logMavg = np.average(logM,axis=1); ssfravg = np.average(ssfr,axis=1)
    alpha = 0.2
    sigN = 0.1*(logMavg-ssfravg)
    beta = [0.2,0.25,-0.2]
    n_sim = alpha + beta[0]*logM + beta[1]*ssfr + beta[2]*logZ + np.random.randn(size*samples).reshape(size,samples)*sigN.reshape(size,1)
    navg = np.average(n_sim,axis=1)
    plot2DDustMean(logMavg,ssfravg,navg,r'$\log M_*$','logM',r'$\log$ sSFR$_{\rm{100}}$','ssfr100','n','n',levels=20,extratext='lindatahetlogZ')
    return n_sim

def lin_with_samples(size=500,samples=10,steps=1000,tune=1000,errmult=1.0,plot=False,extratext=''):
    ''' Same as lin_simple but where each galaxy comes with various log(M) and log(sSFR) values--hopefully simulating posterior sampling '''
    # Model statistics parameters
    tot_err, tot_mcse, tot_gr, tot_effn = np.array([]),np.array([]),np.array([]),np.array([])

    # True linear model
    alpha_true = -0.2
    beta_true = np.array([0.3, 0.35])

    # True "width" of the relation
    width_true = 0.05
    dict_true = {'alpha':alpha_true,'beta':beta_true,'log_width':np.log(width_true)}

    # "True" values of galaxy params
    logM_true = np.random.uniform(8.0, 11.0, size)
    ssfr_true = np.random.uniform(-12.0, -8.0, size)
    n_true = alpha_true + beta_true[0] * logM_true + beta_true[1] * ssfr_true + width_true * np.random.randn(size)

    # Observed values of the galaxy parameters
    sigM, sigS = 0.2*errmult, 0.1*errmult
    sigN = 0.1*errmult

    # First we need to "move" the mean of the posterior, then we generate posterior samples
    logM_samp = logM_true[:, None] + sigM * np.random.randn(size, 1) + sigM * np.random.randn(size, samples)
    ssfr_samp = ssfr_true[:, None] + sigS * np.random.randn(size, 1) + sigS * np.random.randn(size, samples)
    n_samp = n_true[:, None] + sigN * np.random.randn(size, 1) + sigN * np.random.randn(size, samples)
    
    # Create the pymc3 model and run it
    with pm.Model() as model:
        # Prior distributions for alpha, beta, and sigN are normals (and half normal for sigN)
        alpha = pm.Normal("alpha", mu=0, sigma=10, testval=alpha_true)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2, testval=beta_true)
        log_width = pm.Normal("log_width", mu=np.log(width_true), sigma=2.0, testval=np.log(width_true))

        # Compute the expected n at each sample
        mu = alpha + beta[0]*logM_samp + beta[1]*ssfr_samp

        # The line has some width: we're calling it a Gaussian in n
        logp = -0.5 * (n_samp - mu) ** 2 * pm.math.exp(-2*log_width) - log_width

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        # max_logp = np.zeros(len(logM_samp))
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        tic = time()
        trace = pm.sample(draws=steps,tune=tune,target_accept=0.9,init='adapt_full', return_inferencedata=False)
        toc = time()

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join('DataSim',"lin_with_samples%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
        print(az.summary(trace,round_to=2))

        gr = az.rhat(trace)
        eff_n = az.ess(trace)
        mcse = az.mcse(trace)
        ind = np.argmax(trace.model_logp)
        for var in trace.varnames:
            best = getattr(trace,var)[ind]
            tot_err = np.append(tot_err,abs((best-dict_true[var])/dict_true[var]))
            tot_gr = np.append(tot_gr,gr[var])
            tot_effn = np.append(tot_effn,eff_n[var])
            tot_mcse = np.append(tot_mcse,mcse[var])

    if plot:
        import corner
        corner.corner(pm.trace_to_dataframe(trace), truths=[alpha_true] + beta_true + [np.log(width_true)])
        plt.savefig(op.join('DataSim',"lin_with_samples%s_corner.png"%(extratext)),bbox_inches='tight',dpi=300)

    return np.median(tot_err),np.median(tot_gr),np.median(tot_effn)/(steps*trace.nchains),np.median(tot_mcse),np.median(abs(trace.energy_error)), toc-tic

def polyevery(size=100,samples=50,plot=False,extratext='',mlim=[8.0,12.0],ssfrlim=[-12.0,-8.0],logzlim=[-1.2,0.2],zlim=[0.5,3.0],nlim=[-1.0,0.4],degree=2,sampling=1000,tune=1000,errmult=1.0,complex_indep=False):
    ''' Simulation of data where the dust index n is a 4-D polynomial function of stellar mass, sSFR, log(Z), and redshift.

    Parameters
    ----------
    size: Int
        Number of "galaxies" in the sample
    samples: Int
        Number of "posterior samples" available for each galaxy
    plot: Bool
        Whether or not an arviz plot should be made
    extratext: Str
        An addition to the plot name to distinguish it from previous runs
    mlim, ssfrlim, logzlim, zlim, nlim: Two-element lists
        Lower and upper limits for all parameters in question
    degree: Int
        Polynomial degree in each variable
    order: Int
        Order of the polynomial; terms with higher combined degrees will be truncated
    sampling: Int
        Number of draws for sampling (not including burn-in steps)
    tune: Int
        Number of burn-in steps for each sampling chain
    errmult: Float
        Multiplicative factor that determines how large the simulated data errors are
    complex_indep: Bool
        If True, independent variables selected from a complicated probability distribution
    Returns
    -------
    all_err: 1-D Numpy Array
        Errors in each fitted parameter (maximum likelihood sample vs true value)
    
    The function runs the pymc3 sampler on this toy example and prints the results; an arviz plot of the results is also made if plot==True
    '''
    # For getting errors between max likelihood solutions and true parameter values

    # Creating independent-variable arrays
    if not complex_indep:
        logM_true = np.random.uniform(mlim[0],mlim[1],size)
        ssfr_true = np.random.uniform(ssfrlim[0],ssfrlim[1],size)
        logz_true = np.random.uniform(logzlim[0],logzlim[1],size)
        z_true = np.random.uniform(zlim[0],zlim[1],size)
    else:
        lenchoice = size*5
        th = np.linspace(0.0,2.0*np.pi,lenchoice)
        p1, p2 = 1.5+np.sin(th), 1.5+np.cos(th)
        p1/=sum(p1); p2/=sum(p2)
        logM_true = np.random.choice(np.linspace(mlim[0],mlim[1],lenchoice),size=size,p=p1)
        ssfr_true = np.random.choice(np.linspace(ssfrlim[0],ssfrlim[1],lenchoice),size=size,p=p2)
        logz_true = np.random.choice(np.linspace(logzlim[0],logzlim[1],lenchoice),size=size,p=p1)
        z_true = np.random.choice(np.linspace(zlim[0],zlim[1],lenchoice),size=size,p=p2)

    logM_true-=np.median(logM_true); ssfr_true-=np.median(ssfr_true)
    logz_true-=np.median(logz_true); z_true-=np.median(z_true)
    indep_true = np.array([logM_true,ssfr_true,logz_true,z_true])
    # Determining true parameters
    width_true = 0.1
    limlist = [mlim,ssfrlim,logzlim,zlim]
    x = np.empty((0,degree+1)) # Two set up grid on which dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    # aatinv = np.linalg.inv(np.dot(a,a.T)) #The LS solution is with a.T, not a itself!
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    # ngrid_true = np.linspace(nlim[0]+0.01,nlim[-1]-0.01,xx[0].size) 
    ngrid_true = 1.38*np.random.rand(xx[0].size)-0.99 #True values of dust parameter at the grid
    coefs_true = np.linalg.lstsq(a_poly_T,ngrid_true,rc)[0] #Calculation of polynomial coefficients from the grid
    n_true = calc_poly(indep_true,coefs_true,degree) + width_true * np.random.randn(size) #True dust index parameter n is polynomial function of n with small natural width
    dict_true = {'ngrid':ngrid_true,'log_width':np.log(width_true)} # Dictionary of true values for easy computation of errors later

    # Observed values of the galaxy parameters
    sigM, sigS, siglogZ, sigz, sigN = tuple(np.array([0.4, 0.6, 0.4, 0.03, 0.2])*errmult) #Errors in "data" sample
    # First we need to "move" the mean of the posterior, then we generate posterior samples
    logM_samp = logM_true[:, None] + sigM * np.random.randn(size, 1) + sigM * np.random.randn(size, samples)
    ssfr_samp = ssfr_true[:, None] + sigS * np.random.randn(size, 1) + sigS * np.random.randn(size, samples)
    logz_samp = logz_true[:, None] + siglogZ * np.random.randn(size, 1) + siglogZ * np.random.randn(size, samples)
    z_samp = z_true[:, None] + sigz * np.random.randn(size, 1) + sigz * np.random.randn(size, samples)
    indep_samp = np.array([logM_samp,ssfr_samp,logz_samp,z_samp])
    n_samp = n_true[:, None] + sigN * np.random.randn(size, 1) + sigN * np.random.randn(size, samples)

    # Plot true data in a manner similar to Salim+18
    plot2DDustMean(logM_true,ssfr_true,n_true,r'$\log M_*$','logM',r'$\log$ sSFR$_{\rm{100}}$','ssfr100','n','n',levels=3,extratext=extratext,binx=3,biny=3)

    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,2)
    
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        ngrid = pm.Uniform("ngrid",lower=-1.001,upper=0.4001,shape=ngrid_true.size,testval=1.2*np.random.rand(ngrid_true.size)-0.9)
        log_width = pm.Bound(pm.StudentT,lower=-5.0)("log_width", nu=5, mu=np.log(width_true), lam=0.5, testval=0.0)

        # Compute the expected n at each sample
        # coefs = tt.nlinalg.lstsq()(a_poly_T,ngrid[:,None],rc)[0][:,0]
        # coefs = tt.dot(aatinv,tt.dot(a,ngrid))
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
            plt.savefig(op.join('DataSim',"polyevery%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
        print(az.summary(trace,round_to=2))

        all_err = np.median(trace.ngrid,axis=0)-ngrid_true
        all_std = np.std(trace.ngrid,axis=0)
    return ngrid_true, all_err, all_std

def polyeverycoef(size=100,samples=50,plot=False,extratext='',mlim=np.array([8.0,12.0]),ssfrlim=np.array([-12.0,-8.0]),logzlim=np.array([-1.2,0.2]),zlim=np.array([0.5,3.0]),degree=2,order=5,sampling=1000,tune=1000,errmult=1.0):
    ''' Simulation of data where the dust index n is a 4-D polynomial function of stellar mass, sSFR, log(Z), and redshift.

    Parameters
    ----------
    size: Int
        Number of "galaxies" in the sample
    samples: Int
        Number of "posterior samples" available for each galaxy
    plot: Bool
        Whether or not an arviz plot should be made
    extratext: Str
        An addition to the plot name to distinguish it from previous runs
    mlim, ssfrlim, logzlim, zlim, nlim: Two-element lists
        Lower and upper limits for all parameters in question
    degree: Int
        Polynomial degree in each variable
    order: Int
        Order of the polynomial; terms with higher combined degrees will be truncated
    sampling: Int
        Number of draws for sampling (not including burn-in steps)
    tune: Int
        Number of burn-in steps for each sampling chain
    errmult: Float
        Multiplicative factor that determines how large the simulated data errors are

    Returns
    -------
    all_err: 1-D Numpy Array
        Errors in each fitted parameter (maximum likelihood sample vs true value)
    
    The function runs the pymc3 sampler on this toy example and prints the results; an arviz plot of the results is also made if plot==True
    '''
    # For getting errors between max likelihood solutions and true parameter values

    # Creating independent-variable arrays
    logM_true = np.random.uniform(mlim[0],mlim[1],size)
    ssfr_true = np.random.uniform(ssfrlim[0],ssfrlim[1],size)
    logz_true = np.random.uniform(logzlim[0],logzlim[1],size)
    z_true = np.random.uniform(zlim[0],zlim[1],size)

    # lenchoice = size*5
    # th = np.linspace(0.0,2.0*np.pi,lenchoice)
    # p1, p2 = 1.5+np.sin(th), 1.5+np.cos(th)
    # p1/=sum(p1); p2/=sum(p2)
    # logM_true = np.random.choice(np.linspace(mlim[0],mlim[1],lenchoice),size=size,p=p1)
    # ssfr_true = np.random.choice(np.linspace(ssfrlim[0],ssfrlim[1],lenchoice),size=size,p=p2)
    # logz_true = np.random.choice(np.linspace(logzlim[0],logzlim[1],lenchoice),size=size,p=p1)
    # z_true = np.random.choice(np.linspace(zlim[0],zlim[1],lenchoice),size=size,p=p2)

    # logM_true-=np.median(logM_true); ssfr_true-=np.median(ssfr_true)
    # logz_true-=np.median(logz_true); z_true-=np.median(z_true)
    indep_true = np.array([logM_true,ssfr_true,logz_true,z_true])
    # Determining true parameters
    width_true = 0.1
    sh = tuple(np.repeat(degree+1,len(indep_true)))
    order_arr = np.zeros((degree+1)**len(indep_true),dtype=int)
    for index, i in enumerate(np.ndindex(sh)):
        order_arr[index] = sum(i)
    inddontincl = np.where(order_arr>order)[0]
    coefs_true = np.random.randn(len(order_arr))
    coefs_true[inddontincl]=0.0
    n_true = calc_poly(indep_true,coefs_true,degree) + width_true * np.random.randn(size) #True dust index parameter n is polynomial function of n with small natural width

    # Observed values of the galaxy parameters
    sigM, sigS, siglogZ, sigz, sigN = tuple(np.array([0.4, 0.6, 0.4, 0.03, 0.2])*errmult) #Errors in "data" sample
    # First we need to "move" the mean of the posterior, then we generate posterior samples
    logM_samp = logM_true[:, None] + sigM * np.random.randn(size, 1) + sigM * np.random.randn(size, samples)
    ssfr_samp = ssfr_true[:, None] + sigS * np.random.randn(size, 1) + sigS * np.random.randn(size, samples)
    logz_samp = logz_true[:, None] + siglogZ * np.random.randn(size, 1) + siglogZ * np.random.randn(size, samples)
    z_samp = z_true[:, None] + sigz * np.random.randn(size, 1) + sigz * np.random.randn(size, samples)
    indep_samp = np.array([logM_samp,ssfr_samp,logz_samp,z_samp])
    n_samp = n_true[:, None] + sigN * np.random.randn(size, 1) + sigN * np.random.randn(size, samples)

    # Plot true data in a manner similar to Salim+18
    plot2DDustMean(logM_true,ssfr_true,n_true,r'$\log M_*$','logM',r'$\log$ sSFR$_{\rm{100}}$','ssfr100','n','n',levels=3,extratext=extratext,binx=3,biny=3)

    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,2)
    sig = np.ones(len(coefs_true))
    sig[inddontincl]=0.0001
    testval = np.zeros(len(coefs_true))
    testval[0]=-0.2
    
    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        coefs = pm.Normal("coefs",mu=0.0,sigma=sig,shape=len(sig),testval=testval)
        log_width = pm.Bound(pm.StudentT,lower=-5.0)("log_width", nu=5, mu=np.log(width_true), lam=0.5, testval=0.0)

        # Compute the expected n at each sample
        # coefs = tt.nlinalg.lstsq()(a_poly_T,ngrid[:,None],rc)[0][:,0]
        # coefs = tt.dot(aatinv,tt.dot(a,ngrid))
        # coefs = tt.dot(aTinv,ngrid)
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
            plt.savefig(op.join('DataSim',"polyeverycoef%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
        print(az.summary(trace,round_to=2))

    return trace, coefs_true, n_true, indep_true, width_true

def polydata(logMsamp,ssfrsamp,logzsamp,z,nsamp,plot=True,extratext='',sampling=1000,tune=1000,degree=2,sigz=0.008):
    size = len(logMsamp)
    samples = len(logMsamp[0])
    zsamp = z[:,None] + sigz*np.random.randn(size,samples)
    indep_samp = np.array([logMsamp,ssfrsamp,logzsamp,zsamp])
    x = np.empty((0,degree+1)) # Two set up grid on which dust parameter n will be defined
    limlist = []
    for samp in indep_samp:
        lim = np.percentile(samp,[16.0,84.0])
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
        limlist.append(lim)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    # 3-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep_samp,degree)

    # Pymc3 model creation
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and log_width (true width of relation)
        ngrid = pm.Uniform("ngrid",lower=-1.001,upper=0.4001,shape=a_poly_T[0].size,testval=1.2*np.random.rand(a_poly_T[0].size)-0.9)
        log_width = pm.Bound(pm.StudentT,lower=-5.0)("log_width", nu=5, mu=np.log(0.25), lam=0.5, testval=0.0)

        # Compute the expected n at each sample
        coefs = tt.dot(aTinv,ngrid)
        mu = tt.tensordot(coefs,term,axes=1)

        # The line has some width: we're calling it a Gaussian in n
        logp = -0.5 * (nsamp - mu) ** 2 * pm.math.exp(-2*log_width) - log_width

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        #Perform the sampling!
        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=False)

        # Use the arviz module to take a look at the results
        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join('DataTrue',"polydata%s_trace.png"%(extratext)),bbox_inches='tight',dpi=300)
        print(az.summary(trace,round_to=2))

    return trace, x, indep_samp

def main():
    # numgal = 1000
    # with (open("3dhst_resample_50.pickle",'rb')) as openfile:
    #     obj = pickle.load(openfile)
    # logM, ssfr = np.log10(obj['stellar_mass']), np.log10(obj['ssfr_100'])
    # logZ = obj['log_z_zsun']
    # logMavg = np.average(logM,axis=1); logssfravg = np.average(ssfr,axis=1)
    # logZavg = np.average(logZ,axis=1)
    # z = obj['z']
    # n = obj['dust_index']
    # masscomp = mass_completeness(z)
    # cond = np.logical_and.reduce((logMavg>=masscomp,z<3.0,logssfravg>-14.0))
    # ind = np.where(cond)[0]
    # indfin = np.random.choice(ind,size=numgal,replace=False)
    # n_sim = quad_every_data(logM[ind],ssfr[ind],obj['log_z_zsun'][ind],z[ind])
    extratext='_coef_uniform_500_lowerr'
    degree = 2; order=5
    # trace, x, indep_samp = polydata(logM[indfin],ssfr[indfin],logZ[indfin],z[indfin],n[indfin],extratext=extratext)
    # np.savetxt(op.join("DataTrue","Polydata%s_indfin.dat"%(extratext)),indfin,fmt='%d',header='Indfin')

    # ngrid_best = trace.ngrid[np.argmax(trace.model_logp)]
    # med_ngrid, std_ngrid = np.median(trace.ngrid,axis=0), np.std(trace.ngrid,axis=0)
    # np.savetxt(op.join("DataTrue","Polydata%s_ngrid.dat"%(extratext)),np.column_stack((ngrid_best,med_ngrid,std_ngrid)),fmt='%.4f',header='Best_n_grid  Median_n_grid  StD_n_grid')
    # np.savetxt(op.join("DataTrue","Polydata%s_x.dat"%(extratext)),x,fmt='%.4f')
    # xx = np.meshgrid(*x)
    # coefs = polyfitnd(xx,med_ngrid)
    # coefs2 = polyfitnd(xx,ngrid_best)
    # nsamp_fit = calc_poly(indep_samp,coefs,2)
    # nsamp_fit2 = calc_poly(indep_samp,coefs2,2)
    # plot2DDustMean(logMavg[indfin],logssfravg[indfin],np.average(nsamp_fit,axis=1),r'$\log M_*$','logM',r'$\log$ sSFR$_{\rm{100}}$','ssfr100','n','n',levels=10,extratext=extratext,binx=10,biny=10)
    # plot2DDustMean(z[indfin],logZavg[indfin],np.average(nsamp_fit,axis=1),'z','z',r'$\log\ (Z/Z_\odot)$','logZ','n','n',levels=10,extratext=extratext,binx=10,biny=10)

    # navg = np.average(n[indfin],axis=1)
    # navg_samp = np.average(nsamp_fit,axis=1)
    # navg_samp2 = np.average(nsamp_fit2,axis=1)
    # plt.figure()
    # plt.plot(navg,navg_samp-navg,'b.')
    # plt.xlabel("Data average n")
    # plt.ylabel("Sim average n - Data average n")
    # plt.savefig(op.join("DataTrue","Polydata%s_n_err.png"%(extratext)),bbox_inches='tight',dpi=150)

    # plt.figure()
    # plt.plot(navg,navg_samp2-navg,'b.')
    # plt.xlabel("Data average n")
    # plt.ylabel("Sim max like n - Data average n")
    # plt.savefig(op.join("DataTrue","Polydata%s_n_err_max_lik.png"%(extratext)),bbox_inches='tight',dpi=150)

    trace, coefs_true, n_true, indep_true, width_true = polyeverycoef(size=500,errmult=0.05,plot=True,extratext=extratext,degree=degree,order=order)

    # ngrid_true, all_err, all_std = polyevery(size=500,tune=1000,errmult=1.0,plot=True,extratext=extratext)
    # print(all_err,all_std)
    # np.savetxt(op.join("DataSim","Polyevery%s.dat"%(extratext)),np.column_stack((ngrid_true,all_err,all_std)),fmt='%.4f',header='True  Error  Std')

    all_err = np.median(trace.coefs,axis=0)-coefs_true
    all_std = np.median(trace.coefs,axis=0)
    # coef_best = trace.coefs[np.argmax(trace.model_logp)]
    numsamp = 50
    inds = np.random.choice(len(trace.model_logp),size=numsamp,replace=True)
    n_sim = np.zeros((numsamp,len(n_true)))
    for i, ind in enumerate(inds):
        n_sim[i] = calc_poly(indep_true,trace.coefs[ind],degree)
    med_n_sim = np.median(n_sim,axis=0)
    std_n_sim = np.std(n_sim,axis=0)

    plt.figure()
    plt.errorbar(coefs_true,all_err,yerr=all_std,fmt='b.',capsize=2)
    plt.xlabel("True coefs")
    plt.ylabel("Median sim coefs - True coefs")
    plt.savefig(op.join("DataSim","Polyeverycoef_coefs%s.png"%(extratext)),bbox_inches='tight',dpi=150)

    plt.figure()
    plt.errorbar(n_true,med_n_sim-n_true,yerr=std_n_sim,xerr=width_true,fmt='b.',capsize=2)
    plt.xlabel("True n")
    plt.ylabel("Median sim n - True n")
    plt.savefig(op.join("DataSim","Polyeverycoef_n%s.png"%(extratext)),bbox_inches='tight',dpi=150)

    # sizes = [20,100,500]
    # samples = [10,50,250]
    # stepsarr = [200,1000,5000]
    # tunearr = [200,1000,5000]
    # errmults = [0.2,1.0,5.0]
    # n = len(sizes)
    # err, gr, effn_norm, mcse, energy_err, tottime = np.zeros(n*5), np.zeros(n*5), np.zeros(n*5), np.zeros(n*5), np.zeros(n*5), np.zeros(n*5)
    # si, sa, st, tu, er= sizes[1]*np.ones(n*5,dtype=int), samples[1]*np.ones(n*5,dtype=int), stepsarr[1]*np.ones(n*5,dtype=int), tunearr[1]*np.ones(n*5,dtype=int), errmults[1]*np.ones(n*5,dtype=int)

    # for i in range(n):
    #     err[i], gr[i], effn_norm[i], mcse[i], energy_err[i], tottime[i] = lin_with_samples(size=sizes[i])
    #     err[i+n], gr[i+n], effn_norm[i+n], mcse[i+n], energy_err[i+n], tottime[i+n] = lin_with_samples(samples=samples[i])
    #     err[i+2*n], gr[i+2*n], effn_norm[i+2*n], mcse[i+2*n], energy_err[i+2*n], tottime[i+2*n] = lin_with_samples(steps=stepsarr[i])
    #     err[i+3*n], gr[i+3*n], effn_norm[i+3*n], mcse[i+3*n], energy_err[i+3*n], tottime[i+3*n] = lin_with_samples(tune=tunearr[i])
    #     err[i+4*n], gr[i+4*n], effn_norm[i+4*n], mcse[i+4*n], energy_err[i+4*n], tottime[i+4*n] = lin_with_samples(errmult=errmults[i])

    #     si[i]=sizes[i]
    #     sa[i+n]=samples[i]
    #     st[i+2*n]=stepsarr[i]
    #     tu[i+3*n]=tunearr[i]
    #     er[i+4*n]=errmults[i]

    # np.savetxt("lin_with_samples_stats.dat",np.column_stack((si,sa,st,tu,er,err,gr,effn_norm,mcse,energy_err,tottime)),fmt=['%d']*4+['%.5f']*7,header='Sizes  Samples  Steps  Tuning  Err_Mult  Median_Err  Median_GR_Stat  Median_Eff_n_Norm  Median_MCSE  Median_Energy_Err  Tot_Time')

if __name__=='__main__':
    main()