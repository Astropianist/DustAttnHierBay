import numpy as np 
import pymc3 as pm 
import theano.tensor as tt
import arviz as az
import pickle
import matplotlib.pyplot as plt 
import os.path as op
from distutils.dir_util import mkpath
from anchor_points import calc_poly, polyfitnd, calc_poly_tt, get_a_polynd
from DustPymc3_Final import make_prop_dict, plot_model_true, plot_color_map, plot1D, mass_completeness
import argparse as ap
from lmfit import Model
from copy import deepcopy
from astropy.table import Table
import seaborn as sns 
sns.set_context("paper") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

def parse_args(argv=None):
    """ Tool to parse arguments from the command line. The entries should be self-explanatory """
    parser = ap.ArgumentParser(description="DustPymc3",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-si','--size',help='Number of mock galaxies',type=int,default=500)
    parser.add_argument('-pl','--plot',help='Whether or not to plot ArViz plot',action='count',default=0)
    parser.add_argument('-tu','--tune',help='Number of tuning steps',type=int,default=1000)
    parser.add_argument('-st','--steps',help='Number of desired steps included per chain',type=int,default=1000)
    parser.add_argument('-ext','--extratext',help='Extra text to help distinguish particular run',type=str,default='')
    parser.add_argument('-dg','--degree',help='Degree of polynomial (true)',type=int,default=2)
    parser.add_argument('-dir','--dir_orig',help='Parent directory for files',type=str,default='Simple')
    parser.add_argument('-es','--err_samp',help='Number of samples for lmfit error analysis',type=int,default=20)
    parser.add_argument('-em','--err_mult',help='Number of samples for lmfit error analysis',type=float,default=0.05)

    parser.add_argument('-m','--logM',help='Whether or not to include stellar mass (true)',action='count',default=0)
    parser.add_argument('-s','--ssfr',help='Whether or not to include sSFR (true)',action='count',default=0)
    parser.add_argument('-logZ','--logZ',help='Whether or not to include metallicity (true)',action='count',default=0)
    parser.add_argument('-z','--z',help='Whether or not to include redshift (true)',action='count',default=0)
    parser.add_argument('-i','--i',help='Whether or not to include inclination in model',action='count',default=0)
    parser.add_argument('-d1','--d1',help='Whether or not to include birth cloud dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-d2','--d2',help='Whether or not to include diffuse dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-n','--n',help='Whether or not to use dust index as dependent variable',action='count',default=0)

    return parser.parse_args(args=argv)

def label_creation(args,prop_dict,dep_dict):
    indep_name, indep_lab = [], []
    for name in prop_dict['names'].keys():
        if getattr(args,name):
            indep_name.append(prop_dict['names'][name])
            indep_lab.append(prop_dict['labels'][name])
    if args.n: 
        dep_name, dep_lab, nlim = dep_dict['names']['n'], dep_dict['labels']['n'], np.array([-1.0,0.4])
    else:
        dep_name, dep_lab, nlim = dep_dict['names']['tau2'], dep_dict['labels']['tau2'], np.array([0.0,4.0])
    return indep_name, indep_lab, dep_name, dep_lab, nlim

def plot2DDustMean(x,y,z,xplot,xname,yplot,yname,zplot,zname,binx=50,biny=50,extratext='',levels=10,min_bin=1):
    vmin=min(z)-0.001*abs(min(z))
    xdiv = np.linspace(min(x),max(x)+1.0e-8,binx+1)
    ydiv = np.linspace(min(y),max(y)+1.0e-8,biny+1)
    xavg = np.linspace((xdiv[0]+xdiv[1])/2.0,(xdiv[-1]+xdiv[-2])/2.0,binx)
    yavg = np.linspace((ydiv[0]+ydiv[1])/2.0,(ydiv[-1]+ydiv[-2])/2.0,biny)
    zmean = np.empty((biny,binx)); znum = np.empty((biny,binx))
    zmean.fill(np.nan); znum.fill(np.nan)
    # zmean = -99.0*np.ones((biny,binx))
    for j in range(biny):
        condj = np.logical_and(y>=ydiv[j],y<ydiv[j+1])
        for i in range(binx):
            condi = np.logical_and(x>=xdiv[i],x<xdiv[i+1])
            cond = np.logical_and(condi,condj)
            if len(z[cond])>=min_bin:
                # print(len(x[condi]),len(y[condj]),len(z[cond]))
                zmean[j,i] = np.median(z[cond])
                znum[j,i] = len(z[cond])
    # print(zmean[-1,:])
    # print("Min of zmean:"); print(np.amin(zmean))
    plt.figure()
    xx, yy = np.meshgrid(xavg,yavg)
    # print(xx.shape,len(xx[zmean>=vmin]))
    cf = plt.contourf(xx,yy,zmean,levels=levels,cmap='viridis',vmin=vmin)
    cnum = plt.contour(xx,yy,znum,levels=4,cmap='Greys')
    # cf = plt.pcolormesh(xdiv,ydiv,zmean,cmap=my_cmap,vmin=vmin)
    minx,maxx = min(xx[zmean>=vmin]),max(xx[zmean>=vmin])
    miny,maxy = min(yy[zmean>=vmin]),max(yy[zmean>=vmin])
    # print(np.amin(yy),np.amax(yy))
    # print(minx,maxx,miny,maxy)
    plt.gca().set_xlim([minx,maxx])
    plt.gca().set_ylim([miny,maxy])
    plt.xlabel(xplot)
    plt.ylabel(yplot)
    plt.colorbar(cf,label=zplot)
    plt.colorbar(cnum,label='Number of Galaxies')
    pathname = op.join("DataTrue","Simple",extratext,str(min_bin))
    mkpath(pathname)
    plt.savefig(op.join(pathname,"SalimStyle_%s_%svs%s%s_binxy%d_minbin%d.png"%(zname,yname,xname,extratext,binx,min_bin)),bbox_inches='tight',dpi=300)

def getData():
    obj = pickle.load(open("3dhst_samples.pickle",'rb'))
    logM, ssfr, logZ, z, tau1, tau2, n = np.log10(obj['stellar_mass'])[:,0], np.log10(obj['ssfr_100'])[:,0], obj['log_z_zsun'][:,0], obj['z'], obj['dust1'][:,0], obj['dust2'][:,0], obj['dust_index'][:,0]
    inc, dq = np.loadtxt('GalfitParamsProsp.dat',usecols=(2,3),unpack=True)
    logMe = np.std(np.log10(obj['stellar_mass']),axis=1)
    ssfre = np.std(np.log10(obj['ssfr_100']),axis=1)
    logZe, ze = np.std(obj['log_z_zsun'],axis=1), 0.001*np.ones(len(z))
    tau1e, tau2e = np.std(obj['dust1'],axis=1), np.std(obj['dust2'],axis=1)
    ne = np.std(obj['dust_index'],axis=1)
    # comb = np.array([ssfr,logZ])
    # per = np.percentile(comb,[1.0,99.0],axis=1)
    masscomp = mass_completeness(z)
    cond = np.logical_and.reduce((logM>=masscomp,ssfr>=-12.5))
    ind = np.where(cond)[0]
    return logM[ind], ssfr[ind], logZ[ind], z[ind], tau1[ind], tau2[ind], n[ind], inc[ind], logMe[ind], ssfre[ind], logZe[ind], ze[ind], tau1e[ind], tau2e[ind], ne[ind], dq[ind]

def getNecessaryData(args, logM=None, ssfr=None, logZ=None, z=None, tau1=None, tau2=None, n=None, inc=None, logMe=None, ssfre=None, logZe=None, ze=None, tau1e=None, tau2e=None, ne=None, dq=None):
    if logM is None: logM, ssfr, logZ, z, tau1, tau2, n, inc, logMe, ssfre, logZe, ze, tau1e, tau2e, ne, dq = getData()
    prop_dict, dep_dict = make_prop_dict()
    if args.i: cond = inc>=0
    else: cond = logM>=0
    ind = np.where(cond)[0]
    if args.size==-1: args.size = len(ind)
    indfin = np.random.choice(ind,size=args.size,replace=False)
    indep = np.empty((0,len(logM[indfin])))
    if logMe is not None: indep_err = np.empty((0,len(logM[indfin])))
    else: indep_err = None
    med_mod = np.array([])
    if args.logM: 
        md = np.median(logM[indfin])
        indep = np.append(indep,logM[indfin][None,:]-md,axis=0)
        if logMe is not None: indep_err = np.append(indep_err,logMe[indfin][None,:],axis=0)
        med_mod = np.append(med_mod,md)
    if args.ssfr: 
        md = np.median(ssfr[indfin])
        indep = np.append(indep,ssfr[indfin][None,:]-md,axis=0)
        if logMe is not None: indep_err = np.append(indep_err,ssfre[indfin][None,:],axis=0)
        med_mod = np.append(med_mod,md)
    if args.logZ: 
        md = np.median(logZ[indfin])
        indep = np.append(indep,logZ[indfin][None,:]-md,axis=0)
        if logMe is not None: indep_err = np.append(indep_err,logZe[indfin][None,:],axis=0)
        med_mod = np.append(med_mod,md)
    if args.z: 
        md = np.median(z[indfin])
        indep = np.append(indep,z[indfin][None,:]-md,axis=0)
        if logMe is not None: indep_err = np.append(indep_err,ze[indfin][None,:],axis=0)
        med_mod = np.append(med_mod,md)
    if args.i:
        md = np.median(inc[indfin])
        indep = np.append(indep,inc[indfin][None,:]-md,axis=0)
        if logMe is not None: indep_err = np.append(indep_err,dq[indfin][None,:],axis=0)
        med_mod = np.append(med_mod,md)
    if args.d1:
        md = np.median(tau1[indfin])
        indep = np.append(indep,tau1[indfin][None,:]-md,axis=0)
        if logMe is not None: indep_err = np.append(indep_err,tau1e[indfin][None,:],axis=0)
        med_mod = np.append(med_mod,md)
    if args.d2:
        md = np.median(tau2[indfin])
        indep = np.append(indep,tau2[indfin][None,:]-md,axis=0)
        if logMe is not None: indep_err = np.append(indep_err,tau2e[indfin][None,:],axis=0)
        med_mod = np.append(med_mod,md)
    if args.n: 
        dep = n[indfin]
        if logMe is not None: dep_err = ne[indfin]
        else: dep_err = None
        uniform = True
    else: 
        dep = tau2[indfin]
        if logMe is not None: dep_err = tau2e[indfin]
        else: dep_err = None
        uniform = False

    indep_name, indep_lab, dep_name, dep_lab, nlim = label_creation(args,prop_dict,dep_dict)

    return indep, indep_err, dep, dep_err, med_mod, z[indfin], uniform, indep_name, indep_lab, dep_name, dep_lab, nlim

def polymod(**kwargs):
    xx = kwargs['xx']
    # del kwargs['xx']
    indlist = [st for st in kwargs if 'ind' in st]
    shlist = list(kwargs[indlist[0]].shape)
    shlist2 = deepcopy(shlist)
    shlist.insert(0,0); shlist2.insert(0,1); sh2 = tuple(shlist2)
    ind_var = np.empty(tuple(shlist))
    for ind in indlist:
        ind_var = np.append(ind_var,kwargs[ind].reshape(sh2),axis=0)
        # del kwargs[ind]
    deg = len(xx[0])-1
    numpts = (deg+1)**len(xx)
    depval = np.array([kwargs['c'+str(i)] for i in range(numpts)])
    # depval = np.array(list(kwargs.values()))
    coefs = polyfitnd(xx,depval)
    
    return calc_poly(ind_var,coefs,deg)

def make_lmfit(indep,dep,names,degree=2,mindep=-1.0,maxdep=0.4):
    med_dep = np.median(dep)
    numpts = (degree+1)**len(indep)
    names_new = [st+'_ind' for st in names]
    x = np.empty((0,degree+1)) # To set up grid on which true dust parameter n will be defined
    per = np.percentile(indep,[1.0,99.0],axis=1)
    for i in range(len(indep)):
        x = np.append(x,np.linspace(per[0,i],per[1,i],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    pmod = Model(polymod,independent_vars=names_new,param_names=['c'+str(d) for d in range(numpts)])
    pars = pmod.make_params()
    for i in range(numpts):
        pars.add('c'+str(i),value=med_dep,min=mindep,max=maxdep,vary=True)
    kwargs = {'xx':xx}
    for name, ind in zip(names_new, indep):
        kwargs[name] = ind
    res = pmod.fit(dep,params=pars,**kwargs)
    # print(res.fit_report())
    return xx, res

def pymc3_simple(indep,dep,img_dir_orig,degree=2,mindep=-1.0,maxdep=0.4,sampling=1000,tune=1000,uniform=True,extratext='',plot=True):
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)
    mkpath(img_dir)
    ndim = len(indep)
    limlist = []
    for indepi in indep:
        per = np.percentile(indepi,[1.0,99.0])
        limlist.append(per)
    lower, upper = min(mindep,np.amin(dep)), max(maxdep,np.amax(dep)) # Limits for dependent variable
    x = np.empty((0,degree+1)) # To set up grid on which true dust parameter n will be defined
    for lim in limlist:
        x = np.append(x,np.linspace(lim[0],lim[-1],degree+1)[None,:],axis=0)
    xx = np.meshgrid(*x) #N-D Grid for polynomial computations
    a_poly_T = get_a_polynd(xx).T #Array related to grid that will be used in least-squares computation
    aTinv = np.linalg.inv(a_poly_T)
    rc = -1.0 #Rcond parameter set to -1 for keeping all entries of result to machine precision, regardless of rank issues
    # 2-D array that will be multiplied by coefficients to calculate the dust parameter at the observed independent variable values
    term = calc_poly_tt(indep,degree)
    # breakpoint()
    with pm.Model() as model:
        # Priors on the parameters ngrid (n over the grid) and sigma (related to width of relation)
        if uniform: ngrid = pm.Uniform("ngrid",lower=lower-1.0e-5,upper=upper+1.0e-5,shape=xx[0].size,testval=np.random.uniform(lower,upper,xx[0].size))
        else: ngrid = pm.TruncatedNormal("ngrid",mu=0.3,sigma=1.0,lower=lower-1.0e-5,upper=upper+1.0e-5,shape=xx[0].size,testval=np.random.uniform(lower,upper/2.0,xx[0].size))
        sigma = pm.HalfNormal("sigma", sigma=1)
    
        # Compute the expected n at each sample
        coefs = tt.dot(aTinv,ngrid)
        mu = tt.tensordot(coefs,term,axes=1)

        # Likelihood (sampling distribution) of observations
        dep_obs = pm.Normal("dep_obs", mu=mu, sigma=sigma, observed=dep)

        map_estimate = pm.find_MAP()
        print(map_estimate)

        trace = pm.sample(draws=sampling, tune=tune, init='adapt_full', target_accept=0.9, return_inferencedata=True)

        if plot:
            az.plot_trace(trace)
            plt.savefig(op.join(img_dir,"polyND%s_trace_pm_simp.png"%(extratext)),bbox_inches='tight',dpi=300)
        print(az.summary(trace,round_to=2))

    return trace, xx, map_estimate

def plot_lmfit(res,xx,indep,indep_err,dep,dep_err,indep_name,indep_lab,dep_name,dep_lab,med_arr,img_dir_orig=op.join('Simple','lmfit'),extratext='',fine_grid=201,bins=25,levels=15,mindep=-1.0,maxdep=0.4):
    # Extract initial information about best-fit model at original data points
    degree = len(xx[0])-1
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)
    mkpath(img_dir)
    ndim = len(indep)
    kwargs = res.best_values
    ngrid_best = np.array(list(kwargs.values()))
    kwargs['xx'] = xx
    names_new = [st+'_ind' for st in indep_name]
    for name, ind in zip(names_new, indep):
        kwargs[name] = ind
    
    # Save best-fit model to a data file
    dataarr = np.empty((ngrid_best.size,0))
    colnames = []
    for i in range(len(xx)):
        dataarr = np.append(dataarr,xx[i].reshape(ngrid_best.size,1),axis=1)
        dataarr = np.append(dataarr,xx[i].reshape(ngrid_best.size,1)+med_arr[i],axis=1)
        colnames.append(indep_name[i]); colnames.append(indep_name[i]+'_plus_med')
    dataarr = np.append(dataarr,ngrid_best[:,None],axis=1); colnames.append('ngrid')
    # header = '%s  ngrid'%('  '.join(indep_name))
    dataname = ''
    for name in indep_name: dataname += name+'_'
    # np.savetxt(op.join(img_dir,'ngrid_%s%s_%s_lmfit.dat'%(dataname,dep_name,extratext)),dataarr,header=header,fmt='%.5f')
    t = Table(dataarr,names=colnames)
    t.write(op.join(img_dir,'ngrid_%s%s_%s_lmfit.dat'%(dataname,dep_name,extratext)),overwrite=True,format='ascii')

    # Plot stuff!
    plot_model_true(dep,res.best_fit,None,None,img_dir,'Real_n_comp_%s_lmfit'%(extratext),n_err=res.eval_uncertainty(),width_true=dep_err,ngrid_true_plot=False,ylab='LMFIT model %s'%(dep_lab),xlab='Prospector Max L %s'%(dep_lab))

    if ndim==1:
        x = np.linspace(np.amin(xx[0]),np.amax(xx[0]),fine_grid)
        kwargs[names_new[0]]=x
        y, yerr = res.eval(**kwargs), res.eval_uncertainty(**kwargs)
        plot1D(x+med_arr[0],y,xx[0].ravel()+med_arr[0],ngrid_best,img_dir,'Real_Model_%s_%s_%s_lmfit'%(indep_name[0],dep_name,extratext),xlab=indep_lab[0],ylab='LMFIT model '+dep_lab,yerr=yerr)
        return

    med_mod = np.median(indep,axis=1)
    indep_div, indep_avg = np.ones((2,bins+1)), np.ones((2,bins))
    znum = np.zeros((bins,bins)) # Will fill with numbers of galaxies
    for i in range(ndim):
        namei = indep_name[i]
        for j in range(i+1,ndim):
            namej = indep_name[j]
            indep_fine = med_mod[:,None]*np.ones((ndim,fine_grid))
            # Define grids of the independent variables: a fine one (xx_fine) and a coarse one (xx_div)
            index = 0
            for k in [i,j]:
                indep_fine[k] = np.linspace(np.amin(xx[k]),np.amax(xx[k]),fine_grid)
                indep_div[index] = np.linspace(np.amin(xx[k]),np.amax(xx[k])+1.0e-8,bins+1)
                indep_avg[index] = np.linspace((indep_div[index][0]+indep_div[index][1])/2.0,(indep_div[index][-1]+indep_div[index][-2])/2.0,bins)
                index+=1

            xx_fine = np.meshgrid(*indep_fine[(i,j),:])
            xx_div = np.meshgrid(*indep_avg)

            for k in range(i-1,-1,-1):
                medval = indep_fine[k,0]
                xx_fine.insert(0,medval*np.ones_like(xx_fine[0]))
            for k in range(i+1,j):
                medval = indep_fine[k,0]
                xx_fine.insert(-1,medval*np.ones_like(xx_fine[0]))
            for k in range(j+1,ndim):
                medval = indep_fine[k,0]
                xx_fine.insert(k,medval*np.ones_like(xx_fine[0]))

            # Evaluate the model at the fine grid
            for name, xx_dim in zip(names_new, xx_fine):
                kwargs[name] = xx_dim.ravel()
            # Determine number of galaxies in each coarse bin square
            for l in range(bins):
                condj = np.logical_and(indep[j]>=indep_div[1][l],indep[j]<indep_div[1][l+1])
                for k in range(bins):
                    condi = np.logical_and(indep[i]>=indep_div[0][k],indep[i]<indep_div[0][k+1])
                    cond = np.logical_and(condi,condj)
                    znum[l,k] = len(dep[cond])
            dep_mod, dep_mod_err = res.eval(**kwargs), res.eval_uncertainty(**kwargs)

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],dep_mod.reshape(fine_grid,fine_grid),xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'MeanModel_%s_%s_%s%s_lmfit'%(dep_name,indep_name[i],indep_name[j],extratext),levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='LMFIT model %s'%(dep_lab),xtrue=indep[i]+med_arr[i],ytrue=indep[j]+med_arr[j],xtrueerr=indep_err[i],ytrueerr=indep_err[j],minz=mindep,maxz=maxdep)

def plot_pymc3(trace,xx,indep,indep_err,dep,dep_err,indep_name,indep_lab,dep_name,dep_lab,med_arr,img_dir_orig=op.join('Simple','lmfit'),extratext='',fine_grid=201,bins=25,levels=15,mindep=-1.0,maxdep=0.4,numsamp=50,grid_name='ngrid',width_name='sigma',map_estimate=None):
    # Extract initial information about best-fit model at original data points
    degree = len(xx[0])-1
    img_dir = op.join(img_dir_orig,'deg_%d'%(degree),extratext)
    mkpath(img_dir)
    nlen, ndim = len(dep), len(indep)

    # Determine median model values for the dependent variable
    ngrid_0, width_0 = np.array(getattr(trace.posterior,grid_name)), np.array(getattr(trace.posterior,width_name))
    sh = ngrid_0.shape
    ngrid, width = ngrid_0.reshape(sh[0]*sh[1],sh[2]), width_0.reshape(sh[0]*sh[1])
    ngrid_med = np.median(ngrid,axis=0)
    width_med = np.median(width)
    width_mean = np.mean(width)
    all_std = np.std(ngrid,axis=0)
    coefs_med = polyfitnd(xx,ngrid_med)
    n_med = calc_poly(indep,coefs_med,degree) + width_med * np.random.randn(nlen)

    # Save best-fit model to a data file
    dataarr = np.empty((ngrid_med.size,0))
    for i in range(len(xx)):
        dataarr = np.append(dataarr,xx[i].reshape(ngrid_med.size,1),axis=1)
    dataarr = np.append(dataarr,ngrid_med[:,None],axis=1)
    if map_estimate is not None: 
        dataarr = np.append(dataarr,map_estimate[grid_name][:,None],axis=1)
        header = '%s  ngrid  ngrid_MAP'%('  '.join(indep_name))
    else:
        header = '%s  ngrid'%('  '.join(indep_name))
    dataname = ''
    for name in indep_name: dataname += name+'_'
    np.savetxt(op.join(img_dir,'ngrid_%s%s_%s_pm_simp.dat'%(dataname,dep_name,extratext)),dataarr,header=header,fmt='%.5f')
    trace.to_netcdf(op.join(img_dir,'trace_%s%s_%s_pm_simp.nc'%(dataname,dep_name,extratext)))
    
    # Plot stuff!
    inds = np.random.choice(len(width),size=numsamp,replace=False)
    ngrid_mod_all, width_mod_all = np.empty((numsamp,coefs_med.size)), np.empty(numsamp)
    coefs_mod_all = np.empty((numsamp,coefs_med.size))
    n_sim = np.empty((numsamp,nlen))
    for i, ind in enumerate(inds):
        ngrid_mod = ngrid[ind]
        width_mod = width[ind]
        coefs_mod = polyfitnd(xx,ngrid_mod)
        ngrid_mod_all[i] = ngrid_mod
        width_mod_all[i] = width_mod
        coefs_mod_all[i] = coefs_mod
        n_sim[i] = calc_poly(indep,coefs_mod,degree) + width_mod * np.random.randn(*n_sim[i].shape)
    n_mean, n_err = np.mean(n_sim,axis=0), np.std(n_sim,axis=0)

    plot_model_true(dep,n_mean,None,None,img_dir,'Real_n_comp_%s_pm_simp'%(extratext),n_err=n_err,width_true=dep_err,ylab='Mean posterior model %s'%(dep_lab), xlab='Prospector Max L %s'%(dep_lab),ngrid_true_plot=False)

    if ndim==1:
        x = np.linspace(np.amin(xx[0]),np.amax(xx[0]),fine_grid)
        n_sim2 = np.empty((numsamp,fine_grid))
        for i, ind in enumerate(inds):
            n_sim2[i] = calc_poly(x[None,:],coefs_mod_all[i],degree)
        n_mean2, n_err2 = np.mean(n_sim2,axis=0), np.std(n_sim2,axis=0)
        plot1D(x+med_arr[0],n_mean2,xx[0].ravel()+med_arr[0],np.mean(ngrid_mod_all,axis=0),img_dir,'Real_Model_%s_%s_%s_pm_simp'%(indep_name[0],dep_name,extratext),xlab=indep_lab[0],ylab='Mean posterior model '+dep_lab,yerr=n_err2)
        return

    med_mod = np.median(indep,axis=1)
    indep_div, indep_avg = np.ones((2,bins+1)), np.ones((2,bins))
    znum = np.zeros((bins,bins)) # Will fill with numbers of galaxies
    for i in range(ndim):
        namei = indep_name[i]
        for j in range(i+1,ndim):
            namej = indep_name[j]
            indep_fine = med_mod[:,None]*np.ones((ndim,fine_grid))
            # Define grids of the independent variables: a fine one (xx_fine) and a coarse one (xx_div)
            index = 0
            for k in [i,j]:
                indep_fine[k] = np.linspace(np.amin(xx[k]),np.amax(xx[k]),fine_grid)
                indep_div[index] = np.linspace(np.amin(xx[k]),np.amax(xx[k])+1.0e-8,bins+1)
                indep_avg[index] = np.linspace((indep_div[index][0]+indep_div[index][1])/2.0,(indep_div[index][-1]+indep_div[index][-2])/2.0,bins)
                index+=1

            xx_fine = np.meshgrid(*indep_fine[(i,j),:])
            xx_div = np.meshgrid(*indep_avg)

            for k in range(i-1,-1,-1):
                medval = indep_fine[k,0]
                xx_fine.insert(0,medval*np.ones_like(xx_fine[0]))
            for k in range(i+1,j):
                medval = indep_fine[k,0]
                xx_fine.insert(-1,medval*np.ones_like(xx_fine[0]))
            for k in range(j+1,ndim):
                medval = indep_fine[k,0]
                xx_fine.insert(k,medval*np.ones_like(xx_fine[0]))

            # Evaluate the model at the fine grid
            n_sim2 = np.empty((numsamp,fine_grid,fine_grid))
            for ii, ind in enumerate(inds):
                n_sim2[ii] = calc_poly(xx_fine,coefs_mod_all[ii],degree) # + width_mod_all[ii] * np.random.randn(*n_sim2[ii].shape)
            n_mean2, n_err2 = np.mean(n_sim2,axis=0), np.std(n_sim2,axis=0)
            # Determine number of galaxies in each coarse bin square
            for l in range(bins):
                condj = np.logical_and(indep[j]>=indep_div[1][l],indep[j]<indep_div[1][l+1])
                for k in range(bins):
                    condi = np.logical_and(indep[i]>=indep_div[0][k],indep[i]<indep_div[0][k+1])
                    cond = np.logical_and(condi,condj)
                    znum[l,k] = len(dep[cond])

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_mean2,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'MeanModel_%s_%s_%s%s_pm_simp'%(dep_name,indep_name[i],indep_name[j],extratext),levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='Mean posterior model %s'%(dep_lab),xtrue=indep[i]+med_arr[i],ytrue=indep[j]+med_arr[j],xtrueerr=indep_err[i],ytrueerr=indep_err[j],minz=mindep,maxz=maxdep)

            plot_color_map(xx_fine[i]+med_arr[i],xx_fine[j]+med_arr[j],n_err2/width_mean,xx_div[0]+med_arr[i],xx_div[1]+med_arr[j],znum,xx[i].ravel()+med_arr[i],xx[j].ravel()+med_arr[j],img_dir,'ErrModel_%s_%s_%s%s_pm_simp'%(dep_name,indep_name[i],indep_name[j],extratext),levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='Model %s uncertainty / intrinsic width'%(dep_lab),xtrue=indep[i]+med_arr[i],ytrue=indep[j]+med_arr[j],xtrueerr=indep_err[i],ytrueerr=indep_err[j],minz=mindep,maxz=maxdep)

def lmfit_err_analysis(fine_grid=201,bins=25,levels=15):
    args = parse_args()
    size_orig = args.size
    img_dir_orig = op.join('DataTrue',args.dir_orig,'lmfit_err')
    obj = pickle.load(open("3dhst_resample_500_inc.pickle",'rb'))
    logMe = np.std(np.log10(obj['stellar_mass']),axis=1)
    ssfre = np.std(np.log10(obj['ssfr_100']),axis=1)
    logZe, ze = np.std(obj['log_z_zsun'],axis=1), 0.001*np.ones(len(logMe))
    tau1e, tau2e = np.std(obj['dust1'],axis=1), np.std(obj['dust2'],axis=1)
    ne, ince = np.std(obj['dust_index'],axis=1), np.std(obj['inc'],axis=1)
    rand_inds = np.random.choice(500,args.err_samp,replace=False)
    # indep_all, dep_all, best_fit_all, best_unc, xx_all = [], [], [], [], []
    coefs_all = []
    for i, ri in enumerate(rand_inds):
        logM, ssfr, logZ, z, tau1, tau2, n, inc = np.log10(obj['stellar_mass'])[:,ri], np.log10(obj['ssfr_100'])[:,ri], obj['log_z_zsun'][:,ri], obj['z'], obj['dust1'][:,ri], obj['dust2'][:,ri], obj['dust_index'][:,ri], obj['inc'][:,ri]
        masscomp = mass_completeness(z)
        cond = np.logical_and.reduce((logM>=masscomp,ssfr>=-12.5))
        ind = np.where(cond)[0]
        indep, indep_err, dep, dep_err, med_mod, z, uniform, indep_name, indep_lab, dep_name, dep_lab, nlim = getNecessaryData(args,logM[ind], ssfr[ind], logZ[ind], z[ind], tau1[ind], tau2[ind], n[ind], inc[ind], logMe[ind], ssfre[ind], logZe[ind], ze[ind], tau1e[ind], tau2e[ind], ne[ind], ince[ind])
        # indep_all.append(np.array([indep[i]+med_mod[i] for i in range(len(indep))]))
        # dep_all.append(dep)
        xx, res = make_lmfit(indep,dep,indep_name,args.degree,nlim[0],nlim[1])
        ngrid_best = np.array(list(res.best_values.values()))
        # best_fit_all.append(res.best_fit); best_unc.append(res.eval_uncertainty())
        coefs_all.append(polyfitnd(xx,ngrid_best))
        if size_orig==-1: args.size = size_orig

    # indep_all, dep_all, best_fit_all, best_unc = np.array(indep_all), np.array(dep_all), np.array(best_fit_all), np.array(best_unc)
    # indep_all_avg = np.median(indep_all,axis=0); med_avg = np.median(indep_all_avg,axis=0)
    coefs_all = np.array(coefs_all); med_coefs = np.median(coefs_all,axis=0)

    img_dir = op.join(img_dir_orig,'deg_%d'%(args.degree),args.extratext)
    mkpath(img_dir)
    ndim = len(indep)
    # Save best-fit model to a data file
    dataarr = np.empty((ngrid_best.size,0))
    for i in range(len(xx)):
        dataarr = np.append(dataarr,xx[i].reshape(ngrid_best.size,1),axis=1)
    yy = calc_poly(np.array([xxi.ravel() for xxi in xx]),med_coefs,args.degree)
    dataarr = np.append(dataarr,yy[:,None],axis=1)
    header = '%s  ngrid'%('  '.join(indep_name))
    dataname = ''
    for name in indep_name: dataname += name+'_'
    np.savetxt(op.join(img_dir,'ngrid_%s%s_%s_lmfit_err.dat'%(dataname,dep_name,args.extratext)),dataarr,header=header,fmt='%.5f')

    if ndim==1:
        x = np.linspace(np.amin(xx[0]),np.amax(xx[0]),fine_grid)
        n_sim2 = np.empty((args.err_samp,fine_grid))
        for i, coef in enumerate(coefs_all):
            n_sim2[i] = calc_poly(x[None,:],coef,args.degree)
        n_mean2, n_err2 = np.mean(n_sim2,axis=0), np.std(n_sim2,axis=0)
        plot1D(x+med_mod[0],n_mean2,xx[0].ravel()+med_mod[0],yy,img_dir,'Real_Model_%s_%s_%s_lmfit_err'%(indep_name[0],dep_name,args.extratext),xlab=indep_lab[0],ylab='LMFIT Error-Based Model '+dep_lab,yerr=n_err2)
        return

    med_mod2 = np.median(indep,axis=1)
    indep_div, indep_avg = np.ones((2,bins+1)), np.ones((2,bins))
    znum = np.zeros((bins,bins)) # Will fill with numbers of galaxies
    for i in range(ndim):
        namei = indep_name[i]
        for j in range(i+1,ndim):
            namej = indep_name[j]
            indep_fine = med_mod2[:,None]*np.ones((ndim,fine_grid))
            # Define grids of the independent variables: a fine one (xx_fine) and a coarse one (xx_div)
            index = 0
            for k in [i,j]:
                indep_fine[k] = np.linspace(np.amin(xx[k]),np.amax(xx[k]),fine_grid)
                indep_div[index] = np.linspace(np.amin(xx[k]),np.amax(xx[k])+1.0e-8,bins+1)
                indep_avg[index] = np.linspace((indep_div[index][0]+indep_div[index][1])/2.0,(indep_div[index][-1]+indep_div[index][-2])/2.0,bins)
                index+=1

            xx_fine = np.meshgrid(*indep_fine[(i,j),:])
            xx_div = np.meshgrid(*indep_avg)

            for k in range(i-1,-1,-1):
                medval = indep_fine[k,0]
                xx_fine.insert(0,medval*np.ones_like(xx_fine[0]))
            for k in range(i+1,j):
                medval = indep_fine[k,0]
                xx_fine.insert(-1,medval*np.ones_like(xx_fine[0]))
            for k in range(j+1,ndim):
                medval = indep_fine[k,0]
                xx_fine.insert(k,medval*np.ones_like(xx_fine[0]))

            # Determine number of galaxies in each coarse bin square
            for l in range(bins):
                condj = np.logical_and(indep[j]>=indep_div[1][l],indep[j]<indep_div[1][l+1])
                for k in range(bins):
                    condi = np.logical_and(indep[i]>=indep_div[0][k],indep[i]<indep_div[0][k+1])
                    cond = np.logical_and(condi,condj)
                    znum[l,k] = len(dep[cond])

            # Evaluate model at fine grid
            n_sim2 = np.empty((args.err_samp,fine_grid,fine_grid))
            for ii, coef in enumerate(coefs_all):
                n_sim2[ii] = calc_poly(xx_fine,coef,args.degree)
            n_mean2, n_err2 = np.mean(n_sim2,axis=0), np.std(n_sim2,axis=0)

            plot_color_map(xx_fine[i]+med_mod[i],xx_fine[j]+med_mod[j],n_mean2,xx_div[0]+med_mod[i],xx_div[1]+med_mod[j],znum,xx[i].ravel()+med_mod[i],xx[j].ravel()+med_mod[j],img_dir,'MeanModel_%s_%s_%s%s_lmfit_err'%(dep_name,indep_name[i],indep_name[j],args.extratext),levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='LMFIT model %s'%(dep_lab),xtrue=indep[i]+med_mod[i],ytrue=indep[j]+med_mod[j],xtrueerr=indep_err[i],ytrueerr=indep_err[j],minz=nlim[0],maxz=nlim[-1])
            plot_color_map(xx_fine[i]+med_mod[i],xx_fine[j]+med_mod[j],n_err2,xx_div[0]+med_mod[i],xx_div[1]+med_mod[j],znum,xx[i].ravel()+med_mod[i],xx[j].ravel()+med_mod[j],img_dir,'ErrModel_%s_%s_%s%s_lmfit_err'%(dep_name,indep_name[i],indep_name[j],args.extratext),levels=levels,xlab=indep_lab[i],ylab=indep_lab[j],zlab='LMFIT model uncertainty %s'%(dep_lab),xtrue=indep[i]+med_mod[i],ytrue=indep[j]+med_mod[j],xtrueerr=indep_err[i],ytrueerr=indep_err[j],minz=nlim[0],maxz=nlim[-1])


def main():
    # logM, ssfr, logZ, z, tau1, tau2, n, inc = getData()
    # indep = np.array([logM,ssfr,logZ,z,tau1,tau2,inc])
    # dep = np.array([n,tau2])
    # names_indep = ['logM','ssfr100','logZ','z','tau1','tau2','inc']
    # labels_indep = [r'$\log M_*$',r'$\log$ sSFR$_{\rm{100}}$',r'$\log (Z/Z_\odot)$','z',r"$\hat{\tau}_{\lambda,1}$",r"$\hat{\tau}_{\lambda,2}$",'b/a']
    # names_dep = ['n','tau2']
    # labels_dep = ['n',r"$\hat{\tau}_{\lambda,2}$"]
    # bins = 100
    # min_bin=1
    # for i,d in enumerate(dep):
    #     for j,i1 in enumerate(indep):
    #         for k in range(j+1,len(indep)):
    #             if i==1 and (j==5 or k==5): continue
    #             if k==6: cond = inc>=0.0
    #             else: cond = logM>0.0
    #             plot2DDustMean(i1[cond],indep[k][cond],d[cond],labels_indep[j],names_indep[j],labels_indep[k],names_indep[k],labels_dep[i],names_dep[i],extratext='full%d'%(bins),binx=bins,biny=bins,min_bin=min_bin)
    args = parse_args()
    img_dir_orig = op.join('DataTrue',args.dir_orig,'lmfit')
    indep, indep_err, dep, dep_err, med_mod, z, uniform, indep_name, indep_lab, dep_name, dep_lab, nlim = getNecessaryData(args)
    xx, res = make_lmfit(indep,dep,indep_name,args.degree,nlim[0],nlim[1])
    plot_lmfit(res,xx,indep,indep_err,dep,dep_err,indep_name,indep_lab,dep_name,dep_lab,med_mod,img_dir_orig=img_dir_orig,extratext=args.extratext,mindep=nlim[0],maxdep=nlim[1])

    # trace, xx, map_estimate = pymc3_simple(indep,dep,img_dir_orig,degree=args.degree,mindep=nlim[0],maxdep=nlim[1],sampling=args.steps,tune=args.tune,uniform=uniform,extratext=args.extratext,plot=args.plot)
    # plot_pymc3(trace,xx,indep,indep_err,dep,dep_err,indep_name,indep_lab,dep_name,dep_lab,med_mod,img_dir_orig=img_dir_orig,extratext=args.extratext,mindep=nlim[0],maxdep=nlim[1],map_estimate=map_estimate)

if __name__ == '__main__':
    # main()
    lmfit_err_analysis()