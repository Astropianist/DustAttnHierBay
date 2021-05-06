import pickle
from astropy.table import Table
import numpy as np 
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from scipy.stats import norm, multivariate_normal
from time import time
from scipy.optimize import minimize, LinearConstraint
from uncertainties import unumpy
from copy import copy
from elliptical_slice import elliptical_slice
import corner
from itertools import cycle
from dynesty import utils as dyfunc

tollims = 1.0e-8
fdust2 = norm(loc=0.3,scale=1.0)
fdust12 = norm(loc=1.0,scale=0.3)
markers = cycle(['o','^','*','s','<','>','p','P','H','X'])

my_cmap = copy(plt.cm.get_cmap('viridis'))
my_cmap.set_under('lightgrey')

def mass_completeness(zred):
    """used mass-completeness estimates from Tal+14, for FAST masses
    then applied M_PROSP / M_FAST to estimate Prospector completeness
    """

    zref = np.array([0.65,1,1.5,2.1,3.0])
    mcomp_prosp = np.array([8.71614882,9.07108637,9.63281923,9.79486727,10.15444536])
    mcomp = np.interp(zred, zref, mcomp_prosp)

    return mcomp

def getHistDiv(llim,ulim,jtot=None):
    """ Divide N-D parameter space into bins

    Input
    -----
    llim: 1-D Numpy Array
        Lower limits for each parameter
    ulim: 1-D Numpy Array
        Upper limits for each parameter
    jtot: Float (optional)
        Dimensionality of bins for each parameter
    
    Return
    ------
    corners: 2-D Numpy Array
        Histogram bin edges in each parameter
    """
    assert len(llim)==len(ulim)
    d = len(llim)
    if jtot is None: jtot = int(10/(d-1)**0.667)
    # jtot = 3 #Just to ensure things are working for minimization
    ranges = ulim-llim
    corners = np.zeros((d,jtot+1))
    for i in range(d):
        corners[i] = np.linspace(llim[i]-tollims*ranges[i],ulim[i]+tollims*ranges[i],jtot+1)
    return corners

def Qc(omega): #Assume everything complete for now
    """ Completeness as a function of the parameters 
    """
    return 1.0 

def gamtrue(omega,theta,corners):
    """ True occurrence rate 
    """
    n_om = len(omega)
    jtot = len(corners[0])-1
    mult = 1
    indfin = np.zeros(len(omega[0]),dtype=int)
    for i in range(n_om):
        ind = np.searchsorted(corners[i],omega[i])-1
        indfin+=ind*mult
        mult*=jtot
    return np.exp(theta[indfin])

def gamobs(omega,theta,corners):
    return Qc(omega)*gamtrue(omega,theta,corners)

def MCInteg(llims,ulims,theta,corners,ntrial=10**6):
    d = len(llims)
    ranges = ulims-llims
    omegas = ranges*np.random.rand(ntrial,d)+llims
    V = np.prod(ranges,axis=0)
    funcvals = gamobs(omegas.T,theta,corners)
    return V/ntrial * np.sum(funcvals)

def TrueInteg(llims,ulims,theta):
    ranges = ulims-llims
    V = np.prod(ranges,axis=0)
    return V/len(theta) * np.sum(np.exp(theta),axis=0)

def priorprob(omega):
    # return fdust2.pdf(omega[0])*fdust12.pdf(omega[1]/omega[0])*0.7142857*0.758
    # return fdust2.pdf(omega[0])*fdust12.pdf(omega[1]/omega[0])*0.7142857
    return fdust2.pdf(omega[0])*0.7142857

def lnlike(theta,omegafull,corners,llims,ulims,weights,ntrial=10**6):
    # print("Shapes of weights, omegafull, theta, corners:")
    # print(weights.shape,omegafull.shape,theta.shape,corners.shape)
    Nk = len(omegafull[0,0,:])
    nsamples = len(omegafull[0,:,0])
    # lnl = -MCInteg(llims,ulims,theta,corners,ntrial)
    lnl = -TrueInteg(llims,ulims,theta)
    for i in range(nsamples):
        lnl += np.log(1.0/Nk * np.sum(weights[i]*gamobs(omegafull[:,i,:],theta,corners)/priorprob(omegafull[:,i,:]),axis=0))
    return lnl

def neglnlike(theta,omegafull,corners,llims,ulims,weights):
    Nk = len(omegafull[0,0,:])
    nsamples = len(omegafull[0,:,0])
    lnl = -TrueInteg(llims,ulims,theta)
    for i in range(nsamples):
        lnl += np.log(1.0/Nk * np.sum(weights[i]*gamobs(omegafull[:,i,:],theta,corners)/priorprob(omegafull[:,i,:]),axis=0))
    return -lnl

def minimize_lnl(omegafull,corners,llims,ulims,weights):
    # x0 = 10.*np.random.rand((len(corners[0])-1)**len(omegafull[:,0,0]))-5.0
    x0 = np.zeros((len(corners[0])-1)**len(omegafull[:,0,0]))
    res = minimize(neglnlike,x0,args=(omegafull,corners,llims,ulims,weights),method='SLSQP',constraints=LinearConstraint(np.identity(len(x0)),-10.0,10.0))
    print("Success status: %d"%(res.success))
    return res.x

def plotBest2DHist(theta,corners,xlabel,ylabel,xsave,ysave,extratext='',vmin=None):
    if vmin is None: vmin=np.amin(theta)
    plt.figure()
    im = plt.pcolormesh(corners[0],corners[1],theta,cmap=my_cmap,vmin=vmin)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im,label=r'$\ln\ \Gamma_\theta(\mathbf{\omega})$')
    plt.savefig("BestFitHist2D_%s_and_%s_complete%s.png"%(xsave,ysave,extratext),bbox_inches='tight',dpi=300)

def InvDetEff(omega,llims,ulims,corners,missval=-999.0):
    ranges = ulims-llims
    V = np.prod(ranges,axis=0)
    n_om = len(omega)
    jtot = len(corners[0])-1
    jfull = jtot**len(corners)
    Vj = V/jfull #Volume of a single element
    mult = 1
    indfin = np.zeros(len(omega[0]),dtype=int)
    for i in range(n_om):
        ind = np.searchsorted(corners[i],omega[i])-1
        indfin+=ind*mult
        mult*=jtot
    allind = np.arange(jfull)
    uniq, cts = np.unique(indfin,return_counts=True)
    diff = np.setdiff1d(allind,uniq,assume_unique=True)
    theta, thetae = np.zeros(jfull), np.zeros(jfull)
    expth = cts/Vj
    exptherr = expth / np.sqrt(cts)
    uth = unumpy.log(unumpy.uarray(expth,exptherr))
    theta[uniq], thetae[uniq] = unumpy.nominal_values(uth), unumpy.std_devs(uth)
    theta[diff], thetae[diff] = missval, missval
    return theta, thetae

def getLoc(ind,bin_c):
    indtemp = ind
    n = len(bin_c)
    jtot = len(bin_c[0])
    omind = np.zeros(n)
    for k in range(n):
        indk = indtemp // jtot**(n-k-1)
        indtemp -= indk*jtot**(n-k-1)
        omind[-1-k] = bin_c[-1-k][indk]
    return omind

def priorth(mu,lam0,lamvec,corners):
    n = len(lamvec)
    jtot = len(corners[0])-1
    J = jtot**n
    siginv = np.diag(1./lamvec**2)
    bin_c = np.zeros((n,jtot))
    for i in range(n):
        bin_c[i] = [(corners[i][j]+corners[i][j+1])/2.0 for j in range(jtot)]
    K = np.zeros((J,J))
    for i in range(J):
        omi = getLoc(i,bin_c)
        for j in range(J):
            omj = getLoc(j,bin_c)
            matmult = np.ndarray.item(np.matrix(omi-omj)*siginv*np.matrix(omi-omj).T)
            K[i,j] = lam0 * np.exp(-0.5*matmult)
    # return multivariate_normal(mean=mu*np.ones(J),cov=K,allow_singular=True), K
    return K

def hyperprior(hyper):
    flag = hyper[0]<5.0 * hyper[0]>-5.0
    for hy in hyper[1:]:
        flag *= hy>0.01 * hy<10.0
    return flag

def priorMH(hyper,theta,covold,corners,sigma=1.0):
    covhyper = sigma**2 * np.identity(len(hyper))
    # phhyold = hyperprior(hyper)
    phhynew = 0
    while phhynew==0:
        hynew = np.random.multivariate_normal(hyper,covhyper)
        phhynew = hyperprior(hynew)
    u = np.random.rand()
    mv = multivariate_normal(mean=hyper[0]*np.ones(len(theta)),cov=covold)
    covnew = priorth(hynew[0],hynew[1],hynew[2:],corners)
    norm1 = mv.pdf(theta)
    mv.mean=hynew[0]*np.ones(len(theta)); mv.cov = covnew
    norm2 = mv.pdf(theta)
    if u<norm2/norm1: 
        return hynew
    else:
        return hyper

def runESS(omegafull,corners,llims,ulims,weights,sigma=1.0,mcnum=20,return_samples=False,return_hyper_samples=False):
    n_om = len(omegafull)
    jtot = len(corners[0])-1
    mu, lam0 = 0.0, 1.0
    lamvec = np.ones(len(corners))
    hyper = np.concatenate((np.array([mu,lam0]),lamvec))
    th = np.zeros(jtot**n_om)
    K = priorth(hyper[0],hyper[1],hyper[2:],corners)
    lik = lnlike(th,omegafull,corners,llims,ulims,weights)
    samples = np.zeros((mcnum+1,len(th)+1))
    samples[0] = np.append(th,lik)
    samphyp = hyper.reshape(1,len(hyper))
    print("like init: %.3f"%(lik))
    for i in range(mcnum):
        th, lik = elliptical_slice(th,np.linalg.cholesky(K),lnlike,pdf_params=(omegafull,corners,llims,ulims,weights),cur_lnpdf=lik)
        # print("like new: %.3f"%(lik))
        if i%10==9: 
            hyper = priorMH(hyper,th,K,corners,sigma=sigma)
            K = priorth(hyper[0],hyper[1],hyper[2:],corners)
            samphyp = np.append(samphyp,hyper.reshape(1,len(hyper)),axis=0)
            # print("hyper new:"); print(hyper)
        samples[i+1] = np.append(th,lik)
    if return_hyper_samples and return_samples:
        return hyper, th, lik, samples, samphyp
    elif return_samples and not return_hyper_samples:
        return hyper, th, lik, samples
    elif return_hyper_samples and not return_samples:
        return hyper, th, lik, samphyp
    else:
        return hyper, th, lik

def getValsErr(omega,weights):
    omavg = np.sum(weights*omega,axis=1)/np.sum(weights,axis=1)
    numer = np.sum(weights*(omega-omavg.reshape(len(omavg),1))**2,axis=1)
    N = len(omega[0])
    denom = (N-1)/N * np.sum(weights,axis=1)
    return omavg, np.sqrt(numer/denom)

def plot2DErr(omegafull,weights,labels,names,extratext=''):
    n = len(omegafull)
    for i in range(n):
        omi,omie = getValsErr(omegafull[i],weights)
        for j in range(i+1,n):
            fig, ax = plt.subplots()
            omj,omje = getValsErr(omegafull[j],weights)
            il, ih = np.percentile(omi,[2.5,97.5])
            jl, jh = np.percentile(omj,[2.5,97.5])
            ax.scatter(omi,omj,s=1,c='b',marker=',',alpha=0.5)
            ax.errorbar(omi,omj,yerr=omje,xerr=omie,fmt='none',ecolor='k',alpha=0.004)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.set_xlim([il,ih])
            ax.set_ylim([jl,jh])
            fig.savefig("TwoParamPlots/%s_vs_%s%s.png"%(names[j],names[i],extratext),bbox_inches='tight',dpi=300)

def plotPosteriors(omegafull,weights,labels,names,numgals=8,markmin=0.3,markmax=3):
    n = len(omegafull)
    gals = np.random.choice(len(omegafull[0]),size=numgals,replace=False)
    for i in range(n):
        il, ih = np.percentile(omegafull[i,gals,:],[2.5,97.5])
        for j in range(i+1,n):
            jl, jh = np.percentile(omegafull[j,gals,:],[2.5,97.5])
            fig, ax = plt.subplots()
            for k, gal in enumerate(gals):
                sampi = dyfunc.resample_equal(omegafull[i,gal,:],weights[gal])
                sampj = dyfunc.resample_equal(omegafull[j,gal,:],weights[gal])
                wei = weights[gal]
                # s = (markmax-markmin)/(max(wei)-min(wei)) * (wei-min(wei)) + markmin
                ax.scatter(sampi,sampj,marker=next(markers),s=1,alpha=0.5)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.set_xlim([il,ih])
            ax.set_ylim([jl,jh])
            fig.savefig("TwoParamPlots/Posterior_%s_vs_%s.png"%(names[j],names[i]),bbox_inches='tight',dpi=300)

def plot2DDustMean(x,y,z,xplot,xname,yplot,yname,zplot,zname,binx=50,biny=50,extratext='',levels=10):
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
            if len(z[cond])>=1:
                # print(len(x[condi]),len(y[condj]),len(z[cond]))
                zmean[j,i] = np.median(z[cond])
                znum[j,i] = len(z[cond])
    # print(zmean[-1,:])
    # print("Min of zmean:"); print(np.amin(zmean))
    plt.figure()
    xx, yy = np.meshgrid(xavg,yavg)
    print(xx.shape,len(xx[zmean>=vmin]))
    cf = plt.contourf(xx,yy,zmean,levels=levels,cmap=my_cmap,vmin=vmin)
    cnum = plt.contour(xx,yy,znum,levels=4,cmap='Greys')
    # cf = plt.pcolormesh(xdiv,ydiv,zmean,cmap=my_cmap,vmin=vmin)
    minx,maxx = min(xx[zmean>=vmin]),max(xx[zmean>=vmin])
    miny,maxy = min(yy[zmean>=vmin]),max(yy[zmean>=vmin])
    # print(np.amin(yy),np.amax(yy))
    print(minx,maxx,miny,maxy)
    plt.gca().set_xlim([minx,maxx])
    plt.gca().set_ylim([miny,maxy])
    plt.xlabel(xplot)
    plt.ylabel(yplot)
    plt.colorbar(cf,label=zplot)
    plt.colorbar(cnum,label='Number of Galaxies')
    plt.savefig("DataSim/SalimStyle_%s_%svs%s%s.png"%(zname,yname,xname,extratext),bbox_inches='tight',dpi=300)

def main():
    objects = []
    index = 0
    with (open("3dhst_samples.pickle",'rb')) as openfile:
        obj = pickle.load(openfile)
    print(obj.keys())
    print(np.amin(obj['log_z_zsun']),np.amax(obj['log_z_zsun']))
    print(obj['name'][:10])
    logMavg = np.log10(np.sum(obj['weights']*obj['stellar_mass'],axis=1)/np.sum(obj['weights'],axis=1))
    logssfravg = np.log10(np.sum(obj['weights']*obj['ssfr_100'],axis=1)/np.sum(obj['weights'],axis=1))
    logZavg = np.sum(obj['weights']*obj['log_z_zsun'],axis=1)/np.sum(obj['weights'],axis=1)
    lognavg = np.sum(obj['weights']*obj['dust_index'],axis=1)/np.sum(obj['weights'],axis=1)
    logd2avg = np.sum(obj['weights']*obj['dust2'],axis=1)/np.sum(obj['weights'],axis=1)
    logd1avg = np.sum(obj['weights']*obj['dust1'],axis=1)/np.sum(obj['weights'],axis=1)
    z = obj['z']
    print(min(z),max(z))
    print(min(lognavg),max(lognavg))
    masscomp = mass_completeness(z)
    print("logMavg properties:"); print(min(logMavg),max(logMavg))
    print("Masscomp properties:"); print(min(masscomp),max(masscomp))
    # cond = np.logical_and.reduce((z>0.5,z<1.0,logMavg>10.5))
    cond = np.logical_and.reduce((logMavg>=masscomp,z<3.0,logssfravg>-14.0))
    ind = np.where(cond)[0]
    omegafull = np.array([obj['dust2'][ind],obj['dust1'][ind],obj['dust_index'][ind],np.log10(obj['stellar_mass'][ind]),np.log10(obj['ssfr_100'][ind]),obj['log_z_zsun'][ind]])
    # # print(obj['log_z_zsun'])
    # # omegafull = np.array([obj['dust2'][ind],obj['dust_index'][ind]])
    weights = obj['weights'][ind]
    # print("Omegafull shape:")
    # print(omegafull.shape)
    
    # llim = np.array([0.0,-1.0])
    # ulim = np.array([4.0,0.4])
    # corners = getHistDiv(llim,ulim,jtot=3)
    
    # thetabest = minimize_lnl(omegafull,corners,llim,ulim,weights)
    # theta, thetae = InvDetEff(omegafull[:,:,0],llim,ulim,corners)
    # np.savetxt("BestFitTheta3.dat",thetabest,fmt='%.5f',header='Theta')
    # np.savetxt("BestFitThetaIDEJ%d.dat"%(len(theta)),np.column_stack((theta,thetae)),fmt='%.5f',header='Theta   Theta_err')
    # thetabest = np.loadtxt("BestFitTheta2.dat",usecols=0,unpack=True)
    # thetadim = len(corners[0])-1
    # plotBest2DHist(thetabest.reshape(thetadim,thetadim),corners,r"$\hat{\tau}_{\lambda,2}$",'n',"dust2","n",extratext="_3")
    # plotBest2DHist(theta.reshape(thetadim,thetadim),corners,r"$\hat{\tau}_{\lambda,2}$",'n',"dust2","n",extratext="_IDEJ%d"%(len(theta)),vmin=min(theta[theta>-900]))

    # mcnum = 1000
    # hyper, th, lik, samples, samphyp = runESS(omegafull,corners,llim,ulim,weights,return_samples=True,return_hyper_samples=True,mcnum=mcnum)
    # throw_frac = 0.4
    # # samples = samples[int(throw_frac*len(samples)):]
    # # samphyp = samphyp[int(throw_frac*len(samphyp)):]
    # names = [r'$\theta_%d$'%(ind) for ind in range(len(th))]
    # names.append('Ln_Prob')
    # nameshyp = [r'$\mu$',r'$\lambda_0$',r'$\lambda_{D2}$',r'$\lambda_n$']
    # print("hyper: "); print(hyper)
    # print('th_best: '); print(th)
    # print("Final like: %.3f"%(lik))
    # # print(samples); print(samphyp)
    # T = Table(samples, names=names)
    # T.write("BestFitThetaESS_bins%d_mc%d"%(len(th),mcnum),format='ascii.fixed_width_two_line')
    # T = Table(samphyp, names=nameshyp)
    # T.write("BestFitHypESS_bins%d_mc%d"%(len(th),mcnum),format='ascii.fixed_width_two_line')

    # thetadim = len(corners[0])-1
    # plotBest2DHist(th.reshape(thetadim,thetadim),corners,r"$\hat{\tau}_{\lambda,2}$",'n',"dust2","n",extratext="_ESS_bins%d_mc%d"%(len(th),mcnum))

    # fig = corner.corner(samples[int(throw_frac*len(samples)):,:-1],bins=6,labels=names[:-1],range=[0.95]*len(names[:-1]),show_titles=True)
    # fig.savefig("CornerPlot_dust2_n_ESS_bins%d_mc%d.png"%(len(th),mcnum),bbox_inches='tight',dpi=150)

    names = ['dust2','dust1','n','logM','ssfr100','logZ']
    labels= [r"$\hat{\tau}_{\lambda,2}$",r"$\hat{\tau}_{\lambda,1}$",'n',r'$\log M_*$',r'$\log$ sSFR$_{\rm{100}}$',r'$\log (Z/Z_\odot)$']
    # numgals = 8
    # plot2DErr(omegafull,weights,labels,names,extratext='full_z')
    # plotPosteriors(omegafull,weights,labels,names,numgals=numgals)
    cond = np.logical_and.reduce((cond,z>=0.5,z<1.0))
    plot2DDustMean(logMavg[cond],logssfravg[cond],lognavg[cond],labels[3],names[3],labels[4],names[4],labels[2],names[2],extratext='low_z',binx=50,biny=50,levels=20)
    plot2DDustMean(logMavg[cond],logd2avg[cond],lognavg[cond],labels[3],names[3],labels[0],names[0],labels[2],names[2],extratext='low_z',binx=50,biny=50,levels=20)
    plot2DDustMean(logMavg[cond],logZavg[cond],lognavg[cond],labels[3],names[3],labels[5],names[5],labels[2],names[2],extratext='low_z',binx=50,biny=50,levels=20)


if __name__=='__main__':
    main()