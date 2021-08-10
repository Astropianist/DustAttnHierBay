import numpy as np 
import os.path as op
from sedpy.attenuation import noll
import matplotlib.pyplot as plt 
import seaborn as sns
from anchor_points import polyfitnd, calc_poly
from DustSimple import getData
from UsePymc3Results import make_prop_dict
import pymc3 as pm
import arviz as az
from matplotlib import cm
import argparse as ap
from glob import glob
from astropy.table import Table
import pickle
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
sns.set_context("paper") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

wv_arr = np.linspace(1500.0,5000.0,501)
indepext = np.array(['m','s','sfr','logZ','z','i','d1','d2'])

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
    parser = ap.ArgumentParser(description="DustAttnCurves",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-m','--logM',help='Whether or not to include stellar mass in model',action='count',default=0)
    parser.add_argument('-s','--ssfr',help='Whether or not to include sSFR in model',action='count',default=0)
    parser.add_argument('-sfr','--sfr',help='Whether or not to include SFR in model',action='count',default=0)
    parser.add_argument('-logZ','--logZ',help='Whether or not to include metallicity in model',action='count',default=0)
    parser.add_argument('-z','--z',help='Whether or not to include redshift in model',action='count',default=0)
    parser.add_argument('-i','--i',help='Whether or not to include inclination in model',action='count',default=0)
    parser.add_argument('-d1','--d1',help='Whether or not to include birth cloud dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-d2','--d2',help='Whether or not to include diffuse dust optical depth as independent variable in model',action='count',default=0)
    parser.add_argument('-n','--n',help='Whether or not to use dust index as dependent variable',action='count',default=0)
    parser.add_argument('-dg','--degree',help='Degree of polynomial (model)',type=int,default=2)
    parser.add_argument('-np','--npts',help='Number of points for figure',type=int,default=3)
    parser.add_argument('-bv','--bivar',help='Whether or not to perform bivariate fitting',action='count',default=0)
    parser.add_argument('-pr','--prosp',help='Whether or not to use Prospector max likelihood solutions directly',action='count',default=0)
    parser.add_argument('-pre','--prosp_err',help='Whether or not to use Prospector posteriors (including errors) directly',action='count',default=0)
    parser.add_argument('-sl','--sliced',help='Whether or not to keep other variables constant',action='count',default=0)
    parser.add_argument('-po','--poly',help='Whether or not to include only polynomial component',action='count',default=0)
    parser.add_argument('-ma','--mvavg',help='Moving average number for attenuation comparison',type=int,default=10)
    parser.add_argument('-sa','--samples',help='Number of samples from model posteriors',type=int,default=50)
    parser.add_argument('-nproj','--nproj',help='Number of projections for 1D plots',type=int,default=5)
    parser.add_argument('-npm','--npts_mod',help='Number of points to compute model in independent parameter space',type=int,default=4000)

    args = parser.parse_args(args=argv)
    if args.bivar: args.dir_orig = 'Bivar_Prior_Sig'
    else: args.dir_orig = 'Univar_Sample'
    args.prop_dict, args.dep_dict = make_prop_dict()
    args.indep_pickle_name, args.indep_name, args.indep_lab, args.dataname = label_creation(args)
    return args

def label_creation(args,indepext=indepext):
    indep_pickle_name, indep_name, indep_lab = [], [], []
    dataname=''
    for i, name in enumerate(args.prop_dict['names'].keys()):
        if getattr(args,name):
            indep_pickle_name.append(args.prop_dict['pickle_names'][name])
            indep_name.append(args.prop_dict['names'][name])
            indep_lab.append(args.prop_dict['labels'][name])
            dataname+=indepext[i]
    if args.bivar: dataname+='n_d2'
    else:
        if args.n: dataname+='_n'
        else: dataname+='_d2'
    return indep_pickle_name, indep_name, indep_lab, dataname

def get_dust_attn_curve_d2(wave,n=0.0,d2=1.0):
    Eb = 0.85 - 1.9*n
    return noll(wave,tau_v=d2,delta=n,c_r=0.0,Ebump=Eb)

def D_lam(lam,B): # lam in micron
    return B*lam**2*(0.35)**2 / ((lam**2-0.2175**2)**2 + lam**2*0.35**2)

def klam(lam,a0,a1,a2,a3,B,Rv):
    return a0 + a1*lam**-1 + a2*lam**-2 + a3*lam**-3 + D_lam(lam,B) + Rv

def get_dust_attn_curve_d1(wave,d1=1.0):
    return d1*(wave/5500.0)**(-1)

def get_moving_avg(x,num=10):
    n = len(x)
    if n%num==0: rng = n//num
    else: rng = n//num+1
    xavg = np.zeros(rng)
    index = 0
    for i in range(rng):
        maxind = min(index+num,n)
        xavg[i] = np.average(x[index:maxind])
        index+=num
    return xavg

def bin_by_prop(prop,n,d1,d2,propname,proplab,numbins=5,wv_arr=wv_arr,numsamp=100):
    pers = np.linspace(0.,100.,numbins+1)
    bin_edge = np.percentile(prop[prop>-90],pers); bin_edge[-1]+=0.000001
    fig, ax = plt.subplots(2,1,sharex='all',sharey='all')
    attn_curve_d1, attn_curve_d2 = np.empty((numsamp,len(wv_arr))), np.empty((numsamp,len(wv_arr)))
    for i in range(numbins):
        cond = np.logical_and(prop>=bin_edge[i],prop<bin_edge[i+1])
        navg, d1avg, d2avg = np.average(n[cond]), np.average(d1[cond]), np.average(d2[cond])
        ax[0].plot(wv_arr,get_dust_attn_curve_d1(wv_arr,d1avg),linestyle='-',label='%.2f<%s<%.2f'%(bin_edge[i],proplab,bin_edge[i+1]))
        ax[1].plot(wv_arr,get_dust_attn_curve_d2(wv_arr,navg,d2avg),linestyle='--',color=ax[0].lines[-1].get_color(),label='')
        indfin = np.random.choice(len(n[cond]),numsamp,replace=False)
        for j in range(numsamp):
            attn_curve_d1[j] = get_dust_attn_curve_d1(wv_arr,d1[indfin][j])
            attn_curve_d2[j] = get_dust_attn_curve_d2(wv_arr,n[indfin][j],d2[indfin][j])
            # ax[0].plot(wv_arr,get_dust_attn_curve_d1(wv_arr,d1[indfin][j]),label='',color=ax[0].lines[-1].get_color(),alpha=0.01)
            # ax[1].plot(wv_arr,get_dust_attn_curve_d2(wv_arr,n[indfin][j],d2[indfin][j]),label='',color=ax[1].lines[-1].get_color(),alpha=0.01)
        ad1_mean, ad2_mean = np.mean(attn_curve_d1,axis=0), np.mean(attn_curve_d2,axis=0)
        ad1_err, ad2_err = np.std(attn_curve_d1,axis=0), np.std(attn_curve_d2,axis=0)
        ax[0].fill_between(wv_arr,ad1_mean-ad1_err,ad1_mean+ad1_err,alpha=0.2,color=ax[0].lines[-1].get_color(),label='')
        ax[1].fill_between(wv_arr,ad2_mean-ad2_err,ad2_mean+ad2_err,alpha=0.2,color=ax[1].lines[-1].get_color(),label='')
        
    ax[1].set_xlabel(r'$\lambda$ ($\AA$)')
    # ax[1].set_xlabel(r'$\lambda$ ($\AA$)')
    ax[0].set_ylabel(r'$\tau(\lambda)$'); ax[1].set_ylabel(r'$\tau(\lambda)$')
    ax[0].set_xlim(wv_arr[0],wv_arr[-1])
    ax[0].set_ylim(0.5,3.0)
    ax[0].legend(loc='best')
    fig.savefig(op.join('DustAttnCurves',f'DustAttnCurveProspBestFit_{propname}.png'),bbox_inches='tight',dpi=300)

def getProspDataBasic(numsamples=10, numgal=None):
    obj = pickle.load(open("3dhst_samples_%d_inc.pickle"%(numsamples),'rb'))
    logMavg, ssfravg, incavg, z = np.average(np.log10(obj['stellar_mass']),axis=1), np.average(np.log10(obj['ssfr_100']),axis=1), np.average(obj['inc'],axis=1), obj['z']
    masscomp = mass_completeness(z)
    cond = np.logical_and.reduce((logMavg>=masscomp,ssfravg>=-12.5,incavg>=0.0,incavg<=1.0))
    ind = np.where(cond)[0]
    if numgal==None: numgal = len(z[ind])
    indfin = np.random.choice(ind,size=numgal,replace=False)
    return obj, indfin

def marg_by_post(obj,ind,proplist,med,xx,numsamples=10,kdesamples=1000):
    xxlims = np.percentile(xx,[0.0,100.0],axis=tuple(range(1,xx[0].ndim+1)))
    input_arr = np.empty((len(ind)*numsamples,len(proplist)))
    cond = np.array([True]*len(ind)*numsamples)
    for i, prop in enumerate(proplist):
        if prop=='stellar_mass' or prop=='ssfr_100' or prop=='sfr_100': temp = np.log10(obj[prop][ind]).reshape(len(ind)*numsamples)
        elif prop=='z': temp = np.repeat(obj['z'][ind][:,None],numsamples,axis=1).reshape(len(ind)*numsamples)
        else: temp = obj[prop][ind].reshape(len(ind)*numsamples)
        input_arr[:,i] = temp - med[i]
        cond = np.logical_and.reduce((cond,input_arr[:,i]>=xxlims[0,i],input_arr[:,i]<=xxlims[1,i]))
    input_arr = input_arr[cond]
    kde = KernelDensity(bandwidth=0.2*len(proplist)-0.1,algorithm='auto', kernel='tophat', metric='euclidean', atol=1.0e-5, rtol=1.0e-5, breadth_first=True, leaf_size=40)
    kde.fit(input_arr)
    print("Finished fitting data with KDE")
    samp = kde.sample(n_samples=kdesamples, random_state=2039)
    # breakpoint()
    return samp.T

def plotAvSBivar(xx,trace,med,dg,pickle_labels,numsamp=50,npts=4000,fn='AvS/AvSBivar.png',mvavg=10,sliced=False,poly=False,numsamples=10,numgal=None):
    obj, indfin = getProspDataBasic(numsamples=numsamples, numgal=numgal)
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,'ngrid')), np.array(getattr(trace.posterior,'log_width'))
    taugrid_0, log_width2_0 = np.array(getattr(trace.posterior,'taugrid')), np.array(getattr(trace.posterior,'log_width2'))
    rho_0 = np.array(getattr(trace.posterior,'rho'))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],sh[2]), log_width_0.reshape(sh[0]*sh[1])
    taugrid, log_width2, rho = taugrid_0.reshape(sh[0]*sh[1],sh[2]), log_width2_0.reshape(sh[0]*sh[1]), rho_0.reshape(sh[0]*sh[1])
    if sliced: indep_samp = np.zeros((len(xx),npts))
    else: indep_samp = marg_by_post(obj,indfin,pickle_labels,med,xx,numsamples=numsamples,kdesamples=npts)

    inds = np.random.choice(len(log_width),size=numsamp,replace=False)
    n_sim_poly, tau_sim_poly = np.empty((numsamp,npts)), np.empty((numsamp,npts))
    n_sim, tau_sim = np.empty((numsamp,npts)), np.empty((numsamp,npts))
    for i, ind in enumerate(inds):
        ngrid_mod, taugrid_mod = ngrid[ind], taugrid[ind]
        width_mod, width2_mod, rho_mod = np.exp(log_width[ind]), np.exp(log_width2[ind]), rho[ind]
        coefs_mod, coefs2_mod = polyfitnd(xx,ngrid_mod), polyfitnd(xx,taugrid_mod)
        r1, r2 = np.random.randn(*n_sim[i].shape), np.random.randn(*n_sim[i].shape)
        n_sim_poly[i] = calc_poly(indep_samp,coefs_mod,dg)
        tau_sim_poly[i] = calc_poly(indep_samp,coefs2_mod,dg)
        n_sim[i] = n_sim_poly[i] + width_mod * r1
        tau_sim[i] = tau_sim_poly[i] + width2_mod * (rho_mod*r1 + np.sqrt(1.0-rho_mod**2)*r2)
    breakpoint()
    plotAvS(tau_sim*1.086,n_sim,fn,mvavg=mvavg,nsamp_poly=n_sim_poly,Avsamp_poly=tau_sim_poly*1.086)

def plotAvSUnivar(xx,trace,med,pickle_names,dg,numsamp=50,npts=4000,fn='AvS/AvSUnivar.png',mvavg=10,sliced=False,poly=False,numsamples=10,numgal=None):
    obj, indfin = getProspDataBasic(numsamples=numsamples, numgal=numgal)
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,'ngrid')), np.array(getattr(trace.posterior,'log_width'))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],sh[2]), log_width_0.reshape(sh[0]*sh[1])
    indep_samp = np.zeros((len(xx),npts))
    ind_tau = pickle_names.index('dust2')
    pic_not_tau = list(pickle_names)
    pic_not_tau.remove('dust2')
    ind_all = np.arange(len(xx))
    ind_not_tau = ind_all[ind_all!=ind_tau]
    if not sliced:
        indep_samp[ind_tau] = np.random.uniform(np.amin(xx[ind_tau]),np.amax(xx[ind_tau]),npts)
        indep_samp[ind_not_tau] = marg_by_post(obj,indfin,pic_not_tau,med,xx,numsamples=numsamples,kdesamples=npts)

    inds = np.random.choice(len(log_width),size=numsamp,replace=False)
    n_sim, n_sim_poly = np.empty((numsamp,npts)), np.empty((numsamp,npts))
    for i, ind in enumerate(inds):
        ngrid_mod = ngrid[ind]
        width_mod = np.exp(log_width[ind])
        coefs_mod = polyfitnd(xx,ngrid_mod)
        n_sim_poly[i] = calc_poly(indep_samp,coefs_mod,dg)
        n_sim[i] = n_sim_poly[i] + width_mod * np.random.randn(*n_sim[i].shape)
    Av = (indep_samp[ind_tau]+med[ind_tau])*1.086
    indsort = np.argsort(Av)
    plotAvS(Av[indsort],n_sim[:,indsort],fn,mvavg=mvavg,nsamp_poly=n_sim_poly[:,indsort])

def plotAvSProsp(mvavg=10):
    obj = pickle.load(open("3dhst_samples_500_inc.pickle",'rb'))
    logM, ssfr, tau2, n, z = np.log10(obj['stellar_mass'])[:,0], np.log10(obj['ssfr_100'])[:,0], obj['dust2'][:,0], obj['dust_index'][:,0], obj['z']
    masscomp = mass_completeness(z)
    cond = np.logical_and.reduce((logM>=masscomp,ssfr>=-12.5))
    ind = np.where(cond)[0]
    indsort = np.argsort(tau2[ind])
    plotAvS(tau2[ind][indsort]*1.086,n[ind][indsort],'AvS/AvSProsp_mvavg_%d.png'%(mvavg),mvavg=mvavg)

def plotAvSProspErr(mvavg=10):
    obj = pickle.load(open("3dhst_samples_10_inc.pickle",'rb'))
    logMavg, ssfravg, tau2, n, z = np.average(np.log10(obj['stellar_mass']),axis=1), np.average(np.log10(obj['ssfr_100']),axis=1), obj['dust2'], obj['dust_index'], obj['z']
    tau2avg = np.average(tau2,axis=1)
    masscomp = mass_completeness(z)
    cond = np.logical_and.reduce((logMavg>=masscomp,ssfravg>=-12.5))
    ind = np.where(cond)[0]
    indsort = np.argsort(tau2avg[ind])
    # breakpoint()
    plotAvS(tau2[ind][indsort].T*1.086,n[ind][indsort].T,'AvS/AvSProspErr_mvavg_%d.png'%(mvavg),mvavg=mvavg)

def plotAvS(Avsamp,nsamp,plotname,mvavg=10,nsamp_poly=None,Avsamp_poly=None):
    dust15, dust55 = np.empty(nsamp.shape), np.empty(nsamp.shape)
    wvarr = np.array([1500.0,5500.0])
    if nsamp_poly is not None:
        dust15_poly, dust55_poly = np.empty(nsamp.shape), np.empty(nsamp.shape)
    if nsamp.ndim==1:
        for i in range(len(nsamp)):
            dust15[i], dust55[i] = get_dust_attn_curve_d2(wvarr,nsamp[i],Avsamp[i]/1.086)
        S, Serr = dust15/dust55, None
        Av, Averr = Avsamp, None
        reverse = interp1d(nsamp,S,kind='cubic',fill_value='extrapolate')
        forward = interp1d(S,nsamp,kind='cubic',fill_value='extrapolate')
    else:
        for i in range(nsamp.shape[0]):
            for j in range(nsamp.shape[1]):
                if Avsamp.ndim==1: Avij = Avsamp[j]
                else: Avij = Avsamp[i,j]
                dust15[i,j], dust55[i,j] = get_dust_attn_curve_d2(wvarr,n=nsamp[i,j],d2=Avij/1.086)
        if nsamp_poly is not None:
            for i in range(nsamp.shape[0]):
                for j in range(nsamp.shape[1]):
                    if Avsamp.ndim==1: Avij = Avsamp[j]
                    else: Avij = Avsamp_poly[i,j]
                    dust15_poly[i,j], dust55_poly[i,j] = get_dust_attn_curve_d2(wvarr,n=nsamp_poly[i,j],d2=Avij/1.086)
        Ssamp = dust15/dust55
        if nsamp_poly is not None: Ssamp_poly = dust15_poly/dust55_poly
        reverse = interp1d(nsamp.ravel(),Ssamp.ravel(),kind='cubic',fill_value='extrapolate')
        forward = interp1d(Ssamp.ravel(),nsamp.ravel(),kind='cubic',fill_value='extrapolate')
        # breakpoint()
        if Avsamp.ndim==1:
            # S, Serr = np.average(Ssamp,axis=0), np.std(Ssamp,axis=0)
            Sm, S, Sp = np.percentile(Ssamp,[16,50,84],axis=0)
            if nsamp_poly is not None: 
                # Serr_poly = np.std(Ssamp_poly,axis=0)
                Sm_poly, Sp_poly = np.percentile(Ssamp_poly,[16,84],axis=0)
            Av, Averr = Avsamp, None
        else:
            Avorig = np.average(Avsamp,axis=0)
            indsort = np.argsort(Avorig)
            # S, Serr = np.average(Ssamp,axis=0)[indsort], np.std(Ssamp,axis=0)[indsort]
            Sm, S, Sp = np.percentile(Ssamp,[16,50,84],axis=0)[:,indsort]
            if nsamp_poly is not None: 
                # Serr_poly = np.std(Ssamp_poly,axis=0)[indsort]
                Sm_poly, Sp_poly = np.percentile(Ssamp_poly,[16,84],axis=0)[:,indsort]
            Av, Averr = Avorig[indsort], np.std(Avsamp,axis=0)[indsort]
    breakpoint()
    AvSlab = ['Reddy','Buat','KC','MW','A-Marq','SMC','Battisti','Salim (z=0)','Calzetti']
    AvDisc, SDisc = np.loadtxt('AvSDiscrete.csv',delimiter=', ',unpack=True)
    AvSalim, SSalim = np.loadtxt('AvSSalim2018.csv',delimiter=', ',unpack=True)
    fig, ax = plt.subplots()
    for i, lab in enumerate(AvSlab):
        if i<6:
            ax.plot(AvDisc[i],SDisc[i],linestyle='none',marker='^')
            c = ax.lines[-1].get_color()
            if i==0: ax.annotate(lab,(AvDisc[i]+0.02,SDisc[i]-0.25),color=c)
            elif i==1: ax.annotate(lab,(AvDisc[i],SDisc[i]+0.1),color=c)
            elif i==4: ax.annotate(lab,(AvDisc[i]-0.1,SDisc[i]-0.2),color=c)
            else: ax.annotate(lab,(AvDisc[i],SDisc[i]),color=c)
        elif i==6:
            ax.plot(AvDisc[i:],SDisc[i:],'--^')
            ax.annotate(lab,(AvDisc[i+2],SDisc[i+2]),color=ax.lines[-1].get_color())
        elif i==7:
            ax.plot(AvSalim,SSalim,'b--')
            ax.annotate(lab,(AvSalim[10],SSalim[10]),color=ax.lines[-1].get_color())
        else:
            ax.hlines(2.5529,0.0,2.0,color='orange',lw=2,ls='dashed')
            ax.annotate(lab,(0.3,2.7),color='orange')
    win_len, polyorder = len(Av)//3, 1
    if win_len%2==0: win_len+=1
    Ssav = savgol_filter(S,win_len,polyorder)
    # Avavg, Savg = get_moving_avg(Av,num=mvavg), get_moving_avg(S,num=mvavg)
    # if Serr is not None: Serravg = get_moving_avg(Serr,num=mvavg)
    # else: Serravg = None
    ax.plot(Av,Ssav,'k-'); ax.annotate('Nagaraj',(Av[0]+0.05,Ssav[0]+0.1),color=ax.lines[-1].get_color())
    ax.plot(Av,S,'k,')
    if nsamp.ndim>1:
        SplusEsav = savgol_filter(Sp,win_len,polyorder)
        SminusEsav = savgol_filter(Sm,win_len,polyorder)
        ax.fill_between(Av,SminusEsav,SplusEsav,color='k',alpha=0.1)
        if nsamp_poly is not None:
            SplusEpolysav = savgol_filter(Sp_poly,win_len,polyorder)
            SminusEpolysav = savgol_filter(Sm_poly,win_len,polyorder)
            ax.fill_between(Av,SminusEpolysav,SplusEpolysav,color='k',alpha=0.4)
        # ax.fill_between(Avavg,Savg-Serravg,Savg+Serravg,color='k',alpha=0.2)
    secax = ax.secondary_yaxis('right',functions=(forward,reverse))
    ax.set_xlabel(r'$A_V$')
    ax.set_ylabel(r'Slope ($S=A_{\rm 1500}/A_{\rm 5500}$)')
    secax.set_ylabel(r'$n$')
    ax.tick_params(axis='y',which='both',right=False)
    ax.set_xlim([0.0,1.35])
    ax.set_ylim([1,8])
    fig.savefig(plotname,bbox_inches='tight',dpi=300)

def data_version(wv_arr=wv_arr,numsamp=1000):
    logM, ssfr, sfr, logZ, z, tau1, tau2, n, inc, logMe, ssfre, logZe, ze, tau1e, tau2e, ne, dq = getData()
    indfin = np.random.choice(logM.size,numsamp,replace=False)
    attn_curve_d2 = np.empty((numsamp,wv_arr.size))
    attn_curve_d1 = np.empty((numsamp,wv_arr.size))
    fig, ax = plt.subplots(2,1,sharex='all',sharey='all')
    # labeld1, labeld2 = '', ''
    for i in range(numsamp):
        # if i==numsamp-1: 
            # labeld1 = 'Birth Cloud Dust'
            # labeld2 = 'Diffuse Dust'
        attn_curve_d1[i] = get_dust_attn_curve_d1(wv_arr,tau1[indfin][i])
        attn_curve_d2[i] = get_dust_attn_curve_d2(wv_arr,n=n[indfin][i],d2=tau2[indfin][i])
        ax[0].plot(wv_arr,attn_curve_d1[i],'r',alpha=0.01)
        ax[1].plot(wv_arr,attn_curve_d2[i],'b',alpha=0.01)
    ax[1].set_xlabel(r'$\lambda$ ($\AA$)')
    # ax[1].set_xlabel(r'$\lambda$ ($\AA$)')
    ax[0].set_ylabel(r'$\tau(\lambda)$'); ax[1].set_ylabel(r'$\tau(\lambda)$')
    ax[0].set_xlim(wv_arr[0],wv_arr[-1])
    ax[0].set_ylim(0.0,6.0)
    # ax.legend(loc='best')
    fig.savefig('DustAttnCurveAllData.png',bbox_inches='tight',dpi=300)

def data_prop(wv_arr=wv_arr,numbins=5):
    logM, ssfr, sfr, logZ, z, tau1, tau2, n, inc, logMe, ssfre, logZe, ze, tau1e, tau2e, ne, dq = getData()
    prop_dict, dep_dict = make_prop_dict()
    prop_dict['arrays'] = {'logM': logM, 'ssfr': ssfr, 'logZ': logZ, 'z': z, 'i':inc, 'd1':tau1, 'd2':tau2}
    for name in prop_dict['names'].keys():
        bin_by_prop(prop_dict['arrays'][name],n,tau1,tau2,prop_dict['names'][name],prop_dict['labels'][name],numbins=numbins,wv_arr=wv_arr)

def plotDustAttn(xx,trace,pts,meds,names,labels,img_dir,numsamp=50,wv_arr=wv_arr,indep_samp=None):
    ndim, npts = len(meds), len(pts[0])
    dg = len(xx[0])-1
    if ndim==1 or ndim>4:
        print("Invalid number of dimensions for plotting")
        return
    if npts!=3 and npts!=5 and npts!=8:
        print("Can choose either 3, 5, or 8 points")
        return
    
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,'ngrid')), np.array(getattr(trace.posterior,'log_width'))
    taugrid_0, log_width2_0 = np.array(getattr(trace.posterior,'taugrid')), np.array(getattr(trace.posterior,'log_width2'))
    rho_0 = np.array(getattr(trace.posterior,'rho'))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],sh[2]), log_width_0.reshape(sh[0]*sh[1])
    taugrid, log_width2, rho = taugrid_0.reshape(sh[0]*sh[1],sh[2]), log_width2_0.reshape(sh[0]*sh[1]), rho_0.reshape(sh[0]*sh[1])

    inds = np.random.choice(len(log_width),size=numsamp,replace=False)
    n_sim, tau_sim = np.empty((numsamp,npts)), np.empty((numsamp,npts))
    for i, ind in enumerate(inds):
        ngrid_mod, taugrid_mod = ngrid[ind], taugrid[ind]
        width_mod, width2_mod, rho_mod = np.exp(log_width[ind]), np.exp(log_width2[ind]), rho[ind]
        coefs_mod, coefs2_mod = polyfitnd(xx,ngrid_mod), polyfitnd(xx,taugrid_mod)
        r1, r2 = np.random.randn(*n_sim[i].shape), np.random.randn(*n_sim[i].shape)
        n_sim[i] = calc_poly(pts,coefs_mod,dg) + width_mod * r1
        tau_sim[i] = calc_poly(pts,coefs2_mod,dg) + width2_mod * (rho_mod*r1 + np.sqrt(1.0-rho_mod**2)*r2)
    # n_mean, n_err = np.mean(n_sim,axis=0), np.std(n_sim,axis=0)
    # tau_mean, tau_err = np.mean(tau_sim,axis=0), np.std(tau_sim,axis=0)
    cseq = np.random.rand(npts,3)
    smin, smax = 2, 20 # For point sizes
    xt, yt = 0.4, 1.0-0.10*ndim
    if npts==3:
        fig, ax = plt.subplots(2,2,figsize=(6,6))
        axpts, axdust = ax[1,0], np.delete(ax,2)
        # xt, yt = 0.5, 1.0-0.08*ndim
    elif npts==5:
        fig, ax = plt.subplots(2,3,figsize=(9,6))
        axpts, axdust = ax[1,1], np.delete(ax,4)
    else:
        fig, ax = plt.subplots(3,3,figsize=(9,9))
        axpts, axdust = ax[1,1], np.delete(ax,4)
        # xt, yt = 0.3, 0.8-0.08*ndim
    
    axpts.set_xlabel(labels[0]); axpts.set_ylabel(labels[1])
    if ndim==2: axpts.scatter(pts[0]+meds[0],pts[1]+meds[1],c=cseq)
    else:
        s = (smax-smin)/(max(pts[2])-min(pts[2]))*(pts[2]-min(pts[2])) + smin
        if ndim==3: axpts.scatter(pts[0]+meds[0],pts[1]+meds[0],s=s,c=cseq)
        else: 
            cmap = cm.get_cmap('viridis')
            normpt3 = (pts[3]-min(pts[3]))/(max(pts[3])-min(pts[3]))
            sc = axpts.scatter(pts[0]+meds[0],pts[1]+meds[1],s=s,c=pts[3]+meds[3],cmap=cmap)
            cseq = cmap(normpt3)
            fig.colorbar(sc,ax=ax[:,:],label=labels[3])
    if indep_samp is not None:
        axpts.plot(indep_samp[0],indep_samp[1],'k,',alpha=0.2)
    maxattn = 0.0
    attn_curve = np.zeros((numsamp,len(wv_arr)))
    for i in range(npts):
        labi = ''
        for ii, lab in enumerate(labels): labi += lab+' = %0.2f\n'%(pts[ii,i]+meds[ii])
        labi = labi.rstrip(r'\n')
        axdust[i].text(xt,yt,labi,transform=axdust[i].transAxes,fontsize='medium')
        axdust[i].set_xlabel(r'$\lambda$ ($\AA$)')
        axdust[i].set_ylabel(r"$\hat{\tau}_{\lambda,2}$")
        for j in range(numsamp):
            attn_curve[j] = get_dust_attn_curve_d2(wv_arr,n_sim[j,i],tau_sim[j,i])
            # axdust[i].plot(wv_arr,attn,color=cseq[i],linestyle='--')
        attn_mean, attn_std = np.mean(attn_curve,axis=0), np.std(attn_curve,axis=0)
        maxattn = max(maxattn,max(attn_mean+attn_std))
        axdust[i].plot(wv_arr,attn_mean,color=cseq[i],linestyle='-',linewidth=2)
        axdust[i].fill_between(wv_arr,attn_mean-attn_std,attn_mean+attn_std,color=cseq[i],alpha=0.1)
    for i in range(npts):
        axdust[i].set_xlim(min(wv_arr),max(wv_arr))
        axdust[i].set_ylim(0.0,maxattn)
    if ndim<4: plt.tight_layout()
    figname = 'AttnCurves_'
    if indep_samp is not None: figname += 'pts_'
    for name in names: figname+=name
    figname+='_n_d2_dg%d_npts%d.png'%(dg,npts)
    fig.savefig(op.join(img_dir,figname),bbox_inches='tight',dpi=300)

def makeDustAttnCurves(args,img_dir_orig=op.join('DataFinal','Bivar_Sample')):
    trace, xx, med = getPostModelData(args,img_dir_orig=img_dir_orig)
    pts = np.empty((ndim,args.npts))
    for i in range(ndim):
        pts[i] = np.random.uniform(min(xx[i]),max(xx[i]),args.npts)

    logM, ssfr, sfr, logZ, z, tau1, tau2, n, inc, logMe, ssfre, logZe, ze, tau1e, tau2e, ne, dq = getData()
    indep_samp = np.empty((0,len(logM)))
    if args.logM: indep_samp = np.append(indep_samp,logM[None,:],axis=0)
    if args.ssfr: indep_samp = np.append(indep_samp,ssfr[None,:],axis=0)
    if args.logZ: indep_samp = np.append(indep_samp,logZ[None,:],axis=0)
    if args.z: indep_samp = np.append(indep_samp,z[None,:],axis=0)
    if args.i: indep_samp = np.append(indep_samp,inc[None,:],axis=0)
    plotDustAttn(xx,trace,pts,med,args.indep_name,args.indep_lab,'DustAttnCurves',indep_samp=indep_samp)

def getPostModelData(args,img_dir_orig=op.join('DataFinal','Bivar_Sample')):
    globsearch = op.join(img_dir_orig,'deg_%d'%(args.degree),'%s*'%(args.dataname),'*.nc')
    nclist = glob(globsearch)
    if len(nclist)==0:
        print("No netcdf file found in glob search",globsearch)
        return
    if len(nclist)>1:
        print("Multiple files or directories found in search",globsearch)
        return
    trace = az.from_netcdf(nclist[0])
    print("Trace file:", nclist[0])
    globsearch = op.join(img_dir_orig,'deg_%d'%(args.degree),'%s*'%(args.dataname),'*.dat')
    datf = glob(globsearch)[0]
    dat = Table.read(datf,format='ascii')
    print("Dat file:",datf)
    ndim = len(args.indep_name)
    sh = tuple([ndim]+[args.degree+1]*ndim)
    xx = np.empty((ndim,(args.degree+1)**ndim))
    med = np.empty(ndim)
    for i, name in enumerate(args.indep_name):
        xx[i] = dat[name]
        med[i] = dat[name+'_plus_med'][0]-dat[name][0]
    
    return trace, xx.reshape(sh), med

if __name__=='__main__':
    # data_version()
    # data_prop()
    args = parse_args()
    # makeDustAttnCurves(args)
    print("args:", args)
    if args.prosp:
        if args.prosp_err: plotAvSProspErr(mvavg=args.mvavg)
        else: plotAvSProsp(mvavg=args.mvavg)
    else:
        img_dir_orig = op.join('DataFinal',args.dir_orig)
        trace, xx, med = getPostModelData(args,img_dir_orig)
        if args.bivar: plotAvSBivar(xx, trace, med, args.degree, args.indep_pickle_name, mvavg=args.mvavg, fn='AvS/AvSBivar_%s_sl_%d_np_%d_sa_%d_jp.png'%(args.dataname,args.sliced,args.npts_mod,args.samples), sliced=args.sliced, poly=args.poly, npts=args.npts_mod, numsamp=args.samples)
        else: plotAvSUnivar(xx, trace, med, args.indep_pickle_name, args.degree, mvavg=args.mvavg, fn='AvS/AvSUnivar_%s_sl_%d_np_%d_sa_%d_v2.png'%(args.dataname,args.sliced,args.npts_mod,args.samples), sliced=args.sliced, poly=args.poly, npts=args.npts_mod, numsamp=args.samples)