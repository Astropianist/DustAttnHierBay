import numpy as np 
import pickle
from dynesty import utils as dyfunc
import os.path as op
from DustAttnCurves import getPostModelData, parse_args, mass_completeness
from anchor_points import calc_poly, polyfitnd
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm
from itertools import cycle
import seaborn as sns
sns.set_context("paper") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

markers = cycle(['o','^','*','s','<','>','p','P','H','X'])

def get_like(args,indep_orig,dep,correct_med=True):
    if args.bivar: img_dir_orig = op.join('DataFinal','Bivar_Sample')
    else: img_dir_orig = op.join('DataFinal','Univar_Sample')
    trace, xx, med = getPostModelData(args,img_dir_orig)
    if correct_med:
        indep = np.empty(indep_orig.shape)
        for i in range(len(med)):
            indep[i] = indep_orig[i] - med[i]
    else: indep = indep_orig 
    # breakpoint()
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,'ngrid')), np.array(getattr(trace.posterior,'log_width'))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],sh[2]), log_width_0.reshape(sh[0]*sh[1])
    ngrid_med, width_med = np.median(ngrid,axis=0), np.exp(np.median(log_width))
    ncoef = polyfitnd(xx, ngrid_med)
    nmod = calc_poly(indep, ncoef, args.degree)
    if not args.bivar: 
        like = 1.0/(width_med*np.sqrt(2.0*np.pi)) * np.exp(-0.5*(dep-nmod)**2/width_med**2)
    else:
        taugrid_0, log_width2_0, rho_0 = np.array(getattr(trace.posterior,'taugrid')), np.array(getattr(trace.posterior,'log_width2')), np.array(getattr(trace.posterior,'rho'))
        taugrid, log_width2, rho = taugrid_0.reshape(sh[0]*sh[1],sh[2]), log_width2_0.reshape(sh[0]*sh[1]), rho_0.reshape(sh[0]*sh[1])
        taugrid_med = np.median(taugrid,axis=0)
        width2_med, rho_med = np.exp(np.median(log_width2)), np.median(rho)
        taucoef = polyfitnd(xx, taugrid_med)
        taumod = calc_poly(indep, taucoef, args.degree)
        z = (dep[0]-nmod)**2/width_med**2 + (dep[1]-taumod)**2/width2_med**2 - 2.0*rho_med*(dep[0]-nmod)*(dep[1]-taumod)/(width_med*width2_med)
        like = 1.0/(2.0*np.pi*width_med*width2_med*np.sqrt(1.0-rho_med**2)) * np.exp(-0.5*z/(1.0-rho_med**2))
    return like
    
def calcPosteriors(args,omegafull,prior_prob,indep,dep,rep_num,plot=True):
    like = get_like(args, indep, dep)
    print("Finished getting likelihood")
    # weight = like/prior_prob # Weight is new prior probability over old prior probability
    myclip_a, myclip_b, my_mean, my_std = 0.0, 4.0, 0.3, 1.0
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    if args.bivar: 
        prprior = truncnorm.pdf(dep[1],a,b,loc=my_mean,scale=my_std) / 1.4
        weight = like / prprior
    else:  
        weight = like*1.4
    weight /= np.sum(weight) # Normalize
    print("Finished getting weights")
    newSamp = np.empty(omegafull.shape)
    for j in range(len(omegafull)):
        for i, w in enumerate(weight):
            # breakpoint()
            newSamp[j,i,:] = dyfunc.resample_equal(omegafull[j,i,:],w)
    if plot: plotPosteriors(omegafull,newSamp,args.labels,args.names,rep_num)
    return newSamp

def plotPosteriors(omega_orig,omega_new,labels,names,rep_num,markmin=0.3,markmax=3):
    n = len(omega_orig)
    for i in range(n):
        il, ih = np.percentile(omega_orig[i],[2.5,97.5])
        for j in range(i+1,n):
            jl, jh = np.percentile(omega_orig[j],[2.5,97.5])
            fig, ax = plt.subplots()
                # s = (markmax-markmin)/(max(wei)-min(wei)) * (wei-min(wei)) + markmin
            for k in range(len(omega_orig[0])):
                marker_current = next(markers)
                ax.scatter(omega_orig[i,k,:],omega_orig[j,k,:],marker=marker_current,s=1,alpha=0.5)
                ax.scatter(omega_new[i,k,:],omega_new[j,k,:],marker=marker_current,s=1,alpha=0.5)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.set_xlim([il,ih])
            ax.set_ylim([jl,jh])
            fig.savefig("HierShrinkage/Posterior_%s_vs_%s_%d.png"%(names[j],names[i],rep_num),bbox_inches='tight',dpi=300)

def main(numgal=5,numrep=3):
    args = parse_args()
    args.numgal, args.numrep = numgal, numrep
    args.names = ['dust2','dust1','n','logM','ssfr100','logZ']
    args.labels= [r"$\hat{\tau}_{\lambda,2}$",r"$\hat{\tau}_{\lambda,1}$",'n',r'$\log M_*$',r'$\log$ sSFR$_{\rm{100}}$',r'$\log (Z/Z_\odot)$']
    obj = pickle.load(open('3dhst_samples_500_inc.pickle','rb'))
    prior_prob = np.exp(obj['lnprobability']-obj['lnlikelihood'])
    logM, ssfr = np.log10(obj['stellar_mass']), np.log10(obj['ssfr_100'])
    logZ = obj['log_z_zsun']
    logMavg = np.average(logM,axis=1); logssfravg = np.average(ssfr,axis=1)
    z = obj['z']
    n = obj['dust_index']
    tau1 = obj['dust1']
    tau2 = obj['dust2']
    inc = obj['inc']
    masscomp = mass_completeness(z)
    cond = np.logical_and.reduce((logMavg>=masscomp,logssfravg>-12.0))
    if args.i: cond = np.logical_and.reduce((cond,np.median(inc,axis=1)>=0.0,np.median(inc,axis=1)<=1.0))
    ind = np.where(cond)[0]
    indep_samp = np.empty((0,len(z),len(n[0])))
    if args.logM: indep_samp = np.append(indep_samp,logM[None,:,:],axis=0)
    if args.ssfr: indep_samp = np.append(indep_samp,ssfr[None,:,:],axis=0)
    if args.logZ: indep_samp = np.append(indep_samp,logZ[None,:,:],axis=0)
    if args.z: 
        zrep = np.repeat(z[:,None],len(n[0]),axis=1)
        indep_samp = np.append(indep_samp,zrep[None,:,:],axis=0)
    if args.i: indep_samp = np.append(indep_samp,inc[None,:,:],axis=0)
    if args.d1: indep_samp = np.append(indep_samp,tau1[None,:,:],axis=0)
    if args.d2: indep_samp = np.append(indep_samp,tau2[None,:,:],axis=0)
    if args.bivar: dep = np.array([n, tau2])
    else:
        if args.n: dep = n
        else: dep = tau2

    for i in range(numrep):
        indfin = np.random.choice(ind,size=numgal,replace=False)
        omegafull = np.array([tau2[indfin],tau1[indfin],n[indfin],logM[indfin],ssfr[indfin],logZ[indfin]])
        if args.bivar: dep_ind = dep[:,indfin,:]
        else: dep_ind = dep[indfin]
        print("About to run calcPosteriors")
        newSamp = calcPosteriors(args, omegafull, prior_prob[indfin], indep_samp[:,indfin,:], dep_ind, i, plot=False)
        print("Finished running calcPosteriors")
        err_orig, err_new = np.std(omegafull,axis=2), np.std(newSamp,axis=2)
        fig, ax = plt.subplots(dpi=300)
        for j in range(len(omegafull)):
            indsort = np.argsort(err_orig[j])
            ax.plot(err_orig[j][indsort],err_new[j][indsort],linestyle='none',marker=next(markers), markersize=72./fig.dpi,label=args.labels[j],alpha=0.8)
        ax.set_xlabel('Original Error')
        ax.set_ylabel('New Error')
        xplot = np.linspace(0.0,1.0,101)
        ax.plot(xplot,xplot,'k--',label='1-1')
        ax.set_xlim([0.0,0.6])
        ax.set_ylim([0.0,0.6])
        ax.legend(loc='best',frameon=False,fontsize='small',markerscale=12)
        fig.savefig('HierShrinkage/PostComp_%s_ng_%d.png'%(args.dataname,numgal),bbox_inches='tight',dpi=300)

if __name__=='__main__':
    main(numgal=2000, numrep=1)