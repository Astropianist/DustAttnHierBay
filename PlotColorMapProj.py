import numpy as np 
import DustAttnCurveModules as dac
from copy import deepcopy
import os.path as op
from anchor_points import calc_poly, polyfitnd
from RegularLinearInterp import regular_grid_interp_scipy
from distutils.dir_util import mkpath
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_context("paper") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

nlab, taulab = 'n', r"$\hat{\tau}_{2}$"
nlims, taulims = (-1.0,0.4), (0.0, 3.0)
fig_size = (12,5)

def getTraceInfo(trace, bivar=False):
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,'ngrid')), np.array(getattr(trace.posterior,'log_width'))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],*sh[2:]), log_width_0.reshape(sh[0]*sh[1])
    if bivar:
        taugrid_0, log_width2_0, rho_0 = np.array(getattr(trace.posterior,'taugrid')), np.array(getattr(trace.posterior,'log_width2')), np.array(getattr(trace.posterior,'rho'))
        taugrid, log_width2, rho = taugrid_0.reshape(sh[0]*sh[1],*sh[2:]), log_width2_0.reshape(sh[0]*sh[1]), rho_0.reshape(sh[0]*sh[1])
    else:
        taugrid, log_width2, rho = None, None, None
    return ngrid, log_width, taugrid, log_width2, rho

def getTraceInfoCD(trace, bivar=False):
    ncoef_0, log_width_0 = np.array(getattr(trace.posterior,'ncoef')), np.array(getattr(trace.posterior,'log_width'))
    sh = ncoef_0.shape
    ncoef, log_width = ncoef_0.reshape(sh[0]*sh[1],sh[2]), log_width_0.reshape(sh[0]*sh[1])
    if bivar:
        taucoef_0, log_width2_0, rho_0 = np.array(getattr(trace.posterior,'taucoef')), np.array(getattr(trace.posterior,'log_width2')), np.array(getattr(trace.posterior,'rho'))
        taucoef, log_width2, rho = taucoef_0.reshape(sh[0]*sh[1],sh[2]), log_width2_0.reshape(sh[0]*sh[1]), rho_0.reshape(sh[0]*sh[1])
    else:
        taucoef, log_width2, rho = None, None, None
    return ncoef, log_width, taucoef, log_width2, rho

def getModelSamples(xx, indep_samp, ngrid, log_width, taugrid, log_width2, rho, numsamp=50, poly_only=False):
    npts = len(log_width)
    dg = len(xx[0])-1
    inds = np.random.choice(npts,size=numsamp,replace=False)
    sh = list(indep_samp[0].shape)
    sh.insert(0,numsamp)
    n_sim = np.empty(tuple(sh))
    if taugrid is not None: tau_sim = np.empty(tuple(sh))
    for i, ind in enumerate(inds):
        ngrid_mod, width_mod = ngrid[ind], np.exp(log_width[ind])
        coefs_mod = polyfitnd(xx,ngrid_mod)
        r1, r2 = np.random.randn(*n_sim[i].shape), np.random.randn(*n_sim[i].shape)
        if poly_only: n_sim[i] = calc_poly(indep_samp,coefs_mod,dg)
        else: n_sim[i] = calc_poly(indep_samp,coefs_mod,dg) + width_mod * r1
        if taugrid is not None: 
            taugrid_mod, width2_mod, rho_mod = taugrid[ind], np.exp(log_width2[ind]), rho[ind]
            coefs2_mod = polyfitnd(xx,taugrid_mod)
            if poly_only: tau_sim[i] = calc_poly(indep_samp,coefs2_mod,dg)
            else: tau_sim[i] = calc_poly(indep_samp,coefs2_mod,dg) + width2_mod * (rho_mod*r1 + np.sqrt(1.0-rho_mod**2)*r2)
        else: tau_sim = None
    return n_sim, tau_sim

def getModelSamplesCD(indep_samp, ind_just_right, ncoef, log_width, taucoef, log_width2, rho, numsamp=50, poly_only=False,dg=2):
    npts = len(log_width)
    inds = np.random.choice(npts,size=numsamp,replace=False)
    sh = list(indep_samp[0].shape)
    totlen = (dg+1)**len(indep_samp)
    sh.insert(0,numsamp)
    n_sim = np.empty(tuple(sh))
    if taucoef is not None: tau_sim = np.empty(tuple(sh))
    for i, ind in enumerate(inds):
        ncoef_mod, width_mod = ncoef[ind], np.exp(log_width[ind])
        coefs_mod = np.zeros(totlen)
        coefs_mod[ind_just_right] = ncoef_mod
        r1, r2 = np.random.randn(*n_sim[i].shape), np.random.randn(*n_sim[i].shape)
        if poly_only: n_sim[i] = calc_poly(indep_samp,coefs_mod,dg)
        else: n_sim[i] = calc_poly(indep_samp,coefs_mod,dg) + width_mod * r1
        if taucoef is not None: 
            taucoef_mod, width2_mod, rho_mod = taucoef[ind], np.exp(log_width2[ind]), rho[ind]
            coefs2_mod = np.zeros(totlen)
            coefs2_mod[ind_just_right] = taucoef_mod
            if poly_only: tau_sim[i] = calc_poly(indep_samp,coefs2_mod,dg)
            else: tau_sim[i] = calc_poly(indep_samp,coefs2_mod,dg) + width2_mod * (rho_mod*r1 + np.sqrt(1.0-rho_mod**2)*r2)
        else: tau_sim = None
    return n_sim, tau_sim

def getModelSamplesI(xtup, indep_samp, ngrid, log_width, taugrid, log_width2, rho, numsamp=50, poly_only=False, make_breakpt=False):
    npts = len(log_width)
    if indep_samp.ndim == 2: indepI = indep_samp.T
    else: indepI = indep_samp.reshape(len(indep_samp),np.prod(indep_samp.shape[1:])).T
    inds = np.random.choice(npts,size=numsamp,replace=False)
    sh = list(indep_samp[0].shape)
    sh.insert(0,numsamp)
    n_sim = np.empty(tuple(sh))
    if taugrid is not None: tau_sim = np.empty(tuple(sh))
    if make_breakpt: breakpoint()
    for i, ind in enumerate(inds):
        ngrid_mod, width_mod = ngrid[ind], np.exp(log_width[ind])
        r1, r2 = np.random.randn(*n_sim[i].shape), np.random.randn(*n_sim[i].shape)
        interp_part = regular_grid_interp_scipy(xtup, ngrid_mod, indepI).reshape(n_sim[0].shape)
        if poly_only: n_sim[i] = interp_part
        else: n_sim[i] = interp_part + width_mod * r1
        if taugrid is not None: 
            taugrid_mod, width2_mod, rho_mod = taugrid[ind], np.exp(log_width2[ind]), rho[ind]
            interp_part = regular_grid_interp_scipy(xtup, taugrid_mod, indepI).reshape(tau_sim[0].shape)
            if poly_only: tau_sim[i] = interp_part
            else: tau_sim[i] = interp_part + width2_mod * (rho_mod*r1 + np.sqrt(1.0-rho_mod**2)*r2)
        else: tau_sim = None
    return n_sim, tau_sim

def make2DGrid(xx, i, j, med, obj, indfin, proplist, numsamples=10, sliced=True, fine_grid=201, coef_direct=False, xxI=None):
    ndim = len(xx)
    axrange = tuple(range(1,xx[0].ndim+1))
    if coef_direct: mins, maxs = np.percentile(xx,[2.5,97.5],axis=axrange)
    else: mins, maxs = np.amin(xx,axis=axrange), np.amax(xx,axis=axrange)
    if xxI is not None: 
        axrange = tuple(range(1,xxI[0].ndim+1))
        mins, maxs = np.amin(xxI,axis=axrange), np.amax(xxI,axis=axrange)
    indep_fine = np.zeros((ndim,fine_grid))
    indep_fine[i] = np.linspace(mins[i],maxs[i],fine_grid)
    indep_fine[j] = np.linspace(mins[j],maxs[j],fine_grid)
    if ndim>2:
        ind_all = np.arange(ndim)
        ind_not_ij = ind_all[np.isin(ind_all,[i,j],invert=True)]
        if not sliced: indep_fine[ind_not_ij] = dac.marg_by_post(obj, indfin, np.array(proplist)[ind_not_ij], med[ind_not_ij], xx, numsamples, fine_grid)
    xx_fine = np.meshgrid(*indep_fine[(i,j),:])
    for k in range(i-1,-1,-1): xx_fine.insert(0,np.zeros_like(xx_fine[0]))
    for k in range(i+1,j): xx_fine.insert(-1,np.zeros_like(xx_fine[0]))
    for k in range(j+1,ndim): xx_fine.insert(k,np.zeros_like(xx_fine[0]))
    xx_fine = np.array(xx_fine)
    if not sliced and ndim>2: xx_fine[ind_not_ij] = dac.marg_by_post(obj, indfin, np.array(proplist)[ind_not_ij], med[ind_not_ij], xx, numsamples, fine_grid**2).reshape(ndim-2,fine_grid,fine_grid)
    return indep_fine, np.array(xx_fine), mins, maxs

def projPlot(ax,x,ylist,zfull,zfullerr,xlab,ylab,zlab,axis='left'):
    win_len, polyorder = len(x)//3, 1
    if win_len%2==0: win_len+=1
    rangex = np.amax(x)-np.amin(x)
    for ii in range(len(ylist)):
        zsav = savgol_filter(zfull[ii],win_len,polyorder)
        zpesav = savgol_filter(zfull[ii]+zfullerr[ii],win_len,polyorder)
        zmesav = savgol_filter(zfull[ii]-zfullerr[ii],win_len,polyorder)
        ax.plot(x,zsav)
        x_annot = 0.86*ii/len(ylist)*rangex + np.amin(x)
        x_indannot = np.argmin(abs(x-x_annot))
        c = ax.lines[-1].get_color()
        ax.plot(x,zfull[ii],color=c,linestyle='none',marker=',')
        ax.annotate(r"%s$=%.2f$"%(ylab,ylist[ii]),(x_annot,zsav[x_indannot]),color=c,fontsize='x-small')
        ax.fill_between(x,zmesav,zpesav,color=c,alpha=0.1)
    ax.set_xlim([np.amin(x),np.amax(x)])
    # ax.set_ylim([np.amin(zfull),np.amax(zfull)])
    if zlab==nlab: ax.set_ylim([-1.0,0.4])
    else: ax.set_ylim([0.0,2.5])
    ax.set_xlabel(xlab)
    ax.set_ylabel(zlab)
    if axis=='right':
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('both')

def makeColorMapProj(trace, xx, med, indep_lab, indep_name, indep_pickle_name, dataname, img_dir, bivar=False, poly_only=False, numsamp=50, sliced=True, fine_grid=201, numslices=5, numsamples=10, numgal=None, coef_direct=False, ind_just_right=None, dg=2, indep=None, dep_name='n'):
    if dep_name == 'n': dep_lab, dep_lims = nlab, nlims
    else: dep_lab, dep_lims = taulab, taulims
    obj, indfin = dac.getProspDataBasic(numsamples, numgal)
    if coef_direct: ngrid, log_width, taugrid, log_width2, rho = getTraceInfoCD(trace, bivar)
    else: ngrid, log_width, taugrid, log_width2, rho = getTraceInfo(trace, bivar)
    ndim = len(indep_name)
    for i in range(ndim):
        if coef_direct: xxi=None
        else: xxi = xx[i].ravel()+med[i]
        if indep is None: 
            if coef_direct: indepi, indepierr = np.average(xx[i],axis=1), np.std(xx[i],axis=1)
            else: indepi, indepierr = None, None
        else: indepi, indepierr = indep[i], None
        for j in range(i+1,ndim):
            if coef_direct: xxj=None
            else: xxj = xx[j].ravel()+med[j]
            if indep is None: 
                if coef_direct: indepj, indepjerr = np.average(xx[j],axis=1), np.std(xx[j],axis=1)
                else: indepj, indepjerr = None, None
            else: indepj, indepjerr = indep[j], None
            fig, ax = plt.subplots(1,3,figsize=fig_size)
            if bivar: fig2, ax2 = plt.subplots(1,3,figsize=fig_size)
            indep_fine, xx_fine, mins, maxs = make2DGrid(xx, i, j, med, obj, indfin, indep_pickle_name, numsamples=numsamples, sliced=sliced, fine_grid=fine_grid, coef_direct=coef_direct)
            if coef_direct: n_sim, tau_sim = getModelSamplesCD(xx_fine, ind_just_right, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only, dg)
            else: n_sim, tau_sim = getModelSamples(xx, xx_fine, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only)
            n_mean, n_err = np.mean(n_sim,axis=0), np.std(n_sim,axis=0)

            cf2, xlim, ylim = dac.plot_color_map_ind(ax[1],dep_lims[0],dep_lims[1],xx_fine[i]+med[i],xx_fine[j]+med[j],n_mean,xxi,xxj,xlab=indep_lab[i],ylab=indep_lab[j],xtrue=indepi,ytrue=indepj,xtrueerr=indepierr,ytrueerr=indepjerr)
            fig.colorbar(cf2,ax=ax[1],location='top',label=dep_lab)
            if bivar:
                tau_mean, tau_err = np.mean(tau_sim,axis=0), np.std(tau_sim,axis=0)
                cf22, _, _ = dac.plot_color_map_ind(ax2[1],taulims[0],taulims[1],xx_fine[i]+med[i],xx_fine[j]+med[j],tau_mean,xxi,xxj,xlab=indep_lab[i],ylab=indep_lab[j],xtrue=indepi,ytrue=indepj,xtrueerr=indepierr,ytrueerr=indepjerr)
                fig2.colorbar(cf22,ax=ax2[1],location='top',label=taulab)

            xlist = np.linspace(mins[i],maxs[i],numslices)
            ylist = np.linspace(mins[j],maxs[j],numslices)
            xx_proj1, xx_proj2 = deepcopy(indep_fine), deepcopy(indep_fine)
            n_sim_proj1, tau_sim_proj1 = np.empty((numslices,numsamp,fine_grid)), np.empty((numslices,numsamp,fine_grid))
            n_sim_proj2, tau_sim_proj2 = np.empty((numslices,numsamp,fine_grid)), np.empty((numslices,numsamp,fine_grid))
            for ii in range(numslices):
                xx_proj1[i] = xlist[ii]*np.ones(fine_grid)
                xx_proj2[j] = ylist[ii]*np.ones(fine_grid)
                if coef_direct: 
                    n_sim_proj1[ii], tautemp = getModelSamplesCD(xx_proj1, ind_just_right, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only, dg)
                    n_sim_proj2[ii], tautemp2 = getModelSamplesCD(xx_proj2, ind_just_right, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only, dg)
                else:
                    n_sim_proj1[ii], tautemp = getModelSamples(xx, xx_proj1, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only)
                    n_sim_proj2[ii], tautemp2 = getModelSamples(xx, xx_proj2, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only)
                if bivar: tau_sim_proj1[ii], tau_sim_proj2[ii] = tautemp, tautemp2
            nproj1_mean, nproj1_err, nproj2_mean, nproj2_err = np.average(n_sim_proj1,axis=1), np.std(n_sim_proj1,axis=1), np.average(n_sim_proj2,axis=1), np.std(n_sim_proj2,axis=1)
            if bivar: tauproj1_mean, tauproj1_err, tauproj2_mean, tauproj2_err = np.average(tau_sim_proj1,axis=1), np.std(tau_sim_proj1,axis=1), np.average(tau_sim_proj2,axis=1), np.std(tau_sim_proj2,axis=1)
            # breakpoint()
            projPlot(ax[0],indep_fine[i]+med[i],ylist+med[j],nproj2_mean,nproj2_err,indep_lab[i],indep_lab[j],dep_lab)
            projPlot(ax[2],indep_fine[j]+med[j],xlist+med[i],nproj1_mean,nproj1_err,indep_lab[j],indep_lab[i],dep_lab,axis='right')
            fig.savefig(op.join(img_dir,'cmp_%s_%s_vs_%s_dep_%s_po_%d_sl_%d_proj_%d.png'%(dataname,indep_name[j],indep_name[i],dep_name,poly_only,sliced,numslices)),bbox_inches='tight',dpi=300)
            if bivar:
                projPlot(ax2[0],indep_fine[i]+med[i],ylist+med[j],tauproj2_mean,tauproj2_err,indep_lab[i],indep_lab[j],taulab)
                projPlot(ax2[2],indep_fine[j]+med[j],xlist+med[i],tauproj1_mean,tauproj1_err,indep_lab[j],indep_lab[i],taulab,axis='right')
                fig2.savefig(op.join(img_dir,'cmp_%s_%s_vs_%s_dep_d2_po_%d_sl_%d_proj_%d.png'%(dataname,indep_name[j],indep_name[i],poly_only,sliced,numslices)),bbox_inches='tight',dpi=300)

def makeColorMapProjI(trace, xtup, indep_samp, indep_lab, indep_name, indep_pickle_name, dataname, img_dir, bivar=False, poly_only=False, numsamp=50, sliced=True, fine_grid=201, numslices=5, numsamples=10, numgal=None, dep_name='n',extrawords=''):
    if dep_name == 'n': dep_lab, dep_lims = nlab, nlims
    else: dep_lab, dep_lims = taulab, taulims
    obj, indfin = dac.getProspDataBasic(numsamples, numgal)
    ngrid, log_width, taugrid, log_width2, rho = getTraceInfo(trace, bivar)
    ndim = len(indep_name)
    indep, indep_err = np.mean(indep_samp,axis=2), np.std(indep_samp,axis=2)
    xx_full = np.meshgrid(*xtup)
    for i in range(ndim):
        for j in range(i+1,ndim):
            fig, ax = plt.subplots(1,3,figsize=fig_size)
            if bivar: fig2, ax2 = plt.subplots(1,3,figsize=fig_size)
            indep_fine, xx_fine, mins, maxs = make2DGrid(indep_samp, i, j, np.zeros(len(indep_samp)), obj, indfin, indep_pickle_name, numsamples=numsamples, sliced=sliced, fine_grid=fine_grid, xxI=xx_full)
            n_sim, tau_sim = getModelSamplesI(xtup, xx_fine, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only)
            n_mean, n_err = np.mean(n_sim,axis=0), np.std(n_sim,axis=0)

            cf2, xlim, ylim = dac.plot_color_map_ind(ax[1],dep_lims[0],dep_lims[1],xx_fine[i],xx_fine[j],n_mean,xx_full[i].ravel(),xx_full[j].ravel(),xlab=indep_lab[i],ylab=indep_lab[j],xtrue=indep[i],ytrue=indep[j],xtrueerr=indep_err[i],ytrueerr=indep_err[j],xx_size=1)
            fig.colorbar(cf2,ax=ax[1],location='top',label=dep_lab)
            if bivar:
                tau_mean, tau_err = np.mean(tau_sim,axis=0), np.std(tau_sim,axis=0)
                cf22, _, _ = dac.plot_color_map_ind(ax2[1],taulims[0],taulims[1],xx_fine[i],xx_fine[j],tau_mean,xx_full[i].ravel(),xx_full[j].ravel(),xlab=indep_lab[i],ylab=indep_lab[j],xtrue=indep[i],ytrue=indep[j],xtrueerr=indep_err[i],ytrueerr=indep_err[j],xx_size=2)
                fig2.colorbar(cf22,ax=ax2[1],location='top',label=taulab)

            xlist = np.linspace(mins[i],maxs[i],numslices)
            ylist = np.linspace(mins[j],maxs[j],numslices)
            xx_proj1, xx_proj2 = deepcopy(indep_fine), deepcopy(indep_fine)
            n_sim_proj1, tau_sim_proj1 = np.empty((numslices,numsamp,fine_grid)), np.empty((numslices,numsamp,fine_grid))
            n_sim_proj2, tau_sim_proj2 = np.empty((numslices,numsamp,fine_grid)), np.empty((numslices,numsamp,fine_grid))
            for ii in range(numslices):
                xx_proj1[i] = xlist[ii]*np.ones(fine_grid)
                xx_proj2[j] = ylist[ii]*np.ones(fine_grid)
                n_sim_proj1[ii], tautemp = getModelSamplesI(xtup, xx_proj1, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only)
                n_sim_proj2[ii], tautemp2 = getModelSamplesI(xtup, xx_proj2, ngrid, log_width, taugrid, log_width2, rho, numsamp, poly_only)
                if bivar: tau_sim_proj1[ii], tau_sim_proj2[ii] = tautemp, tautemp2
            nproj1_mean, nproj1_err, nproj2_mean, nproj2_err = np.average(n_sim_proj1,axis=1), np.std(n_sim_proj1,axis=1), np.average(n_sim_proj2,axis=1), np.std(n_sim_proj2,axis=1)
            if bivar: tauproj1_mean, tauproj1_err, tauproj2_mean, tauproj2_err = np.average(tau_sim_proj1,axis=1), np.std(tau_sim_proj1,axis=1), np.average(tau_sim_proj2,axis=1), np.std(tau_sim_proj2,axis=1)
            # breakpoint()
            projPlot(ax[0],indep_fine[i],ylist,nproj2_mean,nproj2_err,indep_lab[i],indep_lab[j],dep_lab)
            projPlot(ax[2],indep_fine[j],xlist,nproj1_mean,nproj1_err,indep_lab[j],indep_lab[i],dep_lab,axis='right')
            fig.savefig(op.join(img_dir,'cmp_%s_%s_vs_%s_dep_%s_po_%d_sl_%d_proj_%d%s.png'%(dataname,indep_name[j],indep_name[i],dep_name,poly_only,sliced,numslices,extrawords)),bbox_inches='tight',dpi=300)
            if bivar:
                projPlot(ax2[0],indep_fine[i],ylist,tauproj2_mean,tauproj2_err,indep_lab[i],indep_lab[j],taulab)
                projPlot(ax2[2],indep_fine[j],xlist,tauproj1_mean,tauproj1_err,indep_lab[j],indep_lab[i],taulab,axis='right')
                fig2.savefig(op.join(img_dir,'cmp_%s_%s_vs_%s_dep_d2_po_%d_sl_%d_proj_%d%s.png'%(dataname,indep_name[j],indep_name[i],poly_only,sliced,numslices,extrawords)),bbox_inches='tight',dpi=300)

def main():
    args = dac.parse_args()
    img_dir = op.join('ColorMapProj',args.dir_orig,'deg_%d'%(args.degree),args.dataname)
    mkpath(img_dir)
    trace, xx, med = dac.getPostModelData(args, img_dir_orig=op.join('DataFinal',args.dir_orig))
    makeColorMapProj(trace, xx, med, args.indep_lab, args.indep_name, args.indep_pickle_name, args.dataname, img_dir, bivar=args.bivar, poly_only=args.poly, sliced=args.sliced, numslices=args.nproj)

if __name__=='__main__':
    main()