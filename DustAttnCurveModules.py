import numpy as np 
import os.path as op
from sedpy.attenuation import noll
import arviz as az
import argparse as ap
from glob import glob
from astropy.table import Table
import pickle
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

wv_arr = np.linspace(1500.0,5000.0,501)
indepext = np.array(['m','s','logZ','z','i','d1','d2'])

def parse_args(argv=None):
    """ Tool to parse arguments from the command line. The entries should be self-explanatory """
    parser = ap.ArgumentParser(description="DustAttnCurves",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-m','--logM',help='Whether or not to include stellar mass in model',action='count',default=0)
    parser.add_argument('-s','--ssfr',help='Whether or not to include sSFR in model',action='count',default=0)
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
    if args.bivar: args.dir_orig = 'Bivar_Sample'
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

def make_prop_dict():
    prop_dict, dep_dict = {}, {}
    prop_dict['names'] = {'logM': 'logM', 'ssfr': 'logsSFR', 'logZ': 'logZ', 'z': 'z', 'i':'axis_ratio', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['names'] = {'tau2': 'dust2', 'n': 'n'}
    prop_dict['labels'] = {'logM': r'$\log M_*$', 'ssfr': r'$\log$ sSFR$_{\rm{100}}$', 'logZ': r'$\log (Z/Z_\odot)$', 'z': 'z', 'i':r'$b/a$', 'd1':r"$\hat{\tau}_{1}$", 'd2':r"$\hat{\tau}_{2}$"}
    dep_dict['labels'] = {'tau2': r"$\hat{\tau}_{2}$", 'n': 'n'}
    prop_dict['pickle_names'] = {'logM': 'stellar_mass', 'ssfr': 'ssfr_100', 'logZ': 'log_z_zsun', 'z': 'z', 'i':'inc', 'd1':'dust1', 'd2':'dust2'}
    dep_dict['pickle_names'] = {'tau2': 'dust2', 'n': 'dust_index'}
    return prop_dict, dep_dict

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

def getPostModelData(args,img_dir_orig=op.join('DataFinal','Bivar_Sample'),run_type=''):
    if run_type=='': globsearch = op.join(img_dir_orig,'deg_%d'%(args.degree),'%s*'%(args.dataname),'*.nc')
    else: globsearch = op.join(img_dir_orig,'%s*'%(args.extratext),'*.nc')
    nclist = glob(globsearch)
    if len(nclist)==0:
        print("No netcdf file found in glob search",globsearch)
        return
    if len(nclist)>1:
        print("Multiple files or directories found in search",globsearch)
        return
    trace = az.from_netcdf(nclist[0])
    print("Trace file:", nclist[0])
    if run_type!='I': globsearch = op.join(img_dir_orig,'deg_%d'%(args.degree),'%s*'%(args.dataname),'*.dat')
    else: globsearch = op.join(img_dir_orig,'%s*'%(args.extratext),'*.dat')
    datf = glob(globsearch)[0]
    dat = Table.read(datf,format='ascii')
    print("Dat file:",datf)
    ndim = len(args.indep_name)
    if run_type=='':
        sh = tuple([ndim]+[args.degree+1]*ndim)
        xx = np.empty((ndim,(args.degree+1)**ndim))
        med = np.empty(ndim)
        for i, name in enumerate(args.indep_name):
            xx[i] = dat[name]
            med[i] = dat[name+'_plus_med'][0]-dat[name][0]    
        return trace, xx.reshape(sh), med
    else:
        name_0 = args.indep_name[0]
        temp = np.unique(dat[name_0])
        grid_len = temp.size
        print("Measured grid length from file:", grid_len)
        x = np.empty((ndim,grid_len))
        for i, name in enumerate(args.indep_name):
            x[i] = np.unique(dat[name])
        return trace, tuple(x)

def plot_color_map_ind(ax,vmin,vmax,x,y,z,xx,yy,levels=10,xlab=r'$\log\ M_*$',ylab=r'$\log$ sSFR$_{\rm{100}}$',xtrue=None,ytrue=None,xtrueerr=None,ytrueerr=None,minz=-100.0,maxz=100.0,width_mean=None,xx_size=4):
    if xx is not None:
        if xx_size==4: marker='o'
        else: marker='.'
        xmin, xmax = np.amin(xx),np.amax(xx)
        ymin, ymax = np.amin(yy),np.amax(yy)
        ax.plot(xx,yy,f'r{marker}',markersize=xx_size)
    else:
        xmin, xmax = np.amin(x),np.amax(x)
        ymin, ymax = np.amin(y),np.amax(y)
    cf = ax.contourf(x,y,z,levels=levels,cmap='viridis',vmin=vmin,vmax=vmax)
    if xtrue is not None: 
        ax.plot(xtrue,ytrue,'k,',alpha=1.0)
    if xtrueerr is not None:
        ax.errorbar(xtrue,ytrue,yerr=ytrueerr,xerr=xtrueerr,fmt='none',ecolor='k',alpha=2/xtrue.size)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    return cf, ax.get_xlim(), ax.get_ylim()