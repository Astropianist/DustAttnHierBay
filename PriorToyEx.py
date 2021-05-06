import numpy as np 
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import corner
from scipy.stats import multivariate_normal
from scipy.integrate import trapz
from scipy.interpolate import interp2d, LinearNDInterpolator
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

min_pdf = 1.0e-30 # To make sure prior log likelihoods are not NaNs
N_prior=100000 # Number of prior samples for effective sample
alpha = multivariate_normal(mean=[0.0,0.0,0.0],cov=np.array([[1.8,0.1,0.2],[0.3,2.1,0.4],[0.2,0.1,1.7]])) # Multivariate normal distribution for intermediate priors (3 variables: x,y,z; y is an additional variable in the lower-level code that isn't involved at all in the hierarchical model)

def vf(x):
    ''' A variable used for the hierarchical model that is a function of one of the variables in the lower-level code '''
    return np.sin(np.pi*x)+4.0*x-0.5
    # return 2.0*x**3+1.0
    # return 2.0*x+1.0

def xf(v): 
    ''' Inverse of variable v(x) above '''
    return v # There is no analytic form for this one; we don't use this function in any case
    # return np.power(0.5*v-0.5,0.3333333333333333)
    # return 0.5*v-0.5

def jac(x): 
    ''' Determinant of Jacobian matrix from (x,z) -> (v,z) space '''
    return np.pi*np.cos(np.pi*x)+4.0
    # return 6.0*x**2 
    # return 2.0

def jac_v(v):
    return 0.1666667*np.power(0.5*v-0.5,0.666666666666667)

def toy_data(size=500,samples=50,steps=1000,tune=1000,sigv=0.02,sigx=0.01,sigy=0.008,sigz=0.01,a_true=1.2,b_true=-0.5,width_true=0.01):
    """ Function to create mock data; currently x and y are picked from the intermediate prior distribution (an extreme case where the ground-level data has very large uncertainties); the other option is that they are chosen from uniform or normal distributions """
    # True linear model
    samp = alpha.rvs(size) # Multivariate normal distribution (intermediate prior)
    # x_true, y_true = samp[:,0],samp[:,1] # Just considering x and y
    x_true, y_true = 4.0*np.random.rand(size)-2.0, 2.0*np.random.rand(size)-1.0
    v_true = vf(x_true) #v is an analytic function of x
    # v_true = np.random.uniform(-5.0,5.0,size)
    # x_true = xf(v_true)
    # y_true = 2.0*np.random.randn(size)+1.0
    z_true = a_true*v_true + b_true + width_true*np.random.randn(size) # z is a known function of v, which is a function of x; also some intrinsic dispersion given by width_true

    # Define posterior 'samples' in same way as always
    # v_samp = v_true[:,None] + sigv*np.random.randn(size,1) + sigv*np.random.randn(size,samples)
    # x_samp = xf(v_samp)
    x_samp = x_true[:,None] + sigx*np.random.randn(size,1) + sigx*np.random.randn(size,samples)
    v_samp = vf(x_samp)
    y_samp = y_true[:,None] + sigy*np.random.randn(size,1) + sigy*np.random.randn(size,samples)
    z_samp = z_true[:,None] + sigz*np.random.randn(size,1) + sigz*np.random.randn(size,samples)

    return v_samp,x_samp,y_samp,z_samp

def toy_model(v_samp,z_samp,logp_prior,size=500,samples=50,steps=1000,tune=1000,a_true=1.2,b_true=-0.5,width_true=0.05,extratext='_true'):
    ''' The pymc3 linear model z(v) = a*v+b with natural width in log space log_width. The prior contribution to likelihood is an argument to the function.'''
    with pm.Model() as model:
        a = pm.Normal("a", mu=0, sigma=10, testval=a_true)
        b = pm.Normal("b", mu=0, sigma=10, testval=b_true)
        log_width = pm.Normal("log_width", mu=np.log(width_true), sigma=2.0, testval=np.log(width_true))
        
        mu = a*v_samp + b

        # The line has some width: we're calling it a Gaussian in n
        logp_hyper = -0.5 * (z_samp - mu) ** 2 * pm.math.exp(-2*log_width) - log_width
        # Here we account for the intermediate prior
        logp = logp_hyper - logp_prior

        # Compute the marginalized likelihood
        max_logp = tt.max(logp, axis=1)
        # max_logp = np.zeros(len(logM_samp))
        marg_logp = max_logp + pm.math.log(pm.math.sum(pm.math.exp(logp - max_logp[:, None]), axis=1))
        pm.Potential('marg_logp', marg_logp)

        trace = pm.sample(draws=steps,tune=tune,target_accept=0.9,init='adapt_full', return_inferencedata=False)
        # az.plot_trace(trace)
        print(az.summary(trace,round_to=2))
        print(a_true, b_true, np.log(width_true))
        corner.corner(pm.trace_to_dataframe(trace), truths=[a_true] + [b_true] + [np.log(width_true)]) # Corner plot!
        plt.savefig("PriorToy/Corner_N1000_vfcomplex_prior_samp_mixed%s.png"%(extratext),bbox_inches='tight',dpi=150)
    return

def estimate_prior_orig(N_prior=N_prior,bins=100):
    """ Estimate the joint prior alpha(v,z) from the true prior alpha(x,y,z) using N_prior samples """
    samp = alpha.rvs(N_prior) # Sample from the distribution
    v_samp = vf(samp[:,0])
    z_samp = samp[:,2]
    nums = np.zeros((bins,bins)) # To calculate the joint pdf of the samples
    xdiv = np.linspace(min(v_samp),max(v_samp)+1.0e-8,bins+1) # Bin edges for v_samp
    ydiv = np.linspace(min(z_samp),max(z_samp)+1.0e-8,bins+1) # Bin edges for z_samp
    xcen = np.linspace((xdiv[0]+xdiv[1])/2.0,(xdiv[-1]+xdiv[-2])/2.0,bins) # Bin centers for v_samp
    ycen = np.linspace((ydiv[0]+ydiv[1])/2.0,(ydiv[-1]+ydiv[-2])/2.0,bins) # Bin centers for z_samp
    # Determine the number of samples in each bin
    for k in range(bins):
        condy = np.logical_and(z_samp>=ydiv[k],z_samp<ydiv[k+1])
        for l in range(bins):
            condx = np.logical_and(v_samp>=xdiv[l],v_samp<xdiv[l+1])
            cond = np.logical_and(condx,condy)
            nums[k,l] = len(v_samp[cond])
    # Normalize the pdf using the double integral over the binning grid
    # nums[nums<=0.0]=min_pdf # We don't want any pdf values to be exactly zero as we will be taking the log
    integ = trapz(trapz(nums,xcen),ycen)
    nums /= integ
    xx,yy = np.meshgrid(xcen,ycen) # Defining the v_samp, z_samp grid
    pdf = LinearNDInterpolator(np.column_stack((xx.ravel(),yy.ravel())),nums.ravel(),fill_value=0.0) # Using Scipy's LinearNDInterpolator to create a function for the pdf
    return pdf

def estimate_prior(N_prior=N_prior,bins=100):
    """ Estimate the joint prior alpha(v,z) from the true prior alpha(x,y,z) using N_prior samples """
    samp = alpha.rvs(N_prior) # Sample from the distribution
    x_samp = samp[:,0]
    v_samp = vf(x_samp)
    y_samp = samp[:,1]
    z_samp = samp[:,2]
    # breakpoint()
    kde = KernelDensity(bandwidth=0.2,algorithm='auto',atol=1.0e-5,rtol=1.0e-5)
    kde2 = KernelDensity(bandwidth=0.2,algorithm='auto',atol=1.0e-5,rtol=1.0e-5)
    kde3 = KernelDensity(bandwidth=0.2,algorithm='auto',atol=1.0e-5,rtol=1.0e-5)
    kde.fit(np.column_stack((v_samp,z_samp)))
    kde2.fit(np.column_stack((v_samp,y_samp,z_samp)))
    kde3.fit(np.column_stack((x_samp,y_samp,z_samp)))
    return kde, kde2, kde3

def toy_true(v_samp,x_samp,y_samp,z_samp,size=500,samples=50,steps=1000,tune=1000,a_true=1.2,b_true=-0.5,width_true=0.05):
    """ Modeling z=a*v+b when including the true intermediate prior alpha(x,y,z); y is a parameter in the lower-level model that has no involvement in the hierarchical model. """
    all_samp = np.transpose(np.array([x_samp,y_samp,z_samp]),axes=(1,2,0))
    prior_vals = alpha.pdf(all_samp)
    # breakpoint()
    # prior_vals[prior_vals<=0.0]=min_pdf
    logp_prior = np.log(prior_vals) - np.log(jac(x_samp)) # Try this Jacobian term
    # print("toy true:"); print(logp_prior)
    # toy_model(v_samp,z_samp,logp_prior,size=size,samples=samples,steps=steps,tune=tune,a_true=a_true,b_true=b_true,width_true=width_true,extratext='_true')
    return logp_prior

def toy_approx(v_samp,x_samp,y_samp,z_samp,size=500,samples=50,steps=1000,tune=1000,a_true=1.2,b_true=-0.5,width_true=0.05,N_prior=100000):
    """ Modeling z=a*v+b when including an approximation of the intermediate prior based on the variables at hand alpha(v,z) """
    # prior_pdf = estimate_prior()
    # logp_prior = np.log(prior_pdf(v_samp,z_samp))
    kde, kde2, kde3 = estimate_prior()
    # print("toy approx:"); print(logp_prior)
    # toy_model(v_samp,z_samp,logp_prior,size=size,samples=samples,steps=steps,tune=tune,a_true=a_true,b_true=b_true,width_true=width_true,extratext='_approx')
    return kde.score_samples(np.column_stack((v_samp.ravel(),z_samp.ravel()))).reshape(v_samp.shape), kde2.score_samples(np.column_stack((v_samp.ravel(),y_samp.ravel(),z_samp.ravel()))).reshape(v_samp.shape), kde3.score_samples(np.column_stack((x_samp.ravel(),y_samp.ravel(),z_samp.ravel()))).reshape(v_samp.shape)

def plot_prior(v,x,y,z,lpa,lpt,lpt2,lpt3,extratext=''):
    var_names = ['v','x','y','z']
    for var, var_name in zip([v,x,y,z],var_names):
        fig = plt.figure(dpi=200)
        per = np.percentile(var,[16.0,84.0])
        cond = np.logical_and(var>=per[0],var<=per[1])
        pery = np.percentile(lpt[cond][np.isfinite(lpt[cond])],[5.0,99.5])
        pery2 = np.percentile(lpa[cond][np.isfinite(lpa[cond])],[5.0,99.5])
        pery3 = np.percentile(lpt2[cond][np.isfinite(lpt2[cond])],[5.0,99.5])
        pery4 = np.percentile(lpt3[cond][np.isfinite(lpt3[cond])],[5.0,99.5])
        print('pery',pery)
        plt.plot(var[cond],lpa[cond],'b.',label='PDF in v,z',alpha=0.5,markersize=10./fig.dpi)
        plt.plot(var[cond],lpt[cond],'r.',label='PDF in v,y,z',alpha=0.5,markersize=10./fig.dpi)
        plt.plot(var[cond],lpt2[cond],'k.',label='PDF in x,y,z (No Jacobian)',alpha=0.5,markersize=10./fig.dpi)
        plt.plot(var[cond],lpt3[cond],'c.',label='PDF in x,y,z (Jacobian)',alpha=0.5,markersize=10./fig.dpi)
        plt.gca().set_xlim(per[0],per[1])
        plt.gca().set_ylim(min(pery[0],pery2[0],pery3[0],pery4[0]),max(pery[1],pery2[1],pery3[1],pery4[1]))
        plt.xlabel(var_name)
        plt.ylabel('ln prior')
        plt.legend(loc='best',frameon=False,markerscale=80,fontsize='x-small')
        plt.savefig('PriorToy/Final/%s_KDEcomp_%s.png'%(var_name,extratext),bbox_inches='tight',dpi=200)
        plt.close()

if __name__=='__main__':
    size=1000
    extratext = input('What is extratext for this run? ')
    v_samp,x_samp,y_samp,z_samp = toy_data(size=size,sigx=0.001,sigy=0.001,sigz=0.001)
    logp_prior_true = toy_true(v_samp,x_samp,y_samp,z_samp,size=size)
    logp_prior_approx, logp_prior_approx2, logp_prior_approx3 = toy_approx(v_samp,x_samp,y_samp,z_samp,size=size)
    plot_prior(v_samp,x_samp,y_samp,z_samp,logp_prior_approx,logp_prior_approx2,logp_prior_approx3,logp_prior_true,extratext=extratext)