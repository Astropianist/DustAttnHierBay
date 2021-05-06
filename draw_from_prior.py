import numpy as np
import threedhst_params as pfile
import pickle
import MakePriorHist as M

def main(ndraw=500000,zred=0.5, **args):

    # generate physical model
    obs = pfile.load_obs(**pfile.run_params)
    mod = pfile.load_model(zred=zred,**pfile.run_params)
    sps = pfile.load_sps(**pfile.run_params)

    # print some diagnostic information
    print(mod)
    print(mod.free_params)
    print(mod.fixed_params)
    print(mod.theta_index)

    # pre-compute some useful quantities for nonparametric SFH
    agebins = mod.params['agebins']            # time bins for the SFH, useful later
    mean_age = (10**agebins).mean(axis=1)/1e9  # in Gyr

    # define output; here we sample for mass-weighted age and sSFR
    out = {name:[] for name in ['ssfr', 'mwa', 'stmass','dust1','dust2','met']}
    duration_of_star_formation = np.zeros(ndraw)
    for i in range(ndraw):

        # for each model parameter, sample from prior
        theta = np.zeros(mod.ndim)
        for k, inds in list(mod.theta_index.items()):
            func = mod._config_dict[k]['prior']
            kwargs = mod._config_dict[k].get('prior_args', {})
            theta[inds] = func.sample(**kwargs)

        # now we pass the theta vector to FSPS
        spec, mags, stellar_to_total = mod.predict(theta,obs=obs,sps=sps)

        # now we transform from model parameters to derived quantities
        # first, calculate mass in each time bin
        massmet = theta[mod.theta_index['massmet']]                # combined parameter, logM + metallicity
        logsfr_ratios = theta[mod.theta_index['logsfr_ratios']]    # logSFR ratios (== nonparametric SFH ratios)
        mass_in_bins = pfile.logmass_to_masses(massmet=massmet,logsfr_ratios=logsfr_ratios,agebins=agebins)

        # now calculate average age
        out['mwa'] += [(mass_in_bins*mean_age).sum() / mass_in_bins.sum()]

        # sSFR averaged over 100 Myr
        mformed = mass_in_bins[:2].sum()   # first two bins cover 100 Myr
        time = 100e6
        out['ssfr'] += [np.log10((mformed/time) * stellar_to_total / mass_in_bins.sum())]

        # Stellar mass and metallicity
        out['stmass'] += [massmet[0]+np.log10(stellar_to_total)]
        out['met'] += [massmet[1]]

        # Dust1, Dust2
        d2 = theta[mod.theta_index['dust2']]
        out['dust2'] += [d2[0]]
        out['dust1'] += [d2[0]*theta[mod.theta_index['dust1_fraction']][0]]
        if (i+1)%(ndraw//20)==0: print("Finished up to draw %d"%(i+1))

    # save and exit
    pickle.dump(out,open('samples_z{0:.2f}.pickle'.format(zred), "wb"))

    # 1. nuke your pip install of prospector and python-fsps
    # 2. install and compile FSPS from github
    # 3. note that this involves setting $SPS_HOME
    # 4. install python-fsps from github
    # 5. install prospector from github
    # the command is "python setup.py install"; this will tuck the code somewhere that you can `import` from anywhere

def main_action():
    zred_arr = np.linspace(0.5,3.0,6)
    for zred in zred_arr:
        main(zred=zred)
        # M.main(zred=zred)

if __name__=='__main__':
    main_action()