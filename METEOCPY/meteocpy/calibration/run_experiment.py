import sys
sys.path.append('/Users/strim/Documents/Work/METEOCPY/meteocpy')

import pickle as pkl
import os
import numpy as np
import pandas as pd
from scipy.interpolate import BarycentricInterpolator, KroghInterpolator, CubicSpline

from os.path import join as pjoin

try:
    from meteocpy.forward import apex
except ModuleNotFoundError:
    from forward import apex


def run_experiment(out_path, spectrum_path, recompute, rang, n, intensity_var, batches_per_job, n_jobs,
                   run_mono=True):

    # #### LOAD INPUT SPECTRUM #####################################################
    # this section needs probably to change if you use a different file !!

    calibr = pd.read_csv(spectrum_path)
    calibr = calibr.iloc[:-1, :3].iloc[np.where(np.logical_and(calibr.iloc[:, 0] > rang[0],
                                                               calibr.iloc[:, 0] < rang[-1]))]#[:30]

    if len(calibr) < 2:
        raise Exception('The supplied APEX sensor range (%d - %d) and the spectral range of the supplied '
                        'input spectrum (%d - %d) do not overlap.' % (rang[0], rang[-1],
                                                                      calibr[0, 0], calibr[-1, 0]))

    inp_spectrum = calibr.iloc[:, 1].values
    wvls = calibr.iloc[:, 0].values
    
    # resample
    wvls_ = np.linspace(wvls[0], wvls[-1], int(wvls[-1] - wvls[0]) * n)  # get n samples per nm
    inp_spectrum = CubicSpline(wvls, inp_spectrum)(wvls_) # add MCMC for uncertainty on wvls_

    # create input_spectrum, dirac peak for all wvls in calibr at intensities in intensity_var
    # TODO: check physical variables
    inp_spectrum = np.stack([inp_spectrum * var for var in intensity_var], axis=1) * 5e6
    if run_mono:
        wvls = wvls_.reshape(-1, 1)
        inp_spectrum = inp_spectrum.reshape(len(inp_spectrum), len(intensity_var), 1)
    else:
        wvls = wvls_[None, :]
        inp_spectrum = inp_spectrum.transpose()[None, ...]

    print('***** SIMULATION \n',
          '***** Input shape %s \n' % str(inp_spectrum.shape),
          '***** Input support shape %s \n' % str(wvls.shape))
    




    ##### DEFINE APEX INSTANCE ###################################################
    if not os.path.exists(save_path) or recompute:
        ap = apex.load_apex(unbinned_vnir=pjoin(home, 'params/unbinned'),
                            binned_vnir_swir=pjoin(home, 'params/binned'),
                            binned_meta=pjoin(home, 'params/binned_meta'),
                            vnir_it=27000, swir_it=15000)
    
        # ap.initialize_srfs(rang, abs_res=abs_res, srf_support_in_sigma=3, zero_out=True, do_bin=True)
        ap.initialize_srfs(exact_wvls=wvls, srf_support_in_sigma=3, zero_out=True, 
                           do_bin=True)
    
        with open(save_path, 'wb') as f:
            pkl.dump(ap, f)
    else:
        with open(save_path, 'rb') as f:
            ap = pkl.load(f)
    
    
    
    
    
    ##### RUN #####################################################################
    # Simulate forward
    config = dict(inp_spectrum=inp_spectrum,
                  inp_wvlens=wvls, pad=False, part_covered=True,
                  invert=True, snr=True, dc=True, smear=True, return_binned=False,
                  run_specs=dict(joblib=True, verbose=False,
                                 batches_per_job=batches_per_job, n_jobs=n_jobs))
    
    res, illu_bands = ap.forward(**config)  # actual forward call, where we would supply gains and offsets (iterate)
    
    
    ##### WRITE ###################################################################
    with open(os.path.join(out_path, 'frames'), 'wb') as f:
        pkl.dump(res, f)
    
    with open(os.path.join(out_path, 'band_indices'), 'wb') as f:
        pkl.dump(illu_bands, f)
    
    with open(os.path.join(out_path, 'simulation_config'), 'wb') as f:
        pkl.dump(config, f) 


if __name__ == '__main__':
    """
    This script simulates the APEX sensor under the provided STARS spectrum (mono or multi). The spectrum can be varied
    with simple multiplicative factors. It is saved to simulations/{simulation_name}. The calculated APEX model is saved
    to saved_apex_models.
    """


    ##### SETTINGS ################################################################
    simulation_name = 'test'

    home = '/Users/strim/Documents/Work/METEOCPY'
    spectrum_path = pjoin(home, 'params/meteoc_spectrum/STAR_large_IS_100_percent_data.csv')
    recompute = False
    rang = [700, 1300]    # [700, 1300] works, do change range for simulating different 'chunks' of APEX - don't try full range (RAM)

    n = 3  # n samples per nm
    # intensity_var = np.arange(0.6, 2.5, 1)
    intensity_var = np.array([0.001, 0.2, 0.5, 0.7, 1, 2, 3])  # multiplicative variation of intensity
    # intensity_var = np.array([1])
    batches_per_job = 100
    n_jobs = 4
    run_mono = True
    
    here_path = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(here_path, 'saved_apex_models/apex_%d_%d' % (rang[0], rang[1]))
    
    out_path = os.path.join(here_path, 'simulations', simulation_name)
    os.makedirs(out_path, exist_ok=True)
    
    run_experiment(out_path, spectrum_path, recompute, rang, n, intensity_var,
                   batches_per_job, n_jobs, run_mono=run_mono)

    print('Finished!')

    





