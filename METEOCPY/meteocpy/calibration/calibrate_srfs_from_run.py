import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
from scipy.stats import norm

from calibration.utils_calibrate import gather_simulations
from utils import chunk_list

from functools import partial


# load simulation
simu_path = '/Users/strim/Documents/Work/METEOCPY/meteocpy/simulations/test'

# load simulation
with open(os.path.join(simu_path, 'frames'), 'rb') as f:
    frames = pkl.load(f)

with open(os.path.join(simu_path, 'band_indices'), 'rb') as f:
    illu_bands = pkl.load(f)

with open(os.path.join(simu_path, 'simulation_config'), 'rb') as f:
    config = pkl.load(f)

with open(os.path.join('/Users/strim/Documents/Work/METEOCPY/meteocpy', 'saved_apex_models', 'apex_700_1300'), 'rb') as f:
    ap = pkl.load(f)


# gather_simulations
wvls = config['inp_wvlens'].reshape(-1)
if len(wvls) != len(frames):
    raise ValueError('This script assumes the simulation is run in mono=True, i.e. there is one frame per wavelength.'
                     'This isn\'t the case with the provided simulation.')

band_dict, wvl_dict = gather_simulations(frames, illu_bands, wvls)


# define srf model
def gaussian_w_off(xdata, *params, size=10, batch_mode=False):
    # split params
    if batch_mode:
        a = np.array(params[:size]).reshape(-1, 1)
        mu = np.array(params[size:2*size]).reshape(-1, 1)
        sigma = np.array(params[2*size:3*size]).reshape(-1, 1)
        off = np.array(params[3*size:]).reshape(-1, 1)

    else:
        a, mu, sigma, off = params

    ret = a * norm.pdf(xdata, mu, sigma) + off
    return ret.flatten()


# fit srf model for each simulated pixel
srf_model = gaussian_w_off
channel = -1
plot_xdir_px = [250, 500, 750]

for band in list(band_dict.keys())[100:101]:

    # run in a loop to reduce number of simultaneously fitted params
    # -> circumvent scipy's limitation on 1000 params
    size = 10
    params = []
    xtrack_inds = list(np.arange(ap.DIM_X_AX))
    xdata = np.array(wvl_dict[band])  # .reshape(1, -1)

    for chunk in chunk_list(xtrack_inds, size):
        chunk = [chunk]
        sigma0 = np.sqrt(ap.get('fwhm'))[band].reshape(-1, 1)[chunk]
        mu0 = ap.get('cw')[band].reshape(-1, 1)[chunk]
        init = np.concatenate([np.ones(mu0.shape), mu0, sigma0, np.zeros(mu0.shape)], axis=0)

        ydata = np.stack(band_dict[band])[:, channel, chunk].squeeze().transpose()
        popt, pcov = opt.curve_fit(partial(srf_model, size=size, batch_mode=True),
                                   p0=init, xdata=xdata, ydata=ydata.flatten(), maxfev=int(2e4))

        # update params array
        rows = np.array([popt[i::size] for i in range(len(chunk[0]))])
        params.append(rows)

    params = np.concatenate(params)

    if plot_xdir_px is not None:
        high_res_x = np.linspace(xdata[0] - 5, xdata[-1] + 5, 100)

        plt.figure()
        for px in plot_xdir_px:
            ydata = np.stack(band_dict[band])[:, channel, px]
            plt.plot(high_res_x, srf_model(high_res_x, *params[px]))
            plt.scatter(xdata, ydata)

        output_directory = '/Users/strim/Documents/Work/METEOCPY/figures'
        os.makedirs(output_directory, exist_ok=True)
        output_filepath = os.path.join(output_directory, f'SRF reconstruction.pdf')
        plt.savefig(output_filepath, format='pdf')
        plt.show
        plt.show()