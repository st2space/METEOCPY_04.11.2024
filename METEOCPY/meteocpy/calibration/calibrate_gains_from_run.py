import pickle as pkl
import os
import numpy as np
import scipy.interpolate as interp
from tqdm import tqdm
import time
import sys
sys.path.append('/home/strim/Documents/Work/METEOCPY/meteocpy')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from calibration.utils_calibrate import target_mean, create_inp_spectrum, create_dns


simu_path = '/Users/strim/Documents/Work/METEOCPY/meteocpy/simulations/test'
target = target_mean
is_mono = True
plot = True



# load simulation
with open(os.path.join(simu_path, 'frames'), 'rb') as f:
    frames = pkl.load(f)

with open(os.path.join(simu_path, 'band_indices'), 'rb') as f:
    illu_bands = pkl.load(f)

with open(os.path.join(simu_path, 'simulation_config'), 'rb') as f:
    config = pkl.load(f)

with open(os.path.join('/Users/strim/Documents/Work/METEOCPY/meteocpy', 'saved_apex_models', 'apex_700_1300'), 'rb') as f:
    ap = pkl.load(f)


if is_mono:
    # for each simulated band construct an ensemble of input spectra from delta peaks
    # e.g. uniform or random intensities
    
    # create input spectrum
    inp_spectra, inp_wvls = create_inp_spectrum(config, model='uniform')
    

else:
    inp_spectra, inp_wvls = config['inp_spectrum'].squeeze(0), config['inp_wvlens'][0]
    
# create dn output from delta peak results, if there is only one frame as in
# is_mono == False this just reorders data
dns, wvls_per_band = create_dns(frames=frames, illu_bands=illu_bands, inp_wvls=inp_wvls)
 
# create target, TODO: THIS IS FUNDAMENTALLY WRONG AS IT ASSUMES LINEARITY
target_per_band = [target(ap, band, inp_spectra, inp_wvls) for band in wvls_per_band.keys()]


# fit gain over all input spectra in the ensemble
p = np.asarray([[np.polyfit(dns[band][:, x], target_per_band[band], deg=1, cov=False) 
                 if target_per_band[band] is not None else np.array([np.nan, np.nan])
                 for x in range(ap.DIM_X_AX)]
                 for band in range(len(dns)) ])


if plot:
    band = 8  # band 10 for default
    x = 500
    xs = np.linspace(dns[band][:, x][0], dns[band][:, x][-1], 3)
    fit = np.poly1d(np.polyfit(dns[band][:, x], target_per_band[band], deg=1, cov=False))
    plt.scatter(dns[band][:, x], target_per_band[band]); plt.plot(xs, fit(xs), 'red')
    plt.xlabel('DNs for band 8 normalized by integration time [-]', fontsize=13)
    plt.ylabel('Intensity (multiplicative factors of\ninput radiance [W.m-2.sr-1.nm-1])', fontsize=13)
    plt.title(f'Straight line fitting of radiance vs DNs\nfor band {band}', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)

    output_directory = '/Users/strim/Documents/Work/METEOCPY/figures'
    os.makedirs(output_directory, exist_ok=True)
    output_filepath = os.path.join(output_directory, f'linear_fit_gain_band_{band}.pdf')
    plt.savefig(output_filepath, format='pdf')
    plt.show

# Extract the gain values from p
linear_gains = np.array([[p[band][x][0] for x in range(ap.DIM_X_AX)] for band in range(len(dns))])


# Directory where the text file will be saved
save_directory = '/Users/strim/Documents/Work/METEOCPY'

# Construct the full file path
save_filepath = os.path.join(save_directory, 'linear_gains_run30.txt')

# Save the linear_gains array to a .txt file
np.savetxt(save_filepath, linear_gains)


# Compare the coefficients with the integrated responsivity values
band = 8  # Example band to compare, band 10 for default
pixel = 500  # Example pixel to compare

print(f"Linear Gain (Slope) for Band {band}, Pixel {pixel}: {linear_gains[band, pixel]}")
#print(f"Integrated Responsivity for Band {band}, Pixel {pixel}: {integrated_responsivity[band, pixel]}")

# Plot the gains for a given band
plt.figure(figsize=(10, 5))
plt.plot(linear_gains[band], label='Linear Gain (Slope)')
#plt.plot(integrated_responsivity[band], label='Integrated Responsivity')
plt.xlabel('Across-track pixels (spatial bands)', fontsize=14)
plt.ylabel('Absolute spectral responsivity', fontsize=14)
plt.title(f'Absolute spectral responsivity for Band {band} for all across-track pixels', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
# plt.legend()
# plt.show()

# Save the figure to a PDF file
output_directory = '/Users/strim/Documents/Work/METEOCPY/figures'
os.makedirs(output_directory, exist_ok=True)
output_filepath = os.path.join(output_directory, f'absolute_spectral_responsivity_band_{band}_xtrack_{pixel}.pdf')
plt.savefig(output_filepath, format='pdf')
plt.show()


dns_per_intensity = dns[band][:,x]
print((dns_per_intensity))
something = target_per_band[band]
print(something)
# wvl_check = wvls_per_band
# print(wvl_check)
support_wvl = ap.get('srf_support_per_band')[band]
mean_wvl = support_wvl.mean()
print(support_wvl)
print(mean_wvl)

# print((wvls_per_band[band], dns[band][:,x]))

# EXPERIMENTATION: trying to integrate DNs, there's only one DN value per band per intensity, so for SSF sim you need to use buffer to delimite wvl range and locate which bands' CW fall within
# said range, and sum the DNs from all those bands, for every band's CW - delete if fuck up


def calculate_absolute_spectral_responsivity(ap, frames, illu_bands, inp_wvls, model='uniform'):
    """
    Calculate the absolute spectral responsivity by summing up the DNs from each band within the range of +/- 1.5*FWHM
    about the mean wavelength of every band.

    :param ap: APEX imager configuration object
    :param frames: List of simulation frames
    :param illu_bands: List of illuminated bands
    :param inp_wvls: List of input wavelengths
    :param model: Model used to calculate the DN vectors (default is 'uniform')
    :return: Dictionary of absolute spectral responsivity for each band and pixel
    """
    # Get DN vectors and wavelength dictionary
    # dn_vectors, wvl_dict = create_dns(frames, illu_bands, inp_wvls, model=model)
    dn_vectors, wvls_per_band = create_dns(frames=frames, illu_bands=illu_bands, inp_wvls=inp_wvls)

    mean_wvl_dict = {}
    buffer_range_dict = {}


    # Debugging: Print the keys of wvls_per_band
    print("Keys of wvls_per_band:", (wvls_per_band.keys()))

    # Debugging: Print the length of srf_support_per_band
    print("Length of srf_support_per_band:", (ap.get('srf_support_per_band')))

    # Create a mapping from wvls_per_band keys to srf_support_per_band indices
    band_id_mapping = {key: idx for idx, key in enumerate(wvls_per_band.keys())}



    # Calculate mean wavelength and buffer range for each band
    for band_id in wvls_per_band.keys():          # for band_id in wvls_per_band.keys():
        mapped_band_id = band_id_mapping[band_id]
        support_wvl = ap.get('srf_support_per_band')[mapped_band_id]
        # print(len(support_wvl))
        mean_wvl = np.mean(support_wvl)
        fwhm = ap.get('fwhm')[mapped_band_id]
        buffer_range = 1.5 * fwhm

        mean_wvl_dict[band_id] = mean_wvl
        buffer_range_dict[band_id] = buffer_range

    absolute_spectral_responsivity = {}


    # Start the stopwatch
    start_time = time.time()

    # Calculate absolute spectral responsivity for each band, each illumination condition, and each pixel
    for band_id in tqdm(wvls_per_band.keys(), desc="Processing bands"): # for band_id, dn_vector in zip(wvls_per_band.keys(), dn_vectors):
        mean_wvl = mean_wvl_dict[band_id]
        buffer_range = buffer_range_dict[band_id]

        lower_bound = mean_wvl - buffer_range
        upper_bound = mean_wvl + buffer_range

        # Identify bands with mean wavelength within the buffer range
        bands_within_range = [b for b, mw in mean_wvl_dict.items() if lower_bound.any() <= mw.any() <= upper_bound.any()]
        # bands_within_range = [b for b, mw in mean_wvl_dict.items() if mw[np.logical_and(mw[band_id] >= lower_bound, mw[band_id] <= upper_bound)]] # lower_bound <= mw <= upper_bound

        # Initialize the total DN array for the current band
        total_dn_per_illumination = np.zeros((len(dn_vectors[0]), len(dn_vectors[0][0])))

        # Sum up the DNs from each of these bands within the range for each illumination condition and each pixel
        for illumination_idx in range(len(dn_vectors[0])):
            for pixel_idx in range(len(dn_vectors[0][illumination_idx])):
                total_dn = 0
                wvls_within_range = []
                dn_values_within_range = []
                for b in bands_within_range:
                    mapped_band_id = band_id_mapping[b]
                    support_wvl = ap.get('srf_support_per_band')[mapped_band_id]
                    mean_wvl_b = mean_wvl_dict[b]

                    # Collect DN values and corresponding wavelengths for the current pixel and illumination condition
                    wvls_within_range.append(mean_wvl_b)
                    dn_values_within_range.append(dn_vectors[mapped_band_id][illumination_idx][pixel_idx])

                # Convert lists to arrays for interpolation
                wvls_within_range = np.array(wvls_within_range)
                dn_values_within_range = np.array(dn_values_within_range)

                # Debugging: Print lengths of x and y arrays
                # print(f"Band {band_id}, Illumination {illumination_idx}, Pixel {pixel_idx}:")
                # print(f"  Length of wvls_within_range: {len(wvls_within_range)}")
                # print(f"  Length of dn_values_within_range: {len(dn_values_within_range)}")

                # Interpolate DN values for the current pixel and illumination condition
                if len(wvls_within_range) == len(dn_values_within_range):
                    interp_func = interp.interp1d(wvls_within_range, dn_values_within_range, bounds_error=False,
                                                  fill_value="extrapolate")
                    interpolated_dn = interp_func(mean_wvl)
                    total_dn += interpolated_dn
                else:
                    print(
                        f"Length mismatch for band {band_id}: wvls_within_range has length {len(wvls_within_range)}, dn_values_within_range has length {len(dn_values_within_range)}")

                total_dn_per_illumination[illumination_idx][pixel_idx] = total_dn

        absolute_spectral_responsivity[band_id] = total_dn_per_illumination

    # End the stopwatch
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

    return absolute_spectral_responsivity


# Example usage
# ap = load_apex(...)  # Example APEX imager configuration
# frames = [...]  # Example frames
# illu_bands = [...]  # Example illuminated bands
# inp_wvls = [...]  # Example input wavelengths

# Calculate absolute spectral responsivity (activate/deactivate as needed)
# absolute_spectral_responsivity = calculate_absolute_spectral_responsivity(ap, frames, illu_bands, inp_wvls)

# print(absolute_spectral_responsivity)


# Plot the absolute spectral responsivity for band 80, all pixels (activate/deactivate as needed)
# band_id_to_plot = 80
# if band_id_to_plot in absolute_spectral_responsivity:
#     plt.figure(figsize=(10, 6))
#     for illumination_idx in range(len(absolute_spectral_responsivity[band_id_to_plot])):
#         plt.plot(absolute_spectral_responsivity[band_id_to_plot][illumination_idx], label=f'Illumination {illumination_idx}')
#     plt.title(f'Absolute Spectral Responsivity for Band {band_id_to_plot}')
#     plt.xlabel('Pixel Index')
#     plt.ylabel('DN Value')
#     plt.legend()
#     plt.show()
# else:
#     print(f"Band {band_id_to_plot} not found in the absolute spectral responsivity results.")