from joblib import Parallel, delayed
import numpy as np
import os
from scipy.io import loadmat
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_frame(ap, simu, illu_bands, ind, channel=0, vmax=None, vmin=None, figsize=None, binned='binned', output_directory=None, filename='plot_frame.pdf'):
    tostr = lambda f: "%.2f" % f
    wvls = list(map(tostr, ap.params[binned].cw[illu_bands[ind]].mean(axis=1)))

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    im = axs[0].matshow(simu[ind][channel], aspect='auto', vmax=vmax, vmin=vmin)
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[0].set_yticklabels([''] + wvls, fontsize=12)
    axs[0].set_xlabel('Spatial Bands', fontsize=14)
    axs[0].set_ylabel('Wavelength [nm]', fontsize=14)
    axs[0].set_title('Simulated APEX frame', fontsize=16)

    # Move x-axis labels to the bottom
    axs[0].xaxis.set_ticks_position('bottom')
    axs[0].xaxis.set_label_position('bottom')
    axs[0].tick_params(axis='x', which='major', labelsize=12)

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='1%', pad=0.05)
    plt.gcf().colorbar(im, cax=cax)

    axs[1].plot(wvls, simu[ind][channel][:, [250, 500, 750]], '-o') # so we're plotting spatial bands 250,500,750
    axs[1].set_ylim([np.min(simu[ind][channel][:, [250, 500, 750]]) * 0.9, np.max(simu[ind][channel][:, [250, 500, 750]]) * 1.1])
    axs[1].set_xlabel('Wavelength [nm]', fontsize=14)
    axs[1].set_ylabel('DNs [-]', fontsize=14)
    axs[1].set_title('Spatial Bands 250, 500, 750', fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    # Save the figure to a PDF file
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        output_filepath = os.path.join(output_directory, filename)
        plt.savefig(output_filepath, format='pdf')



def run_jobs(jobs, joblib=True, n_jobs=4, chunks=1, chunk_callback=None, verbose=False, *args, **kwargs):
    if len(jobs) == 0:
        return None

    if joblib:
        jobs = [delayed(job)() for job in jobs]

        chunk_size = max(1, len(jobs) // chunks)
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]

        out = []
        for chunk in chunks:
            chunk_out = Parallel(n_jobs=n_jobs, verbose=verbose, *args, **kwargs)(chunk)
            if chunk_callback is not None:
                ret = chunk_callback(chunk_out, args=[job[0].args for job in chunk],
                                     kwargs=[job[0].keywords for job in chunk])
                out.append((chunk_out, ret))
            else:
                out.append(chunk_out)

    else:
        out = []
        # create chunks
        nr_chunks = chunks
        chunk_size = max(1, len(jobs) // chunks)
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]

        for j, chunk in enumerate(chunks):
            chunk_out = []

            for i, job in enumerate(chunk):
                if verbose:
                    print('\r\r Chunk %d / %d' % (j, nr_chunks) +
                          '\n Working on job %d/%d, ' % (i, len(chunk)) +
                          '\n args: %s, \n kwargs: %s' % (', '.join(job.args), ', '.join([str(tup)
                                                                                          for tup in job.keywords.items()])))
                chunk_out.append(job())

            if chunk_callback is not None:
                ret = chunk_callback(chunk_out, args=[job.args for job in chunk],
                                     kwargs=[job.keywords for job in chunk])
                out.append((chunk_out, ret))
            else:
                out.append(chunk_out)

    # flatten over chunks
    if chunk_callback is not None:
        out, callback_out = zip(*out)
        out = list(itertools.chain(*out))
        callback_out = list(itertools.chain(*out))
        out = (out, callback_out)

    else:
        out = list(itertools.chain(*out))

    return out


def inds_from_slice2d(start, end, axis=0, end_is_rev=False):
    all = np.concatenate([[i] * j for i, j in enumerate(start)]).astype(int) # np.int deprecated: use just 'int'

    inds_start = tuple([all] * axis + [np.concatenate([range(i) for i in start]).astype(int)]
                       + [all] * (2 - axis - 1))

    if end_is_rev:
        inds_end = tuple([all] * axis + [np.concatenate([range(-i, 0) for i in end]).astype(int)]
                         + [all] * (2 - axis - 1))
    else:
        inds_end = tuple([all] * axis + [np.concatenate([range(i) for i in end]).astype(int)]
                         + [all] * (2 - axis - 1))

    inds = tuple([np.concatenate([inds_start[i], inds_end[i]]) for i in range(2)])

    return inds


def load_mat_paths(base_path):
    if type(base_path) is list:
        paths = base_path
    else:
        paths = [os.path.join(base_path, p) for p in os.listdir(os.path.join(base_path))]

    params = {}
    for fil in paths:
        if fil.endswith('mat'):
            mat = loadmat(os.path.join(base_path, 'params', fil), mat_dtype=False)
            keys = [k for k in mat.keys() if not k.startswith('__')]

            for k in keys:
                inarr = mat[k]
                if inarr.dtype.names is not None:
                    inarr = convert_structured_array_to_dict(inarr)
                else:
                    inarr = inarr.astype(float)
                    # np.float was a deprecated alias for the builtin `float`.
                    # To avoid this error in existing code, use `float` by itself.
                    # Doing this will not modify any behavior and is safe.
                params.update({k.lower(): inarr})

    return params


def load_params(calibration_path=None, meta_path=None):
    loaded_params = {'calibr': load_mat_paths(calibration_path) if calibration_path is not None else None,
                     'meta': load_mat_paths(meta_path) if meta_path is not None else None}
    params = {}

    # if len(calibr) == 1 it's a calibration cube
    if (loaded_params['calibr'] is not None
            and len(loaded_params['calibr']) == 1 and 'cube' in loaded_params['calibr']):
        cube = loaded_params['calibr']['cube']

        params['rad_coeffs'] = {'gain': cube[3], 'offset': cube[4]}
        params['cw'] = cube[5]
        params['fwhm'] = cube[6]

    # all input parameters are in individual files
    elif loaded_params['calibr'] is not None:
        params = dict(**loaded_params['calibr'])

    if loaded_params['meta'] is not None:
        params.update(loaded_params['meta'])
    return params


def convert_structured_array_to_dict(sarr):
    dic = {}
    for name in sarr.dtype.names:
        dic[name] = sarr[name][0][0]
    return dic


def pad_and_shift(arr, shift, val=np.nan, pad=True, axis=0, padded=None, add=False):
    if shift == 0:
        return arr

    if not pad and padded is None:
        padded = np.ones(arr) * val
    elif pad and padded is None:
        shape = tuple([s if i != axis else s + np.abs(shift) for i, s in enumerate(arr.shape)])
        padded = np.ones(shape) * val

    if shift > 0:
        slice_pre = tuple([slice(None, None) if i != axis else slice(shift, None) for i in range(len(arr.shape))])
        slice_post = tuple([slice(None, None) if i != axis else slice(None, -shift) for i in range(len(arr.shape))])
        if add:
            padded[slice_pre] += arr[slice_post]
        else:
            padded[slice_pre] = arr[slice_post]
    elif shift < 0:
        slice_pre = tuple([slice(None, None) if i != axis else slice(None, shift) for i in range(len(arr.shape))])
        slice_post = tuple([slice(None, None) if i != axis else slice(-shift, None) for i in range(len(arr.shape))])
        if add:
            padded[slice_pre] += arr[slice_post]
        else:
            padded[slice_pre] = arr[slice_post]

    return padded


def shifted_padded_sum(arrs, pos, axis=0):
    max_pos = np.max([arr.shape[axis] + p for arr, p in zip(arrs, pos)])
    min_pos = np.min(pos)

    new_len = max_pos - min_pos
    prealloc = np.zeros(tuple([s if ax != axis else new_len for ax, s in enumerate(arrs[0].shape)]))

    # for p, arr in zip(pos, arrs):
    #     prealloc.take(slice(p, p + arr.shape[axis]), axis=axis) += pad_and_shift(arr, val=0, padded=prealloc, add=True)


class BiDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if not hasattr(self, 'inverse'):
            self.inverse = {}

        if key in self:
            self.inverse[self[key]].remove(key)
        super().__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super().__delitem__(key)


# def chunk_list(it, size):
#     it = iter(it)
#     return list(iter(lambda: tuple(itertools.islice(it, size)), ()))

def chunk_list(it, size):
    x = [it[i:i + size] for i in range(0, len(it), size)]
    return x


class _AttributeDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError

    __setattr__ = dict.__setitem__