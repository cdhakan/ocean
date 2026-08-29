import scipy.io as sio
import numpy as np
from numpy import linalg as la

from typing import List, Dict, Optional

def dot_prod_matching(dictionary = None, acquired_data = None, dict_fn = None, acquired_data_fn = None, batch_size = 256):
    """
    :param dict_fn: path to dictionary (.mat) with filename
    :param acquired_data_fn: path to acquired data (.mat) with filename
    :param dictionary: dictionary with fields: t1w, t2w, and optionally t1s, t2s, fs, ksw (CEST) and/or t1m, t2m, fm, kss (MT), plus sig
    :param acquired_data: acquired data with dimensions: n_iter x r_raw_data x c_raw_data
    :param batch_size: batch size for dot product matching
    :return: quant_maps - quantitative maps dictionary with the fields: dp, t1w, t2w, and pool-specific maps
    """
    #  Modified by Dhakan 02/12/2026, to process MT pool

    if acquired_data_fn is not None:
        acquired_data = sio.loadmat(acquired_data_fn)['acquired_data']
        acquired_data = np.transpose(acquired_data,(3, 0, 1, 2))  # e.g. 30 x 116 x 116 x 88
    elif acquired_data is None:
        raise Exception("Either acquired_data or acquired_data_fn must be specified")

    if dict_fn is not None:
        synt_dict = sio.loadmat(dict_fn)
    elif dictionary is not None:
        synt_dict = dictionary
    elif dict_fn is None and dictionary is None:
        raise Exception("Either dictionary or dict_fn must be specified")

    # Detect available pools
    has_cest = False
    has_cest2 = False
    has_mt = False

    if len(synt_dict.keys()) < 4:
        # Legacy format: single structured array
        for k in synt_dict.keys():
            if k[0] != '_':
                key = k
        synt_dict = synt_dict[key][0]
        dict_t1w = synt_dict['t1w'][0].transpose()
        dict_t2w = synt_dict['t2w'][0].transpose()
        dict_t1s = synt_dict['t1s'][0].transpose()
        dict_t2s = synt_dict['t2s'][0].transpose()
        dict_fs = synt_dict['fs'][0].transpose()
        dict_ksw = synt_dict['ksw'][0].transpose()
        synt_sig = synt_dict['sig'][0]
        has_cest = True
    else:
        # New format: separate arrays
        dict_t1w = synt_dict['t1w']
        dict_t2w = synt_dict['t2w']

        # CEST pool (optional)
        if 'fs_0' in synt_dict:
            has_cest = True
            dict_t1s = synt_dict['t1s_0']
            dict_t2s = synt_dict['t2s_0']
            dict_fs = synt_dict['fs_0']
            dict_ksw = synt_dict['ksw_0']

            # search for additional CEST pool, add if present
            cpool01_keys = {key for key in synt_dict.keys() if key.endswith('_1')}
            if len(cpool01_keys) > 0:
                has_cest2 = True
                dict_fs2 = synt_dict['fs_1']
                dict_ksw2 = synt_dict['ksw_1']

        # MT pool (optional)
        if 'fm' in synt_dict:
            has_mt = True
            dict_t1m = synt_dict['t1m']
            dict_t2m = synt_dict['t2m']
            dict_fm = synt_dict['fm']
            dict_kss = synt_dict['kss']
            dict_dmw = synt_dict['dmw']

        synt_sig = np.transpose(synt_dict['sig'])  # e.g. 30 x 665,873

    # % Number of schedule iterations and raw data dimensions
    n_iter = np.shape(acquired_data)[0]
    r_raw_data = np.shape(acquired_data)[1]
    c_raw_data = np.shape(acquired_data)[2]
    if len(np.shape(acquired_data)) > 3: # detect if imaging data are 3D
        d_raw_data = np.shape(acquired_data)[3]
    else:
        d_raw_data = 1

    #  Reshaping image data to voxel - associated columns
    total_pixels = r_raw_data * c_raw_data * d_raw_data
    data = acquired_data.reshape((n_iter, total_pixels), order='F')

    # Output quantitative maps, initially as zero - vectors
    dp = np.zeros((1, total_pixels))
    t1w = np.copy(dp)
    t2w = np.copy(dp)

    if has_cest:
        t1s = np.copy(dp)
        t2s = np.copy(dp)
        fs = np.copy(dp)
        ksw = np.copy(dp)
        if has_cest2:
            fs2 = np.copy(dp)
            ksw2 = np.copy(dp)

    if has_mt:
        t1m = np.copy(dp)
        t2m = np.copy(dp)
        fm = np.copy(dp)
        kss = np.copy(dp)
        dmw = np.copy(dp)

    # 2 - norm normalization
    # equivalent to normc in matlab
    norm_dict = synt_sig / (la.norm(synt_sig, axis=0) + 1e-10)
    norm_data = data / (la.norm(data, axis=0) + 1e-10)

    # Matching in batches due to memory considerations
    # The number of image pixel needs to be dividable by batch_size
    assert np.shape(data)[1] / batch_size == np.round(np.shape(data)[1] / batch_size)

    batch_indices = range(0, data.shape[1], batch_size)
    for ind in range(np.size(batch_indices)):
        #  Dot - product for the current batch
        current_score = np.dot(np.transpose(norm_data[:, batch_indices[ind]: batch_indices[ind] + batch_size]),
                               norm_dict)

        # Finding maximum dot - product and storing the corresponding parameters
        dp[0, batch_indices[ind]: batch_indices[ind] + batch_size] = np.max(current_score, axis=1)
        dp_ind = np.argmax(current_score, axis=1)

        t1w[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t1w[0, dp_ind]
        t2w[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t2w[0, dp_ind]

        if has_cest:
            t1s[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t1s[0, dp_ind]
            t2s[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t2s[0, dp_ind]
            fs[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_fs[0, dp_ind]
            ksw[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_ksw[0, dp_ind]

            if has_cest2:
                fs2[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_fs2[0, dp_ind]
                ksw2[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_ksw2[0, dp_ind]

        if has_mt:
            t1m[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t1m[0, dp_ind]
            t2m[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_t2m[0, dp_ind]
            fm[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_fm[0, dp_ind]
            kss[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_kss[0, dp_ind]
            dmw[0, batch_indices[ind]: batch_indices[ind] + batch_size] = dict_dmw[0,dp_ind]

    # Reshaping the output to the original image dimensions
    reshape_dims = (r_raw_data, c_raw_data, d_raw_data) if d_raw_data > 1 else (r_raw_data, c_raw_data)

    quant_maps = {
        'dp': dp.reshape(reshape_dims, order='F'),
        't1w': t1w.reshape(reshape_dims, order='F'),
        't2w': t2w.reshape(reshape_dims, order='F'),
    }

    if has_cest:
        quant_maps['t1s'] = t1s.reshape(reshape_dims, order='F')
        quant_maps['t2s'] = t2s.reshape(reshape_dims, order='F')
        quant_maps['fs'] = fs.reshape(reshape_dims, order='F')
        quant_maps['ksw'] = ksw.reshape(reshape_dims, order='F')
        if has_cest2:
            quant_maps['fs2'] = fs2.reshape(reshape_dims, order='F')
            quant_maps['ksw2'] = ksw2.reshape(reshape_dims, order='F')

    if has_mt:
        quant_maps['t1m'] = t1m.reshape(reshape_dims, order='F')
        quant_maps['t2m'] = t2m.reshape(reshape_dims, order='F')
        quant_maps['fm'] = fm.reshape(reshape_dims, order='F')
        quant_maps['kss'] = kss.reshape(reshape_dims, order='F')
        quant_maps['dmw'] = dmw.reshape(reshape_dims, order='F')

    return quant_maps
