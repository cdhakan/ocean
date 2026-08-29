import scipy.io as sio
import numpy as np
from numpy import linalg as la
import sys


def dot_prod_matching_chunked(dictionary=None, acquired_data=None, dict_fn=None,
                               acquired_data_fn=None, voxel_batch_size=256,
                               dict_chunk_size=5000000):
    """
    Memory-efficient dot product matching that processes dictionary in chunks.

    :param dict_fn: path to dictionary (.mat) with filename
    :param acquired_data_fn: path to acquired data (.mat) with filename
    :param dictionary: dictionary with fields: t1w, t2w, t1s, t2s, fs, ksw, sig
    :param acquired_data: acquired data with dimensions: n_iter x r_raw_data x c_raw_data
    :param voxel_batch_size: batch size for processing voxels (default: 256)
    :param dict_chunk_size: number of dictionary entries to process at once (default: 5M)
    :return: quant_maps - quantitative maps dictionary with the fields: dp, t1w, t2w, fs, ksw

    Key difference from standard dot_prod_matching:
    - Processes dictionary in chunks of dict_chunk_size entries
    - For each chunk: loads, normalizes, computes dot products, keeps best matches
    - Memory usage: O(dict_chunk_size) instead of O(total_dict_size)
    - Enables matching with 44M+ entry dictionaries on limited RAM
    """

    print(f"Starting chunked dictionary matching...")
    print(f"Dictionary chunk size: {dict_chunk_size:,} entries")
    print(f"Voxel batch size: {voxel_batch_size}")

    # Load acquired data
    if acquired_data_fn is not None:
        print(f"Loading acquired data from {acquired_data_fn}...")
        acquired_data = sio.loadmat(acquired_data_fn)['acquired_data']
        acquired_data = np.transpose(acquired_data, (3, 0, 1, 2))
    elif acquired_data is None:
        raise Exception("Either acquired_data or acquired_data_fn must be specified")

    # Load dictionary
    if dict_fn is not None:
        print(f"Loading dictionary from {dict_fn}...")
        synt_dict = sio.loadmat(dict_fn)
    elif dictionary is not None:
        synt_dict = dictionary
    else:
        raise Exception("Either dictionary or dict_fn must be specified")

    # Extract dictionary parameters
    if len(synt_dict.keys()) < 4:
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
    else:
        dict_t1w = synt_dict['t1w']
        dict_t2w = synt_dict['t2w']
        dict_t1s = synt_dict['t1s_0']
        dict_t2s = synt_dict['t2s_0']
        dict_fs = synt_dict['fs_0']
        dict_ksw = synt_dict['ksw_0']

        # Check for additional CEST pool
        cpool01_keys = {key for key in synt_dict.keys() if key.endswith('_1')}
        if len(cpool01_keys) > 0:
            dict_fs2 = synt_dict['fs_1']
            dict_ksw2 = synt_dict['ksw_1']

        synt_sig = np.transpose(synt_dict['sig'])  # e.g. 30 x N_entries

    print(f"Dictionary shape: {synt_sig.shape}")
    print(f"Dictionary entries: {synt_sig.shape[1]:,}")
    print(f"Z-spectrum points: {synt_sig.shape[0]}")

    # Data dimensions
    n_iter = np.shape(acquired_data)[0]
    r_raw_data = np.shape(acquired_data)[1]
    c_raw_data = np.shape(acquired_data)[2]
    if len(np.shape(acquired_data)) > 3:
        d_raw_data = np.shape(acquired_data)[3]
    else:
        d_raw_data = 1

    print(f"Acquired data shape: {acquired_data.shape}")
    print(f"Total voxels: {r_raw_data * c_raw_data * d_raw_data:,}")

    # Reshape image data to voxel columns
    data = acquired_data.reshape((n_iter, r_raw_data * c_raw_data * d_raw_data), order='F')

    # Normalize acquired data (do this once)
    print("Normalizing acquired data...")
    norm_data = data / (la.norm(data, axis=0) + 1e-10)

    # Initialize output maps
    n_voxels = r_raw_data * c_raw_data * d_raw_data
    dp = np.zeros((1, n_voxels))
    t1w = np.zeros((1, n_voxels))
    t2w = np.zeros((1, n_voxels))
    t1s = np.zeros((1, n_voxels))
    t2s = np.zeros((1, n_voxels))
    fs = np.zeros((1, n_voxels))
    ksw = np.zeros((1, n_voxels))
    if len(cpool01_keys) > 0:
        fs2 = np.zeros((1, n_voxels))
        ksw2 = np.zeros((1, n_voxels))

    # Best match indices (global across all chunks)
    best_indices = np.zeros((1, n_voxels), dtype=np.int64)

    # Process dictionary in chunks
    n_dict_entries = synt_sig.shape[1]
    n_chunks = int(np.ceil(n_dict_entries / dict_chunk_size))

    print(f"\nProcessing {n_dict_entries:,} entries in {n_chunks} chunks...")

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * dict_chunk_size
        chunk_end = min((chunk_idx + 1) * dict_chunk_size, n_dict_entries)
        chunk_len = chunk_end - chunk_start

        print(f"\nChunk {chunk_idx + 1}/{n_chunks}: entries {chunk_start:,} to {chunk_end:,} ({chunk_len:,} entries)")

        # Extract and normalize this chunk of the dictionary
        synt_sig_chunk = synt_sig[:, chunk_start:chunk_end]
        norm_dict_chunk = synt_sig_chunk / (la.norm(synt_sig_chunk, axis=0) + 1e-10)

        print(f"  Normalized chunk shape: {norm_dict_chunk.shape}")
        print(f"  Memory: ~{norm_dict_chunk.nbytes / (1024**3):.2f} GB")

        # Process voxels in batches for this dictionary chunk
        assert n_voxels % voxel_batch_size == 0, "Number of voxels must be divisible by voxel_batch_size"

        batch_indices = range(0, n_voxels, voxel_batch_size)
        n_batches = len(list(batch_indices))

        for batch_num, batch_start_idx in enumerate(batch_indices):
            batch_end_idx = batch_start_idx + voxel_batch_size

            if (batch_num + 1) % 10 == 0:  # Progress every 10 batches
                print(f"  Processing voxel batch {batch_num + 1}/{n_batches}", end='\r')
                sys.stdout.flush()

            # Dot product for current voxel batch with current dictionary chunk
            current_score = np.dot(
                np.transpose(norm_data[:, batch_start_idx:batch_end_idx]),
                norm_dict_chunk
            )

            # Find best matches within this chunk
            chunk_best_scores = np.max(current_score, axis=1)
            chunk_best_indices = np.argmax(current_score, axis=1)

            # Update global best matches if this chunk has better scores
            for i, voxel_idx in enumerate(range(batch_start_idx, batch_end_idx)):
                if chunk_best_scores[i] > dp[0, voxel_idx]:
                    dp[0, voxel_idx] = chunk_best_scores[i]
                    # Store global index (chunk_start + local index within chunk)
                    best_indices[0, voxel_idx] = chunk_start + chunk_best_indices[i]

        print(f"  Completed chunk {chunk_idx + 1}/{n_chunks}                    ")

    print("\nAll chunks processed. Extracting final parameter values...")

    # Extract parameters for best matches
    for voxel_idx in range(n_voxels):
        global_idx = best_indices[0, voxel_idx]
        t1w[0, voxel_idx] = dict_t1w[0, global_idx]
        t2w[0, voxel_idx] = dict_t2w[0, global_idx]
        t1s[0, voxel_idx] = dict_t1s[0, global_idx]
        t2s[0, voxel_idx] = dict_t2s[0, global_idx]
        fs[0, voxel_idx] = dict_fs[0, global_idx]
        ksw[0, voxel_idx] = dict_ksw[0, global_idx]

        if len(cpool01_keys) > 0:
            fs2[0, voxel_idx] = dict_fs2[0, global_idx]
            ksw2[0, voxel_idx] = dict_ksw2[0, global_idx]

    # Reshape to original image dimensions
    print("Reshaping output maps...")
    if d_raw_data > 1:
        if len(cpool01_keys) > 0:
            quant_maps = {
                'dp': dp.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                't1w': t1w.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                't2w': t2w.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                'fs': fs.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                'ksw': ksw.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                'fs2': fs2.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                'ksw2': ksw2.reshape((r_raw_data, c_raw_data, d_raw_data), order='F')
            }
        else:
            quant_maps = {
                'dp': dp.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                't1w': t1w.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                't2w': t2w.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                'fs': fs.reshape((r_raw_data, c_raw_data, d_raw_data), order='F'),
                'ksw': ksw.reshape((r_raw_data, c_raw_data, d_raw_data), order='F')
            }
    else:
        if len(cpool01_keys) > 0:
            quant_maps = {
                'dp': dp.reshape((r_raw_data, c_raw_data), order='F'),
                't1w': t1w.reshape((r_raw_data, c_raw_data), order='F'),
                't2w': t2w.reshape((r_raw_data, c_raw_data), order='F'),
                'fs': fs.reshape((r_raw_data, c_raw_data), order='F'),
                'ksw': ksw.reshape((r_raw_data, c_raw_data), order='F'),
                'fs2': fs2.reshape((r_raw_data, c_raw_data), order='F'),
                'ksw2': ksw2.reshape((r_raw_data, c_raw_data), order='F')
            }
        else:
            quant_maps = {
                'dp': dp.reshape((r_raw_data, c_raw_data), order='F'),
                't1w': t1w.reshape((r_raw_data, c_raw_data), order='F'),
                't2w': t2w.reshape((r_raw_data, c_raw_data), order='F'),
                'fs': fs.reshape((r_raw_data, c_raw_data), order='F'),
                'ksw': ksw.reshape((r_raw_data, c_raw_data), order='F')
            }

    print("Chunked matching complete!")
    return quant_maps