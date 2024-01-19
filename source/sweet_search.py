import cooler
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing
from functools import partial
import warnings
from scipy.stats import binom
import scipy
import sys
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

           
def calculate_sweet_sum_on_diag_sparse(diag):
    coords = []
    data = []
    for i in range(diag, diff_array_csr.shape[0], step):
        left_sweet_square = diff_array_csr[i-diag-sweet_size:i-diag, i-sweet_size:i].toarray()
        right_sweet_square = diff_array_csr[i-diag:i-diag+sweet_size, i:i+sweet_size].toarray() 
        
        # ##right_sweet
        left_sweet_sum = np.sum(np.triu(np.rot90(left_sweet_square, 1)))
        right_sweet_sum = np.sum(np.triu(np.rot90(right_sweet_square, 3)))
        coords.append((i-diag, i))
        data.append(left_sweet_sum+right_sweet_sum)
    return diag, coords, data

def calculate_depth_on_all_chroms(chrom):
    ## open hic data for patient and control in sparse format
    sample_data_raw = cool_sample.matrix(balance=False, sparse=True).fetch(chrom)
    ref_data_raw = cool_ref.matrix(balance=False, sparse=True).fetch(chrom)
    if use_only_p_c == True:
        max_coord_const = np.int64(10**7)
        _,_, same_coords_inds = np.intersect1d(np.int64(sample_data_raw.row)*max_coord_const+np.int64(sample_data_raw.col),
        np.int64(ref_data_raw.row)*max_coord_const+np.int64(ref_data_raw.col), assume_unique=True, return_indices=True)
        if len(same_coords_inds) != len(sample_data_raw.data):
            # print("i'm here", len(same_coords_inds), len(sample_data_raw.data))
            ref_data_raw = (ref_data_raw+sample_data_raw).tocoo()
            _,_, same_coords_inds = np.intersect1d(np.int64(sample_data_raw.row)*max_coord_const+np.int64(sample_data_raw.col),
            np.int64(ref_data_raw.row)*max_coord_const+np.int64(ref_data_raw.col), assume_unique=True, return_indices=True)
        ref_data_raw.data = ref_data_raw.data[same_coords_inds]
        ref_data_raw.row = ref_data_raw.row[same_coords_inds]
        ref_data_raw.col = ref_data_raw.col[same_coords_inds]
        # print(len(same_coords_inds), len(sample_data_raw.data), len(ref_data_raw.data))
    depth_sample = np.sum(sample_data_raw.data)
    depth_ref = np.sum(ref_data_raw.data)
    return depth_sample, depth_ref

def main():
    global diff_array_csr, cool_sample, cool_ref, use_only_p_c, sweet_size, step, sample, out_dir

    sample_path = sys.argv[1]
    control_path = sys.argv[2]
    bin_size = int(sys.argv[3])
    sweet_size = int(sys.argv[4])
    chrom_search = sys.argv[5]
    big_depth_sample = int(sys.argv[6])
    big_depth_ref = int(sys.argv[7])
    n_cpus = int(sys.argv[8])
    out_dir = sys.argv[9]
    
    print(datetime.now())

    step = 1
    res_size_data = pd.read_csv("./source/res_size.txt", sep="\t")
    min_diag = res_size_data[res_size_data["resolution"]==bin_size].iloc[0,1]//bin_size
    max_diag = res_size_data[res_size_data["resolution"]==bin_size].iloc[0,2]//bin_size

    use_only_p_c = True
    chroms = list(range(1,23))
    chroms[:] = ["chr" + str(chr_number) for chr_number in chroms]
    
    cool_ref = cooler.Cooler(control_path + "::resolutions/"+str(bin_size))
    cool_sample = cooler.Cooler(sample_path + "::resolutions/"+str(bin_size))
    ##get depth for all chromosomes
    pool = multiprocessing.Pool(n_cpus)
    ## open hic data for patient and control in sparse format
    sample_data_raw = scipy.sparse.triu(cool_sample.matrix(balance=False, sparse=True).fetch(chrom_search))
    ref_data_raw = scipy.sparse.triu(cool_ref.matrix(balance=False, sparse=True).fetch(chrom_search))
    # Use only coordinates where contacts are not zeroes
    if use_only_p_c == True:
        max_coord_const = np.int64(10**7)
        _,_, same_coords_inds = np.intersect1d(np.int64(sample_data_raw.row)*max_coord_const+np.int64(sample_data_raw.col),
        np.int64(ref_data_raw.row)*max_coord_const+np.int64(ref_data_raw.col), assume_unique=True, return_indices=True)
        if len(same_coords_inds) != len(sample_data_raw.data):
            ref_data_raw = (ref_data_raw+sample_data_raw).tocoo()
            _,_, same_coords_inds = np.intersect1d(np.int64(sample_data_raw.row)*max_coord_const+np.int64(sample_data_raw.col),
            np.int64(ref_data_raw.row)*max_coord_const+np.int64(ref_data_raw.col), assume_unique=True, return_indices=True)
        ref_data_raw.data = ref_data_raw.data[same_coords_inds]
        ref_data_raw.row = ref_data_raw.row[same_coords_inds]
        ref_data_raw.col = ref_data_raw.col[same_coords_inds]
    assert ref_data_raw.data.shape ==sample_data_raw.data.shape
    # Replace contacts to values from binomial distribution
    sample_data_raw.data = binom.pmf(sample_data_raw.data, big_depth_sample, ref_data_raw.data/big_depth_ref)
    ref_data_raw.data = binom.pmf(ref_data_raw.data, big_depth_ref, ref_data_raw.data/big_depth_ref)
    diff_array = sample_data_raw
    diff_array.data = np.log(sample_data_raw.data/ref_data_raw.data)
    
    diff_array_csr = diff_array.tocsr()
    # define maximum diagonal near the chromosomes edges
    if max_diag > diff_array_csr.get_shape()[0] - sweet_size:
        max_diag = diff_array_csr.get_shape()[0] - sweet_size
    pool = multiprocessing.Pool(n_cpus)
    # print(diff_array_csr.get_shape())    
    result = pool.map(partial(calculate_sweet_sum_on_diag_sparse), range(min_diag,max_diag))
    row, col, data =[], [], []
    for diag_coord_data in result:
        data+=diag_coord_data[2]
        # print(data)
        row_cols = [[i for i, j in diag_coord_data[1]],
                    [j for i, j in diag_coord_data[1]]]
        row+=row_cols[0]
        col+=row_cols[1]
    with open(out_dir +"/temp/sweet_metric_"+str(sweet_size)+"_"+str(bin_size//1000)+"Kb_"+chrom_search+".pickle", 'wb') as f:
        pickle.dump( np.array([np.array(row), np.array(col), np.array(data)]), f)
if __name__ == "__main__":
    main()