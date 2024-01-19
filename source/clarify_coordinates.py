import cooler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import subprocess
from datetime import datetime
import multiprocessing
from functools import partial
import warnings
from scipy.stats import binom
from scipy.sparse import csr_matrix
import scipy
import sys
import warnings
import logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


def calculate_sweet_sum_on_diag_sparse(diag):
    coords = []
    data = []
    for i in range(diag, diff_array_csr.shape[0], step):
        left_sweet_square = diff_array_csr[i-diag-sweet_size:i-diag, i-sweet_size:i].toarray()
        right_sweet_square = diff_array_csr[i-diag:i-diag+sweet_size, i:i+sweet_size].toarray() 
        left_sweet_sum = np.sum(np.triu(np.rot90(left_sweet_square, 1)))
        right_sweet_sum = np.sum(np.triu(np.rot90(right_sweet_square, 3)))
        coords.append((i-diag, i))
        data.append(left_sweet_sum+right_sweet_sum)
    return diag, coords, data

def main():
    global diff_array_csr, cool_sample, cool_ref, use_only_p_c, sweet_size, step, sample, out_dir

    sample_path = sys.argv[1]
    control_path = sys.argv[2]
    bin_size = int(sys.argv[3])
    big_depth_sample = int(sys.argv[4])
    big_depth_ref = int(sys.argv[5])
    n_cpus = int(sys.argv[6])
    out_dir = sys.argv[7]
    resolutions = [1000000, 250000, 100000, 10000]
    sweet_sizes = [5, 10, 10, 20]
    step = 1

    use_only_p_c = True
    chroms = list(range(1,23))
    chroms[:] = ["chr" + str(chr_number) for chr_number in chroms]
    n_bins = 10
    pred_inv_data = pd.read_csv(out_dir + "/temp/merge_res_inversions.txt", sep="\t")
    if len(pred_inv_data)>0:
        brps1 = []
        brps2 = []
        for i in range(len(pred_inv_data)):
            logging.getLogger(__name__).info(
            "... clarifying " + str(i+1)+ " out of "+ str(len(pred_inv_data)) + " predicted inversions ...")
            chrom_search = pred_inv_data.iloc[i][pred_inv_data.columns.get_loc("chrom1")]
            pred_brp1 = int(pred_inv_data.iloc[i][pred_inv_data.columns.get_loc("brp1")])
            pred_brp2 = int(pred_inv_data.iloc[i][pred_inv_data.columns.get_loc("brp2")])
            initial_bin_size = pred_inv_data.iloc[i][pred_inv_data.columns.get_loc("resolution")]
            index_binsize = resolutions.index(initial_bin_size)
            temp_binsize = initial_bin_size
            if initial_bin_size==10000:
                pred_inv_data["brp1"] =pred_inv_data["brp1"].apply(lambda x: int(x))
                pred_inv_data["brp2"] =pred_inv_data["brp2"].apply(lambda x: int(x))

                brps1 = list(pred_inv_data["brp1"])
                brps2 = list(pred_inv_data["brp2"])
                continue
            for j in range(index_binsize+1, len(resolutions)):
                bin_size = resolutions[j]
                sweet_size = sweet_sizes[j]
                min_diag = ((pred_brp2-pred_brp1) - n_bins*temp_binsize)//bin_size
                max_diag = ((pred_brp2-pred_brp1)  + n_bins*temp_binsize)//bin_size
                
                cool_ref = cooler.Cooler(control_path + "::resolutions/"+str(bin_size))
                cool_sample = cooler.Cooler(sample_path + "::resolutions/"+str(bin_size))
                ##get depth for all chromosomes
                pool = multiprocessing.Pool(n_cpus)
                
                ## open hic data for patient and control in sparse format
                sample_data_raw = scipy.sparse.triu(cool_sample.matrix(balance=False, sparse=True).fetch(chrom_search))
                ref_data_raw = scipy.sparse.triu(cool_ref.matrix(balance=False, sparse=True).fetch(chrom_search))
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
                sample_data_raw.data = binom.pmf(sample_data_raw.data, big_depth_sample, ref_data_raw.data/big_depth_ref)
                ref_data_raw.data = binom.pmf(ref_data_raw.data, big_depth_ref, ref_data_raw.data/big_depth_ref)
                diff_array = sample_data_raw
                diff_array.data = np.log(sample_data_raw.data/ref_data_raw.data)

                diff_array_csr = diff_array.tocsr()
                pool = multiprocessing.Pool(n_cpus)
                if max_diag > diff_array_csr.get_shape()[0] - sweet_size:
                    max_diag = diff_array_csr.get_shape()[0] - sweet_size
                elif min_diag < sweet_size:
                    min_diag = sweet_size
                    
                result = pool.map(partial(calculate_sweet_sum_on_diag_sparse), range(min_diag,max_diag))
                row, col, data =[], [], []
                for diag_coord_data in result:
                    data+=diag_coord_data[2]
                    row_cols = [[i for i, j in diag_coord_data[1]],
                                [j for i, j in diag_coord_data[1]]]
                    row+=row_cols[0]
                    col+=row_cols[1]
                row, col, data = np.array(row), np.array(col), np.array(data)

                pred_brp1 = row[np.argmin(data)]*bin_size
                pred_brp2 = col[np.argmin(data)]*bin_size
                temp_binsize = bin_size
            brps1.append(int(pred_brp1))
            brps2.append(int(pred_brp2))
        pred_inv_data["brp1"] = brps1
        pred_inv_data["brp2"] = brps2
        pred_inv_data["rearr_type"] = ["inversion"]*len(pred_inv_data)
        pred_inv_data[["chrom1", "brp1", "chrom2", "brp2", "rearr_type"]].to_csv(out_dir + "/temp/merge_res_pred_inv_clarified.txt", sep="\t", index=False, header=["chrom1", "brp1", "chrom2", "brp2", "type"])
    else:
        data_to_csv = pd.DataFrame(columns=["chrom1", "brp1", "chrom2", "brp2", "type"])
        data_to_csv.to_csv(out_dir + "/temp/merge_res_pred_inv_clarified.txt", sep="\t", index=False)
if __name__ == "__main__":
    main()
        

