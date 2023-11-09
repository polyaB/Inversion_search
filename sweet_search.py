import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing
from functools import partial
import warnings
from scipy.stats import binom
from scipy.sparse import csr_matrix
import sys
import pickle
import gc
import h5py
import scipy.sparse as sparse
from additional_functions import chr_pair_h5loader
import logging

logging.basicConfig(level=logging.INFO)

# this function calculate "sweet" statistic values for all contacts at one diagonal from hic contact matrix
def calculate_sweet_sum_on_diag_sparse(diag: int) -> tuple:
    coords = []
    data = []
    if diag % 100 == 0:
        logging.getLogger(__name__).info(str(diag) + " " + str(datetime.now()))

    # calculate "sweet" statistic for each contact at the diagonal
    for i in range(diag, diff_array.shape[0], step):
        # calculate sum of values for left triangle of "sweet"
        left_sweet_square = diff_array_csr[
            i - diag - sweet_size : i - diag, i - sweet_size : i
        ].toarray()
        left_sweet_sum = np.sum(np.triu(np.rot90(left_sweet_square, 1)))
        # calculate sum of values for right triangle of "sweet"
        right_sweet_square = diff_array_csr[
            i - diag : i - diag + sweet_size, i : i + sweet_size
        ].toarray()
        right_sweet_sum = np.sum(np.triu(np.rot90(right_sweet_square, 3)))
        # save row and column values of this hic contact
        coords.append((i - diag, i))
        # save "sweet" statistic value for this hic contact
        data.append(left_sweet_sum + right_sweet_sum)
    return diag, coords, data


global diff_array_csr, use_only_p_c, sweet_size, step, out_dir, cool_ref, sample_path

# sample name
sample = sys.argv[1]
sample_path = sys.argv[2]
ref_path = sys.argv[3]
# chromosome name for which is "sweet" statistic values are calculating
chrom_search = sys.argv[5]
# resolution of contact matrix
bin_size = int(sys.argv[4])
# Size of mask using to calculate "sweet" statistic
sweet_size = int(sys.argv[6])
# sum of all contacts for sample
big_depth_sample = int(sys.argv[7])
# sum of all contacts for reference
big_depth_ref = int(sys.argv[8])
# number of cpus
n_cpus = int(sys.argv[9])
# step between two points
step = 1
# Data describing minimum and maximum sizes of inversions which we are going to search at different resolutions
res_size_data = pd.read_csv("./res_size.txt", sep="\t")

logging.getLogger(__name__).info(
    sample
    + " start calculate sweet for chrom "
    + chrom_search
    + " with bin size "
    + str(bin_size)
)
# use only contacts existing as in sample as in reference
use_only_p_c = True
out_dir = "./temp/"
chroms = list(range(1, 23))
chroms[:] = ["chr" + str(chr_number) for chr_number in chroms]
# define minimum and maximum diagonal of hic matrix where we are going to calculate "sweet" statistic
min_diag = res_size_data[res_size_data["resolution"] == bin_size].iloc[0, 1] // bin_size
max_diag = res_size_data[res_size_data["resolution"] == bin_size].iloc[0, 2] // bin_size
#  open hic data for patient and control in sparse format
sample_data_raw = chr_pair_h5loader(
    sample_path, chrom_search + "-" + chrom_search, bin_size
).tocoo()
ref_data_raw = chr_pair_h5loader(
    ref_path, chrom_search + "-" + chrom_search, bin_size
).tocoo()
logging.getLogger(__name__).info("find same coord")
# use only contacts existing as in sample as in reference
if use_only_p_c == True:
    max_coord_const = np.int64(10**7)
    _, _, same_coords_inds = np.intersect1d(
        np.int64(sample_data_raw.row) * max_coord_const + np.int64(sample_data_raw.col),
        np.int64(ref_data_raw.row) * max_coord_const + np.int64(ref_data_raw.col),
        assume_unique=True,
        return_indices=True,
    )
    if len(same_coords_inds) != len(sample_data_raw.data):
        ref_data_raw = (ref_data_raw + sample_data_raw).tocoo()
        _, _, same_coords_inds = np.intersect1d(
            np.int64(sample_data_raw.row) * max_coord_const
            + np.int64(sample_data_raw.col),
            np.int64(ref_data_raw.row) * max_coord_const + np.int64(ref_data_raw.col),
            assume_unique=True,
            return_indices=True,
        )
    ref_data_raw.data = ref_data_raw.data[same_coords_inds]
    ref_data_raw.row = ref_data_raw.row[same_coords_inds]
    ref_data_raw.col = ref_data_raw.col[same_coords_inds]
assert ref_data_raw.data.shape == sample_data_raw.data.shape
logging.getLogger(__name__).info("calc binom")
# replace all raw contact values to the probabilities calculated using probability mass function for binomial distribution
sample_data_raw.data = binom.pmf(
    sample_data_raw.data, big_depth_sample, ref_data_raw.data / big_depth_ref
)
ref_data_raw.data = binom.pmf(
    ref_data_raw.data, big_depth_ref, ref_data_raw.data / big_depth_ref
)
# divide all sample contact values by all reference contact values and use logarithmic values
diff_array = sample_data_raw
logging.getLogger(__name__).info("log")
diff_array.data = np.log(sample_data_raw.data / ref_data_raw.data)
logging.getLogger(__name__).info("create diff_array_csr")
# convert coo to csr matrix
diff_array_csr = diff_array.tocsr()
logging.getLogger(__name__).info("create sweet")
# define maximum diagonal near the chromosomes edges
if max_diag > diff_array_csr.get_shape()[0] - sweet_size:
    max_diag = diff_array_csr.get_shape()[0] - sweet_size
# calculate "sweet" statistic for all contacts at diagonals from minimum diagonal to maximum diagonal at this resolution
pool = multiprocessing.Pool(n_cpus)
logging.getLogger(__name__).info(
    "bin_size "
    + str(bin_size)
    + " min_diag "
    + str(min_diag)
    + " max_diag "
    + str(max_diag)
)
result = pool.map(
    partial(calculate_sweet_sum_on_diag_sparse), range(min_diag, max_diag)
)
row, col, data = [], [], []
# reformat "sweet" statistic data to three lists: row, column, "sweet" statistic value
for diag_coord_data in result:
    data += diag_coord_data[2]
    row_cols = [[i for i, j in diag_coord_data[1]], [j for i, j in diag_coord_data[1]]]
    row += row_cols[0]
    col += row_cols[1]
# save "sweet" statistic data
with open(
    out_dir
    + sample
    + "_sweet_"
    + str(sweet_size)
    + "_"
    + str(bin_size // 1000)
    + "Kb_"
    + chrom_search
    + ".pickle",
    "wb",
) as f:
    pickle.dump(np.array([np.array(row), np.array(col), np.array(data)]), f)

del diff_array_csr
del result
del sample_data_raw
del ref_data_raw
gc.collect()

logging.getLogger(__name__).info(str(datetime.now()))
