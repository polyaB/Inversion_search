import numpy as np
import scipy.sparse as sparse
import pandas as pd
import cooler

# This function calculates sum of all contacts existing in sample and reference
def calculate_depth_on_all_chroms(
    bin_size: int, chrom: str, sample_path: str, ref_path: str, use_only_p_c=True
) -> tuple:
    # open hic data for patient and control in sparse format for defined chromosome and resolution
    cool_sample = cooler.Cooler(sample_path + "::resolutions/"+str(bin_size))
    cool_ref = cooler.Cooler(ref_path + "::resolutions/"+str(bin_size))
    sample_data_raw = cool_sample.matrix(balance=False, sparse=True).fetch(chrom)
    ref_data_raw = cool_ref.matrix(balance=False, sparse=True).fetch(chrom)
    # use only contacts existing as in sample as in reference
    if use_only_p_c == True:
        max_coord_const = np.int64(10**7)
        _, _, same_coords_inds = np.intersect1d(
            np.int64(sample_data_raw.row) * max_coord_const
            + np.int64(sample_data_raw.col),
            np.int64(ref_data_raw.row) * max_coord_const + np.int64(ref_data_raw.col),
            assume_unique=True,
            return_indices=True,
        )
        if len(same_coords_inds) != len(sample_data_raw.data):
            ref_data_raw = (ref_data_raw + sample_data_raw).tocoo()
            _, _, same_coords_inds = np.intersect1d(
                np.int64(sample_data_raw.row) * max_coord_const
                + np.int64(sample_data_raw.col),
                np.int64(ref_data_raw.row) * max_coord_const
                + np.int64(ref_data_raw.col),
                assume_unique=True,
                return_indices=True,
            )
        ref_data_raw.data = ref_data_raw.data[same_coords_inds]
        ref_data_raw.row = ref_data_raw.row[same_coords_inds]
        ref_data_raw.col = ref_data_raw.col[same_coords_inds]

    depth_sample = np.sum(sample_data_raw.data)
    depth_ref = np.sum(ref_data_raw.data)
    return depth_sample, depth_ref


def delete_repeat_inversions(inv_table: pd.DataFrame) -> pd.DataFrame:
    wo_dup_table = inv_table.drop_duplicates(
        subset=["chrom1", "brp1", "chrom2", "brp2"]
    )
    return wo_dup_table
