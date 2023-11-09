import h5py
import numpy as np
import scipy.sparse as sparse
import pandas as pd

# This function loads hic data as csr matrix for one pair of chromosomes form h5 file
def chr_pair_h5loader(
    h5_path: str, chr_pair: str, resolution=int(4096e3), batch=int(1e6)
) -> sparse.csr_matrix:
    h5_list = [h5_path] if type(h5_path) is str else h5_path
    f_it = True
    rev = False
    for h5_p in h5_list:
        with h5py.File(h5_p, "r") as h5file:
            if chr_pair not in h5file["data"].keys():
                chr_pair_new = f"{chr_pair.split('-')[1]}-{chr_pair.split('-')[0]}"
                rev = True
                if chr_pair_new not in h5file["data"].keys():
                    raise KeyError(
                        f"Chromosome pairs {chr_pair} and {chr_pair_new} are not in {h5_path}"
                    )
                else:
                    chr_pair = chr_pair_new
            assert (
                h5file["data"][chr_pair]["row"].shape[0]
                == h5file["data"][chr_pair]["col"].shape[0]
            ), f"Broken data! Len of row and col coords should be the same"
            chr1_size = int(
                np.ceil(h5file["chr_sizes"][chr_pair.split("-")[0]][0] / resolution)
            )
            chr2_size = int(
                np.ceil(h5file["chr_sizes"][chr_pair.split("-")[1]][0] / resolution)
            )
            if f_it:
                mat = sparse.csr_matrix((chr1_size, chr2_size), dtype=np.int)

                f_it = False

            if batch < h5file["data"][chr_pair]["row"].shape[0]:
                for i in range(
                    int(np.ceil(h5file["data"][chr_pair]["row"].shape[0] / batch))
                ):
                    row = (
                        np.array(
                            h5file["data"][chr_pair]["row"][
                                i * batch : (i + 1) * batch
                            ],
                            dtype=np.int,
                        )
                        // resolution
                    )
                    col = (
                        np.array(
                            h5file["data"][chr_pair]["col"][
                                i * batch : (i + 1) * batch
                            ],
                            dtype=np.int,
                        )
                        // resolution
                    )
                    mat = mat + sparse.csr_matrix(
                        (np.ones_like(row, dtype=np.int), (row, col)),
                        shape=(chr1_size, chr2_size),
                    )
            else:
                row = (
                    np.array(h5file["data"][chr_pair]["row"][:], dtype=np.int)
                    // resolution
                )
                col = (
                    np.array(h5file["data"][chr_pair]["col"][:], dtype=np.int)
                    // resolution
                )
                mat += sparse.csr_matrix(
                    (np.ones_like(row, dtype=np.int), (row, col)),
                    shape=(chr1_size, chr2_size),
                )

    if rev:
        mat = mat.T
    return mat


# This function calculates sum of all contacts existing in sample and reference
def calculate_depth_on_all_chroms(
    bin_size: int, chrom: str, sample_path: str, ref_path: str, use_only_p_c=True
) -> tuple:
    # open hic data for patient and control in sparse format for defined chromosome and resolution
    sample_data_raw = chr_pair_h5loader(
        sample_path, chrom + "-" + chrom, bin_size
    ).tocoo()
    ref_data_raw = chr_pair_h5loader(ref_path, chrom + "-" + chrom, bin_size).tocoo()
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
