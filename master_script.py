import pandas as pd
from subprocess import call
import os
from get_inv_coordinates import get_inv_coordinates
from additional_functions import calculate_depth_on_all_chroms, delete_repeat_inversions
import numpy as np
import logging
import shutil
from datetime import datetime


logging.basicConfig(level=logging.INFO)


def predict_inversions(metadata):
    # Create temp directory
    if not os.path.exists("./temp/"):
        os.makedirs("./temp/")
    samples = pd.unique(metadata["sample_name"])
    # path to file with a reference sample
    ref_path = "/mnt/scratch/ws/talagunov/202310191211talagunov13/sber/test/control.h5"
    chroms = list(range(1, 23))
    chroms[:] = ["chr" + str(chr_number) for chr_number in chroms]
    # Define resolutions, size of "sweet" and thresholds for inversion search
    resolutions = [1000000, 250000]
    sweet_sizes = [5, 10]
    thresholds = [-11, -20]
    n_cpus = 10
    inv_sample_datas = []
    for i, sample in enumerate(samples):
        sample_path = metadata[metadata["sample_name"] == sample].iloc[
            0, metadata.columns.get_loc("data_path")
        ]
        for i, bin_size in enumerate(resolutions):
            # calculate depth for sample and reference
            logging.getLogger(__name__).info(
                "calculate depth for sample  "
                + sample
                + " "
                + str(bin_size)
                + str(datetime.now())
            )
            sample_ref_depths = []
            for chrom in chroms:
                sample_ref_depths.append(
                    calculate_depth_on_all_chroms(
                        bin_size, chrom, sample_path, ref_path
                    )
                )

            depths = [
                [i for i, j in sample_ref_depths],
                [j for i, j in sample_ref_depths],
            ]

            big_depth_sample, big_depth_ref = np.sum(depths[0]), np.sum(depths[1])
            logging.getLogger(__name__).info(
                "calculate sweet for sample  " + str(datetime.now())
            )
            # calculate "sweet" statistic for each chromosome
            for chrom in chroms:
                call(
                    [
                        "python",
                        "sweet_search.py",
                        sample,
                        sample_path,
                        ref_path,
                        str(bin_size),
                        chrom,
                        str(sweet_sizes[i]),
                        str(big_depth_sample),
                        str(big_depth_ref),
                        str(n_cpus),
                    ]
                )
        # Define inversion breakpoints using "sweet" statistic
        inv_sample_data = get_inv_coordinates(
            sample, resolutions, thresholds, sweet_sizes
        )
        if len(inv_sample_data) > 0:
            inv_sample_datas.append(inv_sample_data)
    # Concat predicted inversions for all samples
    inversion_table = pd.concat(inv_sample_datas)
    final_inversion_data = delete_repeat_inversions(inversion_table)
    shutil.rmtree("./temp/")
    return final_inversion_data


metadata = pd.read_csv("./private_metadata.tsv", sep="\t")
print(metadata)
inversion_prediction_data = predict_inversions(metadata)

inversion_prediction_data.to_csv("./inversion_for_private.txt", index=False, sep="\t")
