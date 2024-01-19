import pandas as pd
import numpy as np
import os
import pickle
import sys
from numpy import inf
import logging

logging.basicConfig(level=logging.INFO)

# This function merge intervals if they're overlapped
def merge_intervals(intervals: np.array) -> np.array:
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    interval_index = 0
    for i in sorted_intervals:
        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [
                sorted_intervals[interval_index][0],
                i[1],
            ]
    return sorted_intervals[: interval_index + 1]


# This function merge inversion predictions at different resolutions
def merge_data_by_res(data: pd.DataFrame) -> pd.DataFrame:
    datas_by_chrom = []
    # find overlapping intervals at each chromosome and choose prediction at the most highest resolution
    for chrom in pd.unique(data["chrom"]):
        data_chrom = data[data["chrom"] == chrom]
        for i in range(len(data_chrom)):
            interval_coord = pd.Interval(
                left=data_chrom.iloc[i][data_chrom.columns.get_loc("pred_inv_coord1")],
                right=data_chrom.iloc[i][data_chrom.columns.get_loc("pred_inv_coord2")],
                closed="both",
            )
            intersection = data_chrom.index.overlaps(interval_coord)
            intersection_data = data_chrom[intersection].sort_values(
                by=["resolution", "z-score"]
            )
            datas_by_chrom.append(intersection_data.iloc[0, :])
    # merge data for all chromosomes in one dataframe
    merge_data = pd.concat(datas_by_chrom, axis=1).T
    merge_data = merge_data.drop_duplicates()
    return merge_data


# This function defines inversion breakpoints searching points with "sweet"statistic less than threshold
def get_inv_coordinates(
    out_dir: str, resolutions: list, threshold_dict: dict, sweet_sizes: list
) -> pd.DataFrame:

    chroms = list(range(1, 23))
    chroms[:] = ["chr" + str(chr_number) for chr_number in chroms]

    res_pred_datas = []
    # find breakpoints for each resolution
    for i, bin_size in enumerate(resolutions):
        sweet_size = sweet_sizes[i]
        thr = int(threshold_dict[str(bin_size)])
        z_scores_by_chr = {}
        means_for_z_score = []
        stds_for_z_score = []
        # calculate mean and std of "sweet" statistic for all chromosomes
        for chrom in chroms:
            # load "sweet" statistic data. sweet_chr is a list of three numpy arrays: row values, column values and "sweet" statistic values for row-column coordinates
            with open(
                out_dir
                +"/temp/"
                + "sweet_metric_"
                + str(sweet_size)
                + "_"
                + str(bin_size // 1000)
                + "Kb_"
                + chrom
                + ".pickle",
                "rb",
            ) as fin:

                sweet_chr = pickle.load(fin)
                sweet_chr[-1] = np.nan_to_num(sweet_chr[-1])
                sweet_chr[-1][sweet_chr[-1] == -inf] = 0
                sweet_chr[-1][sweet_chr[-1] == inf] = 0
                means_for_z_score.append(np.mean(sweet_chr[-1]))
                stds_for_z_score.append(np.std(sweet_chr[-1]))
                z_scores_by_chr[chrom] = sweet_chr[-1]
        std_sweet_by_all_chromosomes = np.mean(stds_for_z_score)
        mean_sweet_by_all_chromosomes = np.mean(means_for_z_score)

        # calculate z-score "sweet" statistic values for each chromosome
        for chrom in chroms:
            z_scores_by_chr[chrom] = (
                z_scores_by_chr[chrom] - mean_sweet_by_all_chromosomes
            ) / std_sweet_by_all_chromosomes

        # find coordinates of inversions less than thr
        chrom_datas = []
        for chrom in chroms:
            # load "sweet" statistic data. sweet_chr is list of three numpy arrays: row values, column values and "sweet" statistic values for row-column coordinates
            with open(
                out_dir
                +"/temp/"
                + "sweet_metric_"
                + str(sweet_size)
                + "_"
                + str(bin_size // 1000)
                + "Kb_"
                + chrom
                + ".pickle",
                "rb",
            ) as fin:
                sweet_chr = pickle.load(fin)
                sweet_chr[-1] = np.nan_to_num(sweet_chr[-1])
                sweet_chr[-1][sweet_chr[-1] == -inf] = 0
                sweet_chr[-1][sweet_chr[-1] == inf] = 0
                inv_coords_on_chr = np.array(
                    list(
                        zip(
                            sweet_chr[0][np.where(z_scores_by_chr[chrom] < thr)],
                            sweet_chr[1][np.where(z_scores_by_chr[chrom] < thr)],
                        )
                    )
                )
                # create dataframe row and column values whee z-score "sweet" statistic is less than threshold
                temp_data = pd.DataFrame(
                    data={
                        "chrom": [chrom] * inv_coords_on_chr.shape[0],
                        "pred_inv_coord1": sweet_chr[0][
                            np.where(z_scores_by_chr[chrom] < thr)
                        ],
                        "pred_inv_coord2": sweet_chr[1][
                            np.where(z_scores_by_chr[chrom] < thr)
                        ],
                        "z-score": z_scores_by_chr[chrom][z_scores_by_chr[chrom] < thr],
                    }
                )
                # create numpy array with inversion breakpoints
                intervals = np.array(
                    list(
                        zip(temp_data["pred_inv_coord1"], temp_data["pred_inv_coord2"])
                    )
                )
                # merge overlapping breakpoints
                inv_merge_intervals = merge_intervals(intervals)
                # extract only one pair of inversion breakpoints from overlapping predictions usig the lowest "sweet" statistic z-score
                for interval in inv_merge_intervals:
                    all_data_in_merge_interval = temp_data[
                        (temp_data["pred_inv_coord1"] >= interval[0])
                        & (temp_data["pred_inv_coord2"] <= interval[1])
                    ]
                    all_data_in_merge_interval.sort_values(
                        by=["z-score"], ascending=True, inplace=True
                    )
                    chrom_datas.append(all_data_in_merge_interval.iloc[0, :])
        # if some inversions were predicted at this resolutions add them to predicted dataframe
        if len(chrom_datas) > 0:
            pred_inv_data = pd.concat(chrom_datas, axis=1).T
            pred_inv_data["resolution"] = [bin_size] * len(pred_inv_data)
            pred_inv_data["pred_inv_coord1"] = pred_inv_data["pred_inv_coord1"].apply(
                lambda x: int(x) * bin_size
            )
            pred_inv_data["pred_inv_coord2"] = pred_inv_data["pred_inv_coord2"].apply(
                lambda x: int(x) * bin_size
            )
            pred_inv_data.to_csv(
                out_dir
                +"/temp/" 
                + "pred_inversions_sweet_thr"
                + str(thr)
                + "_z-score_all_chr_"
                + str(bin_size // 1000)
                + "Kb.txt",
                sep="\t",
                index=False,
            )
            res_pred_datas.append(pred_inv_data)
    # merge predictions from different resolutions
    if len(res_pred_datas) > 0:
        data_rearr = pd.concat(res_pred_datas)
        data_rearr.set_index(
            data_rearr.apply(
                lambda x: pd.Interval(x.pred_inv_coord1, x.pred_inv_coord2),
                axis="columns",
            ),
            inplace=True,
        )
        data_rearr = merge_data_by_res(data_rearr)
        # convert prediction to final format
        data_final = pd.DataFrame(
            data={
                "chrom1": data_rearr["chrom"],
                "brp1": data_rearr["pred_inv_coord1"],
                "chrom2": data_rearr["chrom"],
                "brp2": data_rearr["pred_inv_coord2"],
                "type": ["inversion"] * len(data_rearr),
                "resolution": [bin_size] * len(data_rearr)
            }
        )
        data_final.to_csv(out_dir + "/temp/merge_res_inversions.txt", sep="\t", index=False)
        return data_final

    else:
        data_rearr = pd.DataFrame(
            columns=[
                "chrom",
                "pred_inv_coord1",
                "pred_inv_coord2",
                "z-score",
                "resolution",
            ]
        )
        logging.getLogger(__name__).info("This sample doesn't have predicted inversions")
        return data_rearr
