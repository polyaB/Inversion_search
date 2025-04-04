import pandas as pd
from subprocess import call
import os
from source.get_inv_coordinates import get_inv_coordinates
from source.additional_functions import calculate_depth_on_all_chroms, delete_repeat_inversions
import numpy as np
import logging
import shutil
from datetime import datetime
import argparse
logging.basicConfig(level=logging.INFO)


def parse_line(command_line):
    global logs_dir, leave_logs, calc_cis
    # Command line parser blocks
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--dir", help="working directory where all predictions will be saved", type=str, required=True)
    parser.add_argument("-s", "--sample", help="path to sample cool file", type=str, required=True)
    parser.add_argument("-c", "--control", help="path to control cool file", type=str, required=True)
    parser.add_argument("-n", "--nproc", help="number of threads", type=int, required=True, default=1)
    parser.add_argument("-r", "--resolutions", help="list of resolutions for inversion search", nargs='*', type=int, default=[1000000, 250000, 100000, 10000])
    parser.add_argument("--thr_inv", help="list of thresholds for inversion 'sweet' metric according to resolutions", nargs='*', type=int, default=[-18, -20, -19, -15])
    parser.add_argument("--sweet_sizes", help="list of sizes for 'sweet' metric according to resolutions", nargs='*', type=float, default=[5, 10, 10, 20])
    parser.add_argument("--clarify_coord", help="this parameter enables to clarify predicted breakpoints coordinates in 10 Kb resolution", action = 'store_true')
    parser.add_argument("--not_del_temp", help="not delete temporary directory", action='store_true')
    parser.set_defaults(not_del_temp=False, clarify_coord=False)
    args = parser.parse_args(command_line)

    return (os.path.realpath(args.dir),
            os.path.realpath(args.sample), 
            os.path.realpath(args.control), 
            args.nproc, 
            args.resolutions,
            args.thr_inv,
            args.sweet_sizes,
            args.clarify_coord,
            args.not_del_temp)

def main(command_line=None):
    (workdir,
     sample_file, 
     control_file, 
     nproc,
     resolutions,
     thresholds,
     sweet_sizes, 
     clarify_coord,
     not_del_temp) = parse_line(command_line)

    sweet_sizes_dict = {"1000000":"5", "250000":"10", "100000":"10", "10000":"20"}
    thresholds_dict = {"1000000":"-18", "250000":"-20", "100000":"-19", "10000":"-15"}
    #if user set own sweet sizes or thresholdes change dict
    if len(resolutions)  == len(sweet_sizes):
        for i, res in enumerate(resolutions):
            sweet_sizes_dict[str(res)] = str(sweet_sizes[i])        
    
    if len(resolutions) == len(thresholds):
        for i, res in enumerate(resolutions):
            thresholds_dict[str(res)] =  str(thresholds[i])
    
    # Create directory for data if it doesn't exist
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    # Create temp directory
    if not os.path.exists(workdir + "/temp/"):
        os.makedirs(workdir +"/temp/")

    chroms = list(range(1, 23))
    chroms[:] = ["chr" + str(chr_number) for chr_number in chroms]
    
    for i, bin_size in enumerate(resolutions):
        # calculate depth for sample and reference
        logging.getLogger(__name__).info(
            "calculate depth for sample  "
            + sample_file
            + " for bin size "
            + str(bin_size)
            + str(datetime.now())
        )
        sample_ref_depths = []
        for chrom in chroms:
            sample_ref_depths.append(
                calculate_depth_on_all_chroms(
                    bin_size, chrom, sample_file, control_file
                )
            )

        depths = [
            [i for i, j in sample_ref_depths],
            [j for i, j in sample_ref_depths],
        ]

        big_depth_sample, big_depth_ref = np.sum(depths[0]), np.sum(depths[1])
        logging.getLogger(__name__).info(
            "calculate 'sweet' metric  with resolution " + str(bin_size) + " " + str(datetime.now())
        )
        # calculate "sweet" metric for each chromosome
        for chrom in chroms:
            logging.getLogger(__name__).info(
            "for  " + chrom +  " with resolution " + str(bin_size) + " " + str(datetime.now())
        )
            call(
                [
                    "python",
                    "./source/sweet_search.py",
                    sample_file,
                    control_file,
                    str(bin_size),
                    str(sweet_sizes_dict[str(bin_size)]),
                    chrom,
                    str(big_depth_sample),
                    str(big_depth_ref),
                    str(nproc),
                    workdir
                ]
            )
    # Define inversion breakpoints using "sweet" statistic
    logging.getLogger(__name__).info(
            "predict inversion breakpoints for different resolutions ")
    
    sweet_sizes = []
    #update sweet sizes from dict
    for res in resolutions:
        sweet_sizes.append(sweet_sizes_dict[str(res)])
    inv_sample_data = get_inv_coordinates(workdir, resolutions, thresholds_dict, sweet_sizes)
    # Clarify coordinates of inversions using maximum 5Kb resolution 
    
    if clarify_coord:
        logging.getLogger(__name__).info(
            "clarify coordinates of breakpoints to 10Kb resolution")
        call(
                [
                    "python",
                    "./source/clarify_coordinates.py",
                    sample_file,
                    control_file,
                    str(bin_size),
                    str(big_depth_sample),
                    str(big_depth_ref),
                    str(nproc),
                    workdir,
                ]
            )
        final_inversion_data = pd.read_csv(workdir + "/temp/merge_res_pred_inv_clarified.txt", sep="\t")
    else:
        final_inversion_data = inv_sample_data
    # Remove temp directory
    if not not_del_temp:
        shutil.rmtree(workdir + "/temp/")

    final_inversion_data.to_csv(workdir + "/inversion_prediction.txt", index=False, sep="\t")
if __name__ == "__main__":
    main()
