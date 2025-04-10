# Inversion_search
This repository contains scripts for searching inversions in Exo-C data. You need hic files for sample and control data in .mcool format. 
## Requirements
* python >= 3.6.10
* cooler==0.8.11
* numpy==1.19.4
* pandas==1.1.4
* scikit_learn==0.24.1
* scipy==1.5.4
## Download
```
git clone https://github.com/polyaB/Inversion_search.git
```
## Installation
You can create virtual environment with conda and install all dependies using following commands:
```
cd Inversion_search
mamba create --name inv_search python=3.6.10
mamba activate inv_search
pip install -r requirements.txt 
```
## Usage
In order to use Inversion_search, you need to first activate the virtual environment that you created before:
```
mamba activate inv_search
```
You can download example data for sample and control:
```
cd Inversion_search
mkdir example_data
```
control and sample for hg19:
```
wget -P ./example_data/ https://genedev.bionet.nsc.ru/ftp/by_Project/ExoC/inversion_search/data/control.mcool
wget -P ./example_data/ https://genedev.bionet.nsc.ru/ftp/by_Project/ExoC/inversion_search/data/s176_P82.mcool
```
control and sample for hg38:
```
wget -P ./example_data/ https://genedev.bionet.nsc.ru/ftp/by_Project/ExoC/hg38_files/sup_pat.XX.exoc.mcool
wget -P ./example_data/ https://genedev.bionet.nsc.ru/ftp/by_Project/ExoC/hg38_files/s132_P10_exoc.mcool
```

Then just run the main script with required arguments (We recommend using several CPUs (`-n` parameter), also for testing you can use only one resolution).
```
python find_inversions.py -d ./test_P82/ -s ./example_data/s176_P82.mcool -c ./example_data/control.mcool -n 2 -r 1000000
```
There are three required arguments: The directory for predicted data (`-d`), path to .mcool sample file (`-s`) and path to .mcool control file (`-c`).
If you want to launch script with standard parameters, just type command with required arguments. Pay attention that it needs about 8 GB memory hard drive memory for temporary files and 5 GB RAM for one cpu. It takes about ? hours for all resolutions for n cpu.
```
python find_inversions.py -d /path_to_dir/ -s /path_to_sample.mcool -c /path_to_control.mcool
```
```
usage: find_inversions.py [-h] -d DIR -s SAMPLE -c CONTROL [-n NPROC]
                          [-r [RESOLUTIONS [RESOLUTIONS ...]]]
                          [--thr_inv [THR_INV [THR_INV ...]]]
                          [--sweet_sizes [SWEET_SIZES [SWEET_SIZES ...]]]
                          [--clarify_coord] [--not_del_temp]

optional arguments:
  -h, --help            show this help message and exit
  -n NPROC, --nproc NPROC
                        number of threads
  -r [RESOLUTIONS [RESOLUTIONS ...]], --resolutions [RESOLUTIONS [RESOLUTIONS ...]]
                        list of resolutions for inversion search
  --thr_inv [THR_INV [THR_INV ...]]
                        list of thresholds for inversion 'sweet' metric
                        according to resolutions
  --sweet_sizes [SWEET_SIZES [SWEET_SIZES ...]]
                        list of sizes for 'sweet' metric according to
                        resolutions
  --clarify_coord       this parameter enables to clarify predicted
                        breakpoints coordinates in 10 Kb resolution
  --not_del_temp        not delete temporary directory

required named arguments:
  -d DIR, --dir DIR     working directory where all predictions will be saved
  -s SAMPLE, --sample SAMPLE
                        path to sample cool file
  -c CONTROL, --control CONTROL
                        path to control cool file
```
This tool works for following resolutions: 1000000, 250000, 100000, 10000 bp. It means that script search inversions using this resolution at appropriate distances which you can find in source/res_size.txt file. Script uses all these resolutions by default. You can specify resolution by `-r` argument.


