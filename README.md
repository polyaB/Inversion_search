# Inversion_search
This repository contains scripts for searching inversions in Exo-C data. You need hic files for sample and control data in .mcool format. 
## Requirements
* python >= 3.6.10
* cooler==0.8.11
* numpy==1.20.1
* pandas==1.2.4
* scikit_learn==0.24.1
* scipy==1.6.2
## Download
```
git clone https://github.com/polyaB/Inversion_search.git
```
## Installation
You can create virtual environment with conda and install all dependies using following commands:
```
conda create 
```
## Usage
In order to use Inversion_search, you need to first activate the virtual environment that you created before:
```
conda activate
```
You can download example data for sample and control:
```
cd Inversion_search
mkdir example_data
wget https://genedev.bionet.nsc.ru/ftp/by_Project/ExoC/inversion_search/data/control.mcool
wget https://genedev.bionet.nsc.ru/ftp/by_Project/ExoC/inversion_search/data/s176_P82.mcool
Then just run the main script with required arguments.
```
cd Inversion_search
 python find_inversions.py -d ./test_P82/ -s ./data_example/s176_P82.mcool -c ./data_example/control.mcool -n 2 -r 1000000
```
There 


