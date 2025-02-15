# Postprocessing of Ensemble Weather Forecasts Using Permutation-invariant Neural Networks

[![DOI](https://zenodo.org/badge/689015596.svg)](https://zenodo.org/badge/latestdoi/689015596)

Code for the paper "Postprocessing of Ensemble Weather Forecasts Using Permutation-invariant Neural Networks" by Kevin Höhlein, Benedikt Schulz, Rüdiger Westermann and Sebastian Lerch.

## Installation

The main part of the project is written in Python. The repository contains a file [`requirements.txt`](requirements.txt), from which a virtual Python environment can be built as follows:

    python -m venv venv
    source venv/bin/activate
    pip install wheel
    pip install -r requirements.txt

Parts of the evaluation rely on the availability of an R installation. We recommend e.g. using Anaconda to install R and running R subsequently to install required packages. The R dependencies originate from the use of baseline evaluation metrics from a third-party [repository](https://github.com/benediktschulz/paper_pp_wind_gusts). We refer to this one for further information.

## Datasets

The paper investigates the utility of permutation-invariant neural networks for statistical postprocessing of ensemble weather forecasts using two forecast-observation datasets for wind-gust (COSMO-DE dataset) and surface temperature postprocessing (EUPPBench dataset). 
Training own models requires downloading parts of the data and listing the data location in path config files for it to be found by the training scripts. 

Researchers interested in reproducing our results for comparisons against their own models should additionally consider the known issues section at the end of this document. 

### Downloading data

The COSMO-DE data is proprietary and can only be retrieved directly from the German weather service (DWD).
The EUPPBench dataset is open access and consists of two parts, the reforecast and the forecast datasets. 
Scripts for downloading both parts to disk in zarr format are provided [here](data/euppbench/download_reforecasts.py) and [here](data/euppbench/download_forecasts.py) and are used as follows:

    python data/euppbench/download_reforecasts.py --path /path/to/reforecasts
    python data/euppbench/download_forecasts.py --path /path/to/forecasts  # optional if no evaluation on forecast data is intended

Memory requirements amount to roughly 10.0 GB for the reforecast data and 8.5 GB for the forecast data. 

Note that the training scripts cannot be run from within the standard Python environment of the project due to package version conflicts between `climetlab` and other dependencies. A `conda` environment for the download can be built from [`environment_download.yml`](environment_download.yml). 
For further information, we refer to the documentations of [CliMetLab](https://climetlab.readthedocs.io/en/latest/) and [EUPPBench](https://eupp-benchmark.github.io/EUPPBench-doc/files/EUPPBench_datasets.html#).

### Path configuration interface

Access to the downloaded data is managed automatically using `.json`-based path configuration files in [`data/config`](data/config). 
Paths to the dataset base directories should be listed in the respective configuration files for [reforecast](data/config/euppbench_reforecasts_data.json) and [forecast](data/config/euppbench_forecasts_data.json) datasets as a mapping of host name to data path. 
An example configuration is shown in [`example.json`](data/config/example.json).

To check the host name as used by the program, run `python -c "import socket; print(socket.gethostname())"`.

### Caching preprocessed data

Loading the data from disk and preprocessing it for training may take a few minutes. 
To speed up this process in repeated training runs, a preprocessed version of the dataset can be cached and loaded directly when re-running scripts with identical data settings. 
To use this feature, paths to caching directories for reforecast and forecast data can be specified in [`euppbench_reforecasts_cache.json`](data/config/euppbench_reforecasts_cache.json) and [`euppbench_forecasts_cache.json`](data/config/euppbench_forecasts_cache.json), respectively.

### Exporting CSV files for baseline methods

Training of the baseline methods (DRN, BQN, EMOS) requires summarization and export of the data to `.csv`-format. 
Data in the required form can be exported by running the scripts [`export_reforecasts_csv.py`](data/euppbench/export_reforecasts_csv.py) and [`export_forecasts_csv.py`](data/euppbench/export_forecasts_csv.py), which write out two files, containing the training (and validation) dataset and the test set, respectively. 
The target directory can be specified by passing the command line argument `--csv-directory /path/to/target/dir` when running the script. Note that the generated training data are identical in both cases.

## Reproducing experiments

### Training ensemble-based models

The main entry point for training ensemble-based models are [`experiments/cosmo_de/run_training.py`](experiments/cosmo_de/run_training.py) and [`experiments/euppbench_reforecasts/run_training.py`](experiments/euppbench_reforecasts/run_training.py) for COSMO-DE and EUPPBench data, respectively. 
The scripts are mostly identical, but load different datasets internally. 
For examples, we focus on the EUPPBench case study. Training COSMO-DE models works equivalently.  

#### Examples (EUPPBench)

- ED-DRN with truncated logistic prediction parameterization and mean-based merger:

        python experiments/euppbench_reforecasts/run_training.py
            --output:path /path/to/output/dir
            --data:flt 6                    # intended lead time
            --training:batch-size 64
            --training:optimizer:lr 1.e-4
            --training:ensemble-size 10     # number of models for ensemble averaging
            --loss:type logistic 
            --model:merger:type mean
            --model:encoder:type mlp
            --model:encoder:num-layers 3
            --model:encoder:num-channels 64
            --model:decoder:type mlp
            --model:decoder:num-layers 3
            --model:decoder:num-channels 64,
            --model:bottleneck 64
            # for GPU training: --training:use-gpu

- ST-BQN with attention-based merger:
        
        python experiments/euppbench_reforecasts/run_training.py
            --output:path /path/to/output/dir
            --data:flt 6                    # intended lead time
            --training:batch-size 64
            --training:optimizer:lr 1.e-4
            --training:ensemble-size 10     # number of models for ensemble averaging
            --loss:type bqn
            --loss:kwargs "{'integration_scheme': 'uniform', 'num_quantiles': 99}"
            --model:merger:type weighted-mean
            --model:merger:kwargs "{'num_heads': 8}"
            --model:encoder:type attention
            --model:encoder:num-layers 8    # corresponds to two attention blocks
            --model:encoder:num-channels 64
            --model:decoder:type mlp
            --model:decoder:num-layers 3
            --model:decoder:num-channels 64,
            --model:bottleneck 64
            # for GPU training: --training:use-gpu

#### Exporting predictions

To export predictions for validation and test data in a unified format, run 

    python experiments/euppbench_reforecasts/predict.py --path /path/to/training/output --valid --test

### Training baselines

Scripts for training the baseline models are provided in [`experiments/baselines`](experiments/baselines). Note that these scripts require input data in `.csv` format.

#### Examples (EUPPBench)

- DRN with truncated logistic prediction parameterization:
    
        python experiments/baselines/run_training_drn_eupp.py
            --data-train /path/to/data/train.csv
            --data-test /path/to/data/test.csv
            --output:path /path/to/output/dir
            --model:posterior logistic

- BQN:
    
        python experiments/baselines/run_training_bqn_eupp.py
            --data-train /path/to/data/train.csv
            --data-test /path/to/data/test.csv
            --output:path /path/to/output/dir
            --model:p-degree 12
            # for ensemble input: --model:use-ensemble

- EMOS (custom Pytorch implementation using LBFGS optimizer):
    
        python experiments/baselines/run_training_emos_eupp.py
            --data-train /path/to/data/train.csv
            --data-test /path/to/data/test.csv
            --output:path /path/to/output/dir
            --model:posterior logistic

#### Exporting predictions

The training scripts for the baseline methods export predictions for validation and test data automatically. 
To copy the data to the required location for subsequent evaluation, run

    python evaluation/copy_predictions.py --path /path/to/training/output

### Predicting for forecast data

The folder [`experiments/euppbench_forecasts`](experiments/euppbench_forecasts) contains scripts for exporting predictions for the EUPPBench forecast dataset. 
Separate scripts are used for ensemble-based models ([`predict.py`](experiments/euppbench_forecasts/predict.py)), 
DRN and BQN ([`predict_minimal.py`](experiments/euppbench_forecasts/predict_minimal.py))
and EMOS([`predict_emos.py`](experiments/euppbench_forecasts/predict_emos.py))

### Computing evaluation metrics

The script [`evaluation/eval.py`](evaluation/eval.py) provides functionality for computing forecast evaluation metrics. 
A distinction between ensemble-based and baseline models is not required, but the prediction format must be specified.

#### Examples (EUPPBench)

- Truncated logistic predictions for reforecast data:

        python evaluation/eval.py exp logistic 
            --path /path/to/training/output
            --valid --test
            --num-members 11  # size of reference ensemble: 11 for reforecasts

- BQN predictions for forecast data:

        python evaluation/eval.py exp bqn 
            --path /path/to/training/output 
            --forecasts
            --num-members 51  # size of reference ensemble: 51 for forecasts

### Feature Permutation Analysis

The functionality to reproduce feature permutation results is contained in [`evaluation/feature_permutation`](evaluation/feature_permutation).
The scripts [`predict_perturbed.py`](evaluation/feature_permutation/euppbench/predict_perturbed.py), 
and [`predict_perturbed_minimal.py`](evaluation/feature_permutation/euppbench/predict_perturbed_minimal.py)
can be used to compute perturbed predictions for ensemble-based models and summary-based models (DRN and BQN), respectively. 

Permutation importance scores can be computed with [`eval_scalar_predictors.py`](evaluation/feature_permutation/euppbench) for summary-based models (DRN and BQN) 
and [`eval_single_features.py`](evaluation/feature_permutation/euppbench/eval_single_features.py) for ensemble-based architectures.

The implementation of the binned shuffling perturbation is located in [`perturbations.py`](evaluation/feature_permutation/perturbations.py).

## Known issues

In subsequent work, Sebastian Lerch and colleagues pointed out that the data preparation code in the EUPPBench case study deviates from the procedures described in the paper.

- Table A2 in the paper states that parameters q, r, and cin are used as inputs to the models. However, the parameters are not loaded in the code. The accurate list of dynamic input parameters can be found in the parameter `DYNAMIC_PREDICTORS` in the file [`data/euppbench/reforecasts.py`](data/euppbench/reforecasts.py).

- The training-validation-test splitting of the available data is not based on forecast valid times, as stated in the paper, but based on the yearly shift of the reforecast dates backwards in time. The code splits the `year` axis in the source data array into non-overlapping sections of length 12 (training), 4 (validation), and 4 (test), which is not aligned with years in terms of forecast valid times. The procedure is implemented in in the file [`data/euppbench/reforecasts.py`](data/euppbench/reforecasts.py). For details about the timestamp layout of the EUPPBench dataset, see the [source publication](https://doi.org/10.5194/essd-15-2635-2023), in which the dataset was presented.

The code has been executed identically for all experiments related to the EUPPBench dataset, equally for both reforecasts and forecasts. Accordingly all comparisons remain valid and consistent. Effects on the shown performance scores are expected to be minor. Still, authors interested in reproducing our results should consider the differences. The experiments concerning the wind gust study are unaffected. 
