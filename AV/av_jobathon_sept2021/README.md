# JOB-A-THON - September 2021

## Overview
* [Solution](https://github.com/dc-aichara/competitions/tree/master/AV/av_jobathon_sept2021) has an end to end pipeline for quick experimentation with feature generation, model training, and model serving. Pipeline is a configuration based pipeline. Configuration helps to control features and training parameters with more ease. If you want to start using end to end pipeline for competitions, refer [this](https://github.com/dc-aichara/competition_template) ready to consume template. 
* I have used streaming prediction approach with LightGBM for time series predictions.
* I only build a generalized model. But score could be improved by adding more lag features and building individual store models. 

## Development Env
 1. Create conda environment `jobathon` with `environment.yml` file.
    ```shell
    conda env create -f environment.yml
    ```
 2. Activate `jobathon` conda env
    ```shell
    conda activate jobathon
    ```
    
## Pipeline

1. Add CSV files to data/raw
2. Change configuration parameters in `config.yml` if you want. 
3. Run whole pipeline
   ```shell
    sh start_pipeline.sh
   ```