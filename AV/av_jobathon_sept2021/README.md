# JOB-A-THON - September 2021

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