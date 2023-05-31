# CGM Preprocessing Pipeline

This repository contains the code for the CGM preprocessing pipeline, a crucial step for preparing the raw data from the Child Growth Monitor (CGM) project for machine learning model training. The pipeline downloads the raw data, preprocesses it, and stores it in a suitable format for further analysis.

## Overview

The CGM preprocessing pipeline performs the following steps:

1. Fetching raw scans from the database.
2. Calculating z-scores and diagnosis based on World Health Organization (WHO) growth standards.
3. Downloading RGB and depthmap data from Azure Blob Storage.
4. Preprocessing the RGB and depthmap data, including matching, rotation, and fusion (if required).
5. Saving the preprocessed data in pickle format for easy consumption by machine learning models.

## Pipeline

This Azure pipeline  processes data on a Databricks Spark cluster. Here is a step by step description of what this pipeline does:

### Trigger 
The pipeline doesn't have a specified trigger, which means it's manually started.

### Pool 
The pipeline is set to run on a hosted agent with an Ubuntu image. The job is configured to timeout if it runs for longer than 360 minutes.

### Variables

The following variables are defined to be used in the pipeline:

- `pythonVersion`: This variable defines the version of Python to be used in the pipeline.
- `DATA_CATEGORY`: This variable can either be 'Train' or 'Test'. Depending on its value, the corresponding type of data will be pulled from the database for processing.
- `DATASET_TYPE`: This variable defines the type of images that are included in the output pickle files. Possible values are: 'rgb' for RGB images, 'depthmap' for depth images, and 'rgbd' for merged RGB and depth images.
- `JOB_PATH`: This variable specifies the location where the output pickle files will be stored.
- `NUM_ARTIFACTS`: This variable is used to limit the number of artifacts taken from the database for processing.
- `UPLOAD_TO_BLOB_STORAGE`: This variable can be either TRUE or FALSE. If TRUE, the final output pickle files will be uploaded to blob storage. If FALSE, this step will be skipped.
- `WORKFLOW_ID_POSE`: This GUID is used in the SQL queries for retrieving pose data from the database.


### Jobs
- `PreProc:` A job named "PreProc" is defined, which has its own timeout of 360 minutes.
- `UsePythonVersion:` The job starts with setting up the Python version specified by the `pythonVersion` variable.
- `Install Databricks CLI:` Databricks CLI is installed which is a command line interface for interacting with Databricks.
- `Configure Databricks CLI:` The Databricks CLI is then configured with the host and token which are required for authentication.
- `Upload Repository to DBFS:` The local repository files are copied to Databricks File System (DBFS) using the Databricks CLI.
- `Create Databricks Job:` A Databricks job is created with a new cluster configuration, library dependencies, and a Python file to execute. The job configuration also includes various parameters for the script to run, like path to log, dataset type, database host/user/password, blob connection string, number of artifacts, whether to upload to blob storage, blob upload connection string, workflow id for pose etc.
- `Execute Databricks Job:` The job is then run. The pipeline continuously monitors the lifecycle state of the job, providing updates on whether the job is still running or has reached a terminal state (TERMINATED, FAILED, INTERNAL_ERROR). If the job is still running, it logs the last entries and if the job has finished, it logs the final state. If the job fails, it prints an error message and exits with code 1.

Overall, this Azure pipeline is designed to perform a data preprocessing operation using a Databricks Spark cluster, monitor the operation, and handle any potential errors in the process.

### Process

TBD