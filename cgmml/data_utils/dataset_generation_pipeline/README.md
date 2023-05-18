# CGM Preprocessing Pipeline

This repository contains the code for the CGM preprocessing pipeline, a crucial step for preparing the raw data from the Child Growth Monitor (CGM) project for machine learning model training. The pipeline downloads the raw data, preprocesses it, and stores it in a suitable format for further analysis.

## Overview

The CGM preprocessing pipeline performs the following steps:

1. Fetching raw scans from the database.
2. Calculating z-scores and diagnosis based on World Health Organization (WHO) growth standards.
3. Downloading RGB and depthmap data from Azure Blob Storage.
4. Preprocessing the RGB and depthmap data, including matching, rotation, and fusion (if required).
5. Saving the preprocessed data in pickle format for easy consumption by machine learning models.