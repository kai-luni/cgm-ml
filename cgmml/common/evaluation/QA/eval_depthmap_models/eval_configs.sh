#!/usr/bin/env bash

set -euox pipefail

python eval_main.py

# Combine results
python combine_results.py --model_measurement height
