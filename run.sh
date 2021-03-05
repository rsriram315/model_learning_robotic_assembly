#! /bin/bash

# VANILLA="configs/mlp.json"
# python train.py -c $VANILLA

ENSEMBLE="configs/ensemble.json"
python main.py -c $ENSEMBLE