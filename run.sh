#! /bin/bash

python main.py -c "configs/ensemble_abs.json"
mv saved experiments/ensemble_abs

python main.py -c "configs/ensemble_res.json"
mv saved experiments/ensemble_res

python main.py -c "configs/mlp_abs_norm.json"
mv saved experiments/mlp_abs_norm

python main.py -c "configs/mlp_abs_standard.json"
mv saved experiments/mlp_abs_standard

python main.py -c "configs/mlp_res_norm.json"
mv saved experiments/mlp_res_norm

python main.py -c "configs/mlp_res_standard.json"
mv saved experiments/mlp_res_standard

python main.py -c "configs/mc_dropout.json"
mv saved experiments/mc_dropout
