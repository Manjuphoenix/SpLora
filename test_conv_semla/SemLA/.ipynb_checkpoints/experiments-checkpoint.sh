#!/bin/bash

semla_config='./semla_config.yaml'
source_domains='./source_domains.yaml'
target_domains='./target_domains.yaml'
# results_folder='./results/only-aerial/conv/labels-corrected/oracle/indraeye'
# results_folder='./results/only-aerial/conv/labels-corrected/uniform/cartr'
results_folder='./results/only-ground/conv/analysis/semla/indraeye'

# Uncomment the experiments you want to run

# python experiments.py --experiment zeroshot \
#     --source_domains $source_domains \
#     --target_domains $target_domains \
#     --output_dir $results_folder

# python experiments.py --experiment oracle \
#     --source_domains $source_domains \
#     --target_domains $target_domains \
#     --output_dir $results_folder

# python experiments.py --experiment uniform \
#     --source_domains $source_domains \
#     --target_domains $target_domains \
#     --remove_target_adapter \
#     --output_dir $results_folder

python experiments.py --experiment semla \
    --source_domains $source_domains \
    --target_domains $target_domains \
    --semla_config $semla_config \
    --remove_target_adapter \
    --output_dir $results_folder