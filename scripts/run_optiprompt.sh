#!/bin/bash

python code/run_optiprompt.py \
        --model_name $1 \
        --output_predictions \
        --do_train  \
        --init_manual_template 
#python code/accumulate_results.py ${OUTPUTS_DIR}