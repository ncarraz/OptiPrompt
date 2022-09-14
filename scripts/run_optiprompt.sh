#!/bin/bash

python code/run_optiprompt.py \
        --model_name $1 \
        --output_predictions \
	--do_train \
	--output_dir output  
	--learning_rate 3e-2
#python code/accumulate_results.py ${OUTPUTS_DIR}
