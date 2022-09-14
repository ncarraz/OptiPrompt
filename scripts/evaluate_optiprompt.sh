#!/bin/bash
SOURCE_MODEL=$1
TARGET_MODEL=$2
python code/run_eval_prompts.py \
        --model_name $TARGET_MODEL \
        --source_dir output/$SOURCE_MODEL \
        --target_dir results/$SOURCE_MODEL \
#        --init_manual_template	
#python code/accumulate_results.py ${OUTPUTS_DIR}
