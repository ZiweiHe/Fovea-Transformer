# !/usr/bin/env bash

ROOT_PATH=/your/root/path/
MAX_INPUT_LEN=16384
MAX_TARGET_LEN=256
LR=1e-3
OUTPUT_DIR=${ROOT_PATH}/output/$1
BATCH_SIZE=2
UPDATE_FREQ=16
SEED=666

deepspeed --include="localhost:0,1,2,3" --master_port 26000 scripts/run.py \
    --deepspeed ds_config.json \
    --model_name_or_path google/long-t5-tglobal-base \
    --cache_dir ${ROOT_PATH}/cached \
    --run_name $1 \
    --seed $SEED \
    --data_seed $SEED \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name $2 \
    --dataset_config_name $3 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir True \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE  \
    --gradient_accumulation_steps $UPDATE_FREQ \
    --gradient_checkpointing \
    --learning_rate $LR \
    --optim adafactor \
    --warmup_ratio 0.05 \
    --lr_scheduler_type polynomial \
    --group_by_length \
    --logging_strategy steps \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --greater_is_better True \
    --num_train_epochs 10 \
    --predict_with_generate \
    --generation_num_beams 3 \
    --max_source_length $MAX_INPUT_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --ddp_find_unused_parameters False \
    --report_to wandb \
    --bf16 \
    --wing_size  1 \
    --block_len 128 \
    --stop_level 20