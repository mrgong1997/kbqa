#!/usr/bin/env bash
DATA_DIR="F:\PycharmProject\kbqa\CBLUE-main\CaseDatasets"

TASK_NAME="qic"
MODEL_TYPE="bert"
MODEL_DIR="F:\PycharmProject\kbqa\CBLUE-main\Huggingface"
MODEL_NAME="chinese-bert-wwm-ext"
OUTPUT_DIR="F:\PycharmProject\kbqa\CaseDataSet\outputmodel"
RESULT_OUTPUT_DIR="F:\PycharmProject\kbqa\CaseDataSet\outputresult"

MAX_LENGTH=64

echo "Start running"

if [ $# == 0 ]; then
    python baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_dir=${MODEL_DIR} \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_train \
        --max_length=${MAX_LENGTH} \
        --train_batch_size=64 \
        --eval_batch_size=64 \
        --learning_rate=2e-5 \
        --epochs=3 \
        --warmup_proportion=0.1 \
        --earlystop_patience=3 \
        --logging_steps=2000 \
        --save_steps=2000 \
        --seed=2021
elif [ $1 == "predict" ]; then
    python baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_name=${MODEL_NAME} \
        --model_dir=${MODEL_DIR} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_predict \
        --max_length=${MAX_LENGTH} \
        --eval_batch_size=16 \
        --seed=2021
fi