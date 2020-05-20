if [ ! -d "./output" ]; then
    mkdir "./output"
fi

export OUT_DIR=./output/
export SAVE_STEPS=50000
export BATCH_SIZE=8
export LR=5e-5
export N_EPOCHS=30
export TRAIN_FILE=../data/path_to_unlabeled_train
export TEST_FILE=../data/path_to_unlabeled_test

CUDA_VISIBLE_DEVICES=0 python3 -u run_lm_finetuning.py \
    --output_dir=$OUT_DIR \
    --model_type=bert \
    --save_steps=$SAVE_STEPS \
    --per_gpu_train_batch_size=$BATCH_SIZE \
    --per_gpu_eval_batch_size=$BATCH_SIZE \
    --learning_rate=$LR \
    --num_train_epochs=$N_EPOCHS \
    --model_name_or_path=bert-base-chinese \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm