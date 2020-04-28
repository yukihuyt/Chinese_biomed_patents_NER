export OUT_DIR=/data/s2071932/Transformers/resources/output/lmft_HG_load_19/
export SAVE_STEPS=50000
export BATCH_SIZE=8
export LR=5e-5
export N_EPOCHS=19
export LOAD_PATH=/local/s2071932/Transformers/resources/output/lmft_HG_40epoch/checkpoint-100000/
export TRAIN_FILE=/local/s2071932/Transformers/resources/cbp_ft_train_HG.txt
export TEST_FILE=/local/s2071932/Transformers/resources/cbp_ft_test_HG.txt

CUDA_VISIBLE_DEVICES=10 python3 -u run_lm_finetuning.py \
    --output_dir=$OUT_DIR \
    --model_type=bert \
    --save_steps=$SAVE_STEPS \
    --per_gpu_train_batch_size=$BATCH_SIZE \
    --per_gpu_eval_batch_size=$BATCH_SIZE \
    --learning_rate=$LR \
    --num_train_epochs=$N_EPOCHS \
    --model_name_or_path=$LOAD_PATH \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm