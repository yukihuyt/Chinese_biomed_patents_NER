export OUT_DIR=/data/s2071932/Transformers/resources/output/lmft_BC_load_39/
export SAVE_STEPS=50000
export BATCH_SIZE=8
export LR=5e-5
export N_EPOCHS=39
export LOAD_PATH=/data/s2071932/Transformers/resources/output/first_lmft_BC/
export TRAIN_FILE=/data/s2071932/Transformers/resources/cbp_ft_demo_train.txt
export TEST_FILE=/data/s2071932/Transformers/resources/cbp_ft_demo_test_HG.txt

CUDA_VISIBLE_DEVICES=11 python3 -u run_lm_finetuning.py \
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