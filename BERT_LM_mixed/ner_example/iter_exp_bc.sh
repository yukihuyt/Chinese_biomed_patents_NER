if [ ! -d "./output" ]; then
    mkdir "./output"
fi

if [ ! -d "./output/ner_bc" ]; then
    mkdir "/output/ner_bc"
fi

oldifs="$IFS"
IFS=$'\n'

for i in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python3 -u run_ner.py --data_dir ../../data/cbp_gold/$i/ \
    --train_filename no_long_cbp_gold_v2_train.bio \
    --test_filename no_long_cbp_gold_v2_test.bio \
    --dev_filename no_long_cbp_gold_v2_dev.bio \
    --model_type bert \
    --model_name_or_path ../../models/partBC_30epochs/ \
    --output_dir ./output/ner_bc/$i/ \
    --max_seq_length  512 \
    --num_train_epochs 40 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --save_steps 15000 \
    --seed 1 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict

done

IFS="$oldifs"