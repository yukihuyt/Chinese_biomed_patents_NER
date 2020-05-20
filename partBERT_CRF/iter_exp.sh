if [ ! -d "./output" ]; then
	mkdir "./output"
fi

oldifs="$IFS"
IFS=$'\n'

for i in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python3 -u run_ner.py --data_dir ../data/cbp_gold/$i/ \
    --train_filename no_long_cbp_gold_v2_train.bio \
    --test_filename no_long_cbp_gold_v2_test.bio \
    --dev_filename no_long_cbp_gold_v2_dev.bio \
    --output_dir ./output/$i/ \
    --num_train_epochs 40 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_test_batch_size 4 \
    --learning_rate 0.1

done

IFS="$oldifs"