if [ ! -d "./output" ]; then
	mkdir "./output"
fi

export OUTPUT_DIR=./output/ner/
export BATCH_SIZE=4
export NUM_EPOCHS=40
export START_LR=0.1
export CUDA_VISIBLE_DEVICES=0

python3 -u run_ner.py --data_dir ../data/cbp_gold/0/ \
--train_filename no_long_cbp_gold_v2_train.bio \
--test_filename no_long_cbp_gold_v2_test.bio \
--dev_filename no_long_cbp_gold_v2_dev.bio \
--output_dir $OUTPUT_DIR \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_test_batch_size $BATCH_SIZE \
--learning_rate $START_LR \