export OUTPUT_DIR=/local/s2071932/FlairChinese/output/cbp_ner02/
export BATCH_SIZE=12
export NUM_EPOCHS=30
export START_LR=0.1
export CUDA_VISIBLE_DEVICES=11

python3 -u run_ner.py --data_dir /local/s2071932/general_data/cbp_gold_small_v2/2/ \
--train_filename no_long_cbp_gold_v2_train.bio \
--test_filename no_long_cbp_gold_v2_test.bio \
--dev_filename no_long_cbp_gold_v2_dev.bio \
--output_dir $OUTPUT_DIR \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--learning_rate $START_LR \