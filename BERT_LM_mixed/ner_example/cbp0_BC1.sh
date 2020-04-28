export OUTPUT_DIR=/data/s2071932/Transformers/resources/output/cbp_ner00_BC1/
export BATCH_SIZE=8
export NUM_EPOCHS=40
export SAVE_STEPS=750
export SEED=1
export MAX_LENGTH=512
export BERT_MODEL=/data/s2071932/Transformers/resources/output/first_lmft_BC/
export CUDA_VISIBLE_DEVICES=0

python3 -u run_ner.py --data_dir /local/s2071932/general_data/cbp_gold_small_v2/0/ \
--train_filename no_long_cbp_gold_v2_train.bio \
--test_filename no_long_cbp_gold_v2_test.bio \
--dev_filename no_long_cbp_gold_v2_dev.bio \
--model_type bert \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--overwrite_output_dir \
--do_train \
--do_eval \
--do_predict

