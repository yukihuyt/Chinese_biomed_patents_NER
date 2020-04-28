export OUTPUT_DIR=/home/s2071932/codes/Transformers/resources/output/demozh_ner_example/
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1
export MAX_LENGTH=512
export BERT_MODEL=bert-base-chinese
export CUDA_VISIBLE_DEVICES=9

python3 -u run_ner.py --data_dir /home/s2071932/data/zh_ner_small_demo/ \
--train_filename no_long_demo.train.char \
--test_filename no_long_demo.test.char \
--dev_filename no_long_demo.dev.char \
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