export OUTPUT_DIR=/data/s2071932/Transformers/resources/output/pred_BC/
export BATCH_SIZE=256
export NUM_EPOCHS=1
export SAVE_STEPS=2000
export SEED=1
export MAX_LENGTH=512
export BERT_MODEL=/data/s2071932/Transformers/resources/output/final_train_ner00/
export CUDA_VISIBLE_DEVICES=8

python3 -u make_predictions.py --pred_input_dir /local/s2071932/general_data/first_clean_BC/ \
--model_type bert \
--model_name_or_path $BERT_MODEL \
--pred_output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \