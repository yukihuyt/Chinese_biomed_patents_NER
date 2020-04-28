CUDA_VISIBLE_DEVICES=6 python3 -u train.py --logdir /data/s2071932/BertNER/cbp04 \
					--finetuning \
					--testing \
					--batch_size 8 \
					--lr 5e-5 \
					--n_epochs 40 \
					--trainset /local/s2071932/general_data/cbp_gold_small_v2/4/no_long_cbp_gold_v2_train.bio \
					--validset /local/s2071932/general_data/cbp_gold_small_v2/4/no_long_cbp_gold_v2_dev.bio \
					--testset /local/s2071932/general_data/cbp_gold_small_v2/4/no_long_cbp_gold_v2_test.bio \