CUDA_VISIBLE_DEVICES=0 python3 -u train.py --logdir /data/s2071932/BertNER/cbp00 \
					--finetuning \
					--testing \
					--batch_size 8 \
					--lr 5e-5 \
					--n_epochs 40 \
					--trainset /data/s2071932/General_data/cbp_gold_small_v2/0/no_long_cbp_gold_v2_train.bio \
					--validset /data/s2071932/General_data/cbp_gold_small_v2/0/no_long_cbp_gold_v2_dev.bio \
					--testset /data/s2071932/General_data/cbp_gold_small_v2/0/no_long_cbp_gold_v2_test.bio \