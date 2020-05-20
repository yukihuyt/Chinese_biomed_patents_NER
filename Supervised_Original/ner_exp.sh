if [ ! -d "./output" ]; then
	mkdir "./output"
fi

CUDA_VISIBLE_DEVICES=0 python3 -u train.py --logdir ./output/ner/ \
					--finetuning \
					--testing \
					--batch_size 8 \
					--lr 5e-5 \
					--n_epochs 40 \
					--trainset ../data/cbp_gold/$i/no_long_cbp_gold_v2_train.bio \
					--validset ../data/cbp_gold/$i/no_long_cbp_gold_v2_dev.bio \
					--testset ../data/cbp_gold/$i/no_long_cbp_gold_v2_test.bio \