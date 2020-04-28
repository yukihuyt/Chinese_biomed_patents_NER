echo "Parameter tuning experiments on Bert-original with 2-layer bi-LSTM added on top."

for var in 0.1 1e-3 1e-4 1e-5
do
check_gpu=`python3 check_free_gpu.py`
output=${check_gpu}
CUDA_VISIBLE_DEVICES=${output} python3 ../train.py --logdir ../checkpoints/cbp_small_demo00 \
					--finetuning \
					--testing \
					--batch_size 16 \
					--lr ${var} \
					--weight_decay 1e-4 \
					--rnn_layers 1 \
					--hidden_size 128 \
					--n_epochs 3 \
					--trainset /home/s2071932/data/cbp_small_demo_random/train_char.txt \
					--validset /home/s2071932/data/cbp_small_demo_random/valid_char.txt \
					--testset /home/s2071932/data/cbp_small_demo_random/test_char.txt ; done
