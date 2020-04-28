echo "Parameter tuning experiments on Bert-original with no rnn top layer."

check_gpu=`python3 check_free_gpu.py`
output=${check_gpu}

CUDA_VISIBLE_DEVICES=${output} python3 ../train.py --logdir ../checkpoints/demo_ft00 \
					--finetuning \
					--testing \
					--batch_size 16 \
					--lr 1e-5 \
					--weight_decay 1e-4 \
					--n_epochs 10 \
					--trainset /home/s2071932/codes/LatticeLSTM_py35/data/demo.train.char \
					--validset /home/s2071932/codes/LatticeLSTM_py35/data/demo.dev.char \
					--testset /home/s2071932/codes/LatticeLSTM_py35/data/demo.test.char \