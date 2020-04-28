check_gpu=`python3 check_free_gpu.py`
output=${check_gpu}

CUDA_VISIBLE_DEVICES=${output} python3 ../train_demo.py --logdir checkpoints/trail00 \
					--finetuning \
					--testing \
					--top_rnns \
					--batch_size 8 \
					--lr 5e-5 \
					--weight_decay 0.0 \
					--rnn_layers 2 \
					--hidden_size 128 \
					--n_epochs 5 \
					--trainset /home/s2071932/codes/LatticeLSTM_py35/data/demo.train.char \
					--validset /home/s2071932/codes/LatticeLSTM_py35/data/demo.dev.char \
					--testset /home/s2071932/codes/LatticeLSTM_py35/data/demo.test.char \
