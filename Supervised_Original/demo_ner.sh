CUDA_VISIBLE_DEVICES=1 python3 train_demo.py --logdir checkpoints/demo05 \
					--finetuning \
					--testing \
					--batch_size 8 \
					--lr 5e-5 \
					--n_epochs 100 \
					--trainset /home/s2071932/codes/LatticeLSTM_py35/data/demo.train.char \
					--validset /home/s2071932/codes/LatticeLSTM_py35/data/demo.dev.char \
					--testset /home/s2071932/codes/LatticeLSTM_py35/data/demo.test.char \