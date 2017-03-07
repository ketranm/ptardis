#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH -N 1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1
#SBATCH -o log/wmt.log
#SBATCH -e log/err
#SBATCH -J buggywmt
module load cuda80/toolkit/8.0.44
module load cuDNN/cuda80/5_5.1.5-1
python -u train.py --datasets wmt/all_de-en.tok.bpe.de wmt/all_de-en.tok.bpe.en \
    --valid_datasets wmt/newstest2013.tok.bpe.de wmt/newstest2013.tok.bpe.en \
    --dicts wmt/all_de-en.tok.bpe.de.pkl wmt/all_de-en.tok.bpe.en.pkl \
    --emb_size 1024 --rnn_size 1024 --gpus 0 --lr 0.002 --layers 3 --valid_freq 5000 --batch_size 64 \
    --ref wmt/newstest2013.en.tok --report_freq 50 --max_generator_batches 32
