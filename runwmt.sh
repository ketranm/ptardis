#!/bin/bash
#SBATCH --time=99:00:00
#SBATCH -N 1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1
#SBATCH -o log/wmt20k.log
#SBATCH -e log/err
#SBATCH -J buggy2k
module load cuda80/toolkit/8.0.44
module load cuDNN/cuda80/5_5.1.5-1
python -u train.py --datasets wmt20k/all_de-en.de.tok.bpe wmt20k/all_de-en.en.tok.bpe \
    --valid_datasets wmt20k/newstest2014.de.tok.bpe wmt20k/newstest2014.en.tok.bpe \
    --dicts wmt20k/all_de-en.de.tok.bpe.pkl wmt20k/all_de-en.en.tok.bpe.pkl \
    --emb_size 1024 --rnn_size 1024 --gpus 0 --lr 0.002 --layers 3 --valid_freq 5000 --batch_size 64 \
    --ref wmt20k/newstest2014.en.tok --report_freq 50 --max_generator_batches 32 --saveto tardis20k.pt
