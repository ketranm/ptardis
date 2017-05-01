# ptardis
pytorch implementation of Tardis

# Data Preparation
Prepare your own proprecessing script. Example of such a script is
`data/preprocess_wmt.sh`. The proprecssing step is borrowed from [dl4mt](https://github.com/nyu-dl/dl4mt-tutorial). We own a thank to Orhan Firat, who constantly brags "_we have the best data processing pipe-line in the world_". Taking a leap of faith, we simply steal his data processing script.

```
sh data/preprocess_wmt.sh de en . subword-nmt ../wmt
```

# Building your NMT system

Example
```
python -u train.py --datasets wmt20k/all_de-en.de.tok.bpe wmt20k/all_de-en.en.tok.bpe \
    --valid_datasets wmt20k/newstest2014.de.tok.bpe wmt20k/newstest2014.en.tok.bpe \
    --dicts wmt20k/all_de-en.de.tok.bpe.pkl wmt20k/all_de-en.en.tok.bpe.pkl \
    --emb_size 1024 --rnn_size 1024 --gpus 0 --lr 0.002 --layers 3 --valid_freq 5000 --batch_size 64 \
    --ref wmt20k/newstest2014.en.tok --report_freq 50 --max_generator_batches 32 --saveto tardis20k.pt
```

Note that, Ke Tran is too lazy to write up the documentation. You probably have to figure it out by yourself. If you have any trouble, pass by his office and offer Ke a cup of expresso before asking any engineering questions. Ke is really angry in general.

# Credit (for whom we stole)
 - Orhan Firat and dl4mt
 - OpenMT
 - Ke Tran (_doing with without being paid_)
 - Coffee and Cigarettes
