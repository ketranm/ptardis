#!/bin/bash
# This script preprocesses bitext with Byte Pair Encoding for NMT.
# Executes the following steps:
#     1. Tokenize source and target side of bitext
#     2. Learn BPE-codes for both source and target side
#     3. Encode source and target side using the codes learned
#     4. Shuffle bitext for SGD
#     5. Build source and target dictionaries

if [ "$#" -ne 5 ]; then
    echo ""
    echo "Usage: $0 src trg path_to_data path_to_subword"
    echo ""
    exit 1
fi

export PYTHON=python2


if [ -z $PYTHON ]; then
    echo "Please set PYTHON to a Python interpreter"
    exit 1
fi

echo "Using $PYTHON"

# number of merge ops (codes) for bpe
SRC_CODE_SIZE=20000
TRG_CODE_SIZE=20000

# source language (example: fr)
S=$1
# target language (example: en)
T=$2

# path to dl4mt/data
P1=$3

# path to subword NMT scripts (can be downloaded from https://github.com/rsennrich/subword-nmt)
P2=$4

# path to text files
P3=$5

# merge all parallel corpora
./merge.sh $1 $2 $5

# tokenize training and validation data
perl $P1/tokenizer.perl -threads 5 -l $S < ${P3}/all_${S}-${T}.${S} > ${P3}/all_${S}-${T}.${S}.tok
perl $P1/tokenizer.perl -threads 5 -l $T < ${P3}/all_${S}-${T}.${T} > ${P3}/all_${S}-${T}.${T}.tok
perl $P1/tokenizer.perl -threads 5 -l $S < ${P3}/ted2013/ted.tst2013.${S} > ${P3}/ted2013.${S}.tok
perl $P1/tokenizer.perl -threads 5 -l $T < ${P3}/ted2014/ted.tst2014.${T} > ${P3}/ted2014.${T}.tok

# BPE
if [ ! -f "${P3}/${S}.bpe" ]; then
    $PYTHON $P2/learn_bpe.py -s 20000 < ${P3}/all_${S}-${T}.${S}.tok >${P3}/${S}.bpe
fi
if [ ! -f "${P3}/${T}.bpe" ]; then
    $PYTHON $P2/learn_bpe.py -s 20000 < ${P3}/all_${S}-${T}.${T}.tok > ${P3}/${T}.bpe
fi

# utility function to encode a file with bpe
encode () {
    if [ ! -f "$3" ]; then
        $PYTHON $P2/apply_bpe.py -c $1 < $2 > $3
    else
        echo "$3 exists, pass"
    fi
}

# apply bpe to training data
encode ${P3}/${S}.bpe ${P3}/all_${S}-${T}.${S}.tok ${P3}/all_${S}-${T}.${S}.tok.bpe
encode ${P3}/${T}.bpe ${P3}/all_${S}-${T}.${T}.tok ${P3}/all_${S}-${T}.${T}.tok.bpe
encode ${P3}/${S}.bpe ${P3}/ted2013.${S}.tok ${P3}/ted2013.${S}.tok.bpe
encode ${P3}/${T}.bpe ${P3}/ted2014.${T}.tok ${P3}/ted2014.${T}.tok.bpe

# shuffle
$PYTHON $P1/shuffle.py ${P3}/all_${S}-${T}.${S}.tok.bpe ${P3}/all_${S}-${T}.${T}.tok.bpe

# build dictionary
$PYTHON $P1/build_dictionary.py ${P3}/all_${S}-${T}.${S}.tok.bpe
$PYTHON $P1/build_dictionary.py ${P3}/all_${S}-${T}.${T}.tok.bpe
