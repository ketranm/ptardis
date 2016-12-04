"""UvA Neural Machine Translation
author: Ke Tran <m.k.tran@uva.nl>
"""

import torch
import torch.nn as nn

import argparse
import data
import model


parser = argparse.ArgumentParser(description="Neural Machine Translation")
# data
parser.add_argument('-data', type=str, default='./wmt', help="data location")
parser.add_argument('-source', type=str, default='de',
                    help="extension of the source file")
parser.add_argument('-target', type=str, default='en',
                    help="extension of target file")
parser.add_argument('-batch_size', type=int, default=64, help="batch size")

# network
parser.add_argument('-embsize', type=int, default=1024, help="embedding size")
parser.add_argument('-hidsize', type=int, default=1024, help="hidden size")
parser.add_argument('-nlayers', type=int, default=3, help="number of layers")
# optimization
parser.add_argument('-dropout', type=float, default=0.4, help="dropout rate")
parser.add_argument('-clip', type=float, default=5, help="Gradient clipping")
parser.add_argument('-maxepoch', type=int, default=13, help="max number epochs")
# cuda
parser.add_argument('-cuda', action='store_true', help="use CUDA")

args = parser.parse_args()


#if torch.cuda.is_avalable() and not args.cuda:
#    print("WARNING: you have a CUDA device, you should probably run with -cuda")

bitext = data.BitextIterator(args.data, 'de', 'en', 64)

source_size = bitext.source_dict.size()
target_size = bitext.target_dict.size()
encdec = model.NMT(source_size, target_size, args.embsize,
                  args.hidsize, args.nlayers, args.dropout)

# loop over epoches
prev_loss = None
for epoch in range(1, args.maxepoch + 1):
    total_loss = 0
