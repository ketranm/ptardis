"""UvA Neural Machine Translation
author: Ke Tran <m.k.tran@uva.nl>
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
import data
import model
import math
import time

parser = argparse.ArgumentParser(description="Neural Machine Translation")
# data
parser.add_argument('-data', type=str, default='./iwslt', help="data location")
parser.add_argument('-source', type=str, default='de',
                    help="extension of the source file")
parser.add_argument('-target', type=str, default='en',
                    help="extension of target file")
parser.add_argument('-batch_size', type=int, default=32, help="batch size")

# network
parser.add_argument('-embsize', type=int, default=1024, help="embedding size")
parser.add_argument('-hidsize', type=int, default=1024, help="hidden size")
parser.add_argument('-nlayers', type=int, default=3, help="number of layers")
# optimization
parser.add_argument('-dropout', type=float, default=0.4, help="dropout rate")
parser.add_argument('-clip', type=float, default=5, help="Gradient clipping")
parser.add_argument('-lr', type=float, default=1, help="Learning rate")
parser.add_argument('-maxepoch', type=int, default=13, help="max number epochs")
# cuda
parser.add_argument('-cuda', action='store_true', help="use CUDA")

args = parser.parse_args()


if torch.cuda.is_available() and not args.cuda:
    print("WARNING: you have a CUDA device, you should probably run with -cuda")

bitext = data.BitextIterator(args.data, args.source, args.target, args.batch_size)

source_size = bitext.source_dict.size()
target_size = bitext.target_dict.size()
encdec = model.NMT(source_size, target_size, args.embsize,
                  args.hidsize, args.nlayers, args.dropout)
weight = torch.ones(target_size)
weight[0] = 0
criterion = nn.CrossEntropyLoss(weight)
if args.cuda:
    encdec.cuda()
    criterion.cuda()

def prepro(sample):
    sample =  [sample[0].cuda(), sample[1].cuda()]
    input = (Variable(sample[0], requires_grad=False),
            Variable(sample[1][0:-2], requires_grad=False))
    target = Variable(sample[1][1:-1].contiguous().view(-1), requires_grad=False)

    return input, target

def clip_grad(model, clip):
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))
# loop over epoches
prev_loss = None

nbatches = len(bitext.data)
print('number of batches {:d}'.format(nbatches))
start_time = time.time()
total_loss = 0
for i in range(1, nbatches + 1):
    encdec.zero_grad()
    sample = bitext.next()
    inp, target = prepro(sample)
    batch_size = inp[0].data.size(1)
    hidden = encdec.init_hidden(batch_size)
    output = encdec(inp, hidden)
    loss = criterion(output.view(-1, target_size), target)
    loss.backward()
    clipped_lr = args.lr * clip_grad(encdec, args.clip)

    for p in encdec.parameters():
        p.data.sub_(p.grad.mul(clipped_lr))
    total_loss += loss.data[0]
    loss = 0 # do we need it?
    if i%20 == 0:
        elapsed = time.time() - start_time
        cur_loss = total_loss / i
        print('| train perplexity {:.4f} | batch/sec {:.1f}'.format(
            math.exp(cur_loss), i/elapsed))
        #start_time = time.time()
