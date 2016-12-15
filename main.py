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
parser.add_argument('-embsize', type=int, default=256, help="embedding size")
parser.add_argument('-hidsize', type=int, default=256, help="hidden size")
parser.add_argument('-nlayers', type=int, default=3, help="number of layers")
# optimization
parser.add_argument('-dropout', type=float, default=0.4, help="dropout rate")
parser.add_argument('-clip', type=float, default=5, help="Gradient clipping")
parser.add_argument('-lr', type=float, default=1.0, help="Learning rate")
parser.add_argument('-decay_after', type=int, default=8,
                    help="decay lr after this epoch!")
parser.add_argument('-decay', type=float, default=0.5, help="decay rate")
parser.add_argument('-maxepochs', type=int, default=13, help="max number epochs")
# cuda
parser.add_argument('-cuda', action='store_true', help="use CUDA")

# Misc
parser.add_argument('-reportint', type=int, default=20, help='Report interval')
parser.add_argument('-saveint', type=int, default=10000,
                    help="Save checkpoint interval")
parser.add_argument('-checkpoint', type=str, default='./checkpoint',
                    help="Checkpoint location")
args = parser.parse_args()

# fix random seed
torch.manual_seed(528491)
torch.cuda.manual_seed(528491)
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: you have a CUDA device, you should probably run with -cuda")

bitext = data.BitextIterator(args.data, args.source, args.target,
                            args.batch_size)

source_size = bitext.source_dict.size()
target_size = bitext.target_dict.size()
encdec = model.NMT(source_size, target_size, args.embsize,
                  args.hidsize, args.nlayers, args.dropout)
weight = torch.ones(target_size)
weight[0] = 0
criterion = nn.CrossEntropyLoss(weight)

# transfer model to cuda
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

# constructing an optimizer

prev_loss = 1e20

nbatches = len(bitext.data)
print('number of batches {:d}'.format(nbatches))

nupdates = 0

for epoch in range(1, args.maxepochs+1):
    total_loss = 0.0
    start_time = time.time()

    for i in range(1, nbatches+1):
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
        loss = 0
        nupdates += 1
        if nupdates % args.reportint == 0:
            elapsed = time.time() - start_time
            cur_loss = total_loss / i
            print('| epoch {:4f} | train ppl {:.4f} | updates {:d} | {:.1f} batch/sec'.format(
                nupdates*1.0/nbatches, math.exp(cur_loss), nupdates, i/elapsed))

        if nupdates % args.saveint == 0:
            checkpoint = '{}/model.{:06d}.pt'.format(args.checkpoint, nupdates)
            with open(checkpoint, 'wb') as f:
                torch.save(encdec, f)
            print('| saved model: ' + checkpoint)

    if prev_loss < total_loss:
        args.lr = args.lr * args.decay
        print('current learning rate {:.5f}'.format(args.lr))
    if args.lr < 1e-5:
        break
    prev_loss = total_loss
