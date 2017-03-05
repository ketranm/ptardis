import argparse
import torch
import torch.nn as nn

from torch.autograd import Variable
from data_iterator import TextIterator
import models


# build args parser
parser = argparse.ArgumentParser(description='Training NMT')

parser.add_argument('--src_train', required=True,
                    help='Path to source train file.')
parser.add_argument('--tgt_train', required=True,
                    help='Path to target train file.')
parser.add_argument('--src_valid', required=True,
                    help='Path to source valid file.')
parser.add_argument('--tgt_valid', required=True,
                    help='Path to target valid file.')
# dictionaries
parser.add_argument('--src_dict', required=True,
                    help='Path to source vocab file.')
parser.add_argument('--tgt_dict', required=True,
                    help='Path to target vocab file.')

parser.add_argument('--n_words_src', type=int, default=-1,
                    help='Number of source words')
parser.add_argument('--n_words_tgt', type=int, default=-1,
                    help='Number of target words')
# Model options

parser.add_argument('--layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('--rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('--emb_size', type=int, default=500,
                    help='Word embedding size')

# Optimization
parser.add_argument('--batch_size', type=int, default=32,
                    help='Maximum batch size.')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout rate.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate.')
parser.add_argument('--clip', type=float, default=1.,
                    help='Gradient norm clip threshold.')
# Memory management
parser.add_argument('--max_generator_batches', type=int, default=32,
                    help='Flush the output by this number')


parser.add_argument('--cuda', type=bool, default=False,
                    help='Using cuda.')
args = parser.parse_args()
print(args)

# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000, eval=False):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    # sort source sentences in dicresing length
    sorted_x, indices = torch.sort(torch.LongTensor(lengths_x), 0, True)

    new_seqs_x = []
    new_seqs_y = []
    for i, idx in enumerate(list(indices)):
        if sorted_x[i] == 0:
            continue
        new_seqs_x.append(seqs_x[idx])
        new_seqs_y.append(seqs_y[idx])
    seqs_x = new_seqs_x
    seqs_y = new_seqs_y
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    maxlen_x = torch.IntTensor(lengths_x).max()
    maxlen_y = torch.IntTensor(lengths_y).max()
    n_samples = len(lengths_x)
    x = torch.zeros(maxlen_x, n_samples).long()
    y = torch.zeros(maxlen_y, n_samples).long()
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = torch.LongTensor(s_x)
        y[:lengths_y[idx], idx] = torch.LongTensor(s_y)
    x = Variable(x, volatile=eval)
    y = Variable(y, volatile=eval)
    return x, y, lengths_x


def build_crit(n_words):
    weight = torch.ones(n_words)
    weight[0] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if args.cuda:
        crit.cuda()
    return crit


def memory_efficient_loss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval).contiguous()

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, args.max_generator_batches)
    targets_split = torch.split(targets.contiguous(), args.max_generator_batches)
    for out_t, targ_t in zip(outputs_split, targets_split):
        out_t = out_t.view(-1, out_t.size(2))
        pred_t = generator(out_t)
        loss_t = crit(pred_t, targ_t.view(-1))
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output


def train(args):
    train = TextIterator(args.src_train, args.tgt_train,
                         args.src_dict, args.tgt_dict,
                         n_words_source=args.n_words_src,
                         n_words_target=args.n_words_tgt,
                         batch_size=args.batch_size,
                         maxlen=100)
    print '| build model'
    if args.n_words_src < 0:
        args.n_words_src = len(train.source_dict)
    if args.n_words_tgt < 0:
        args.n_words_tgt = len(train.target_dict)

    print '| build criterion'
    crit = build_crit(args.n_words_tgt)

    encoder = models.Encoder(args)
    decoder = models.Decoder(args)
    generator = nn.Sequential(
        nn.Linear(args.rnn_size, args.n_words_tgt),
        nn.LogSoftmax()
    )
    model = models.NMT(encoder, decoder, generator)

    if args.cuda:
        model.cuda()

    print '| Use Adam optimizer by default!'
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_samples = 0
    for x, y in train:
        n_samples += len(x)
        x, y, lengths_x = prepare_data(x, y)
        outputs = model(x, y[:-1], lengths_x)
        
        #log_prob = generator(outputs.view())
        loss, df_do = memory_efficient_loss(outputs, y[1:], generator, crit)
        outputs.backward(df_do)

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        print loss

        print '-' * 100
    pass

train(args)
