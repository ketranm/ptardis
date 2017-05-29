import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_iterator import TextIterator
import models
import math
import time
import subprocess
from infer import Beam
import re
import torch.nn.init as init
# build args parser
parser = argparse.ArgumentParser(description='Training NMT')

parser.add_argument('--datasets', required=True, default=[],
                    nargs='+', type=str,
                    help='source_file target_file.')
parser.add_argument('--valid_datasets', required=True, default=[],
                    nargs='+', type=str,
                    help='valid_source valid target files.')
parser.add_argument('--ref', required=True, default=None,
                    help="reference file")
# dictionaries
parser.add_argument('--dicts', required=True, default=[],
                    nargs='+',
                    help='source_vocab.pkl target_vocab.pkl files.')
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
parser.add_argument('--clip', type=float, default=5.,
                    help='Gradient norm clip threshold.')
parser.add_argument('--max_epochs', type=int, default=15,
                    help='Maximum number of epochs.')
parser.add_argument('--finish_after', type=int, default=1000000,
                    help='Maximum number of iterations.')
parser.add_argument('--beam_size', type=int, default=5,
                    help="size of the beam for decoding.")

# Memory management
parser.add_argument('--max_generator_batches', type=int, default=32,
                    help="""split updates of the outputs to smaller chunks.
                    This is useful for reduce memory usage.""")
# Utils
parser.add_argument('--gpus', default=[], nargs='+', type=int,
                    help="Use CUDA")
parser.add_argument('--seed', default=528491, type=int,
                    help="inception seed.")
parser.add_argument('--report_freq', type=int, default=20,
                    help="display training progress.")
parser.add_argument('--valid_freq', type=int, default=20,
                    help="evaluate after every this number of updates.")
parser.add_argument('--saveto', default='tardis.pt',
                    help="saved file of train model.")
parser.add_argument('--load', default=None,
                    help="trained model.")
parser.add_argument('--output', default='output', help='temporal output files.')

args = parser.parse_args()
args.cuda = len(args.gpus)

print 'configuration'
print(args)
print '-' * 42
torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.cuda:
    print("""WARNING: You have a CUDA device,
          so you should probably run with -gpus 0""")

if args.cuda:
    torch.cuda.set_device(args.gpus[0])
    torch.cuda.manual_seed(args.seed)

# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, eval=False):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    # sort source sentences in decreasing length
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
    if args.cuda:
        x = x.cuda()
        y = y.cuda()
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



def eval(model, valid, generator, crit):
    """Evaluate model based on perplexity"""
    valid_nlls= []
    n_words = 0
    for x, y in valid:
        x, y, lengths_x = prepare_data(x, y, maxlen=50, eval=True)
        outputs = model(x, y[:-1], lengths_x)
        nll, _ = memory_efficient_loss(outputs, y[1:], generator, crit, True)
        valid_nlls.append(nll)
        n_words += y[1:].data.ne(0).int().sum()
    return torch.FloatTensor(valid_nlls).sum() / n_words


def build_model(args):
    encoder = models.Encoder(args)
    decoder = models.Decoder(args)
    generator = nn.Sequential(
        nn.Linear(args.rnn_size, args.n_words_tgt),
        nn.LogSoftmax()
    )
    model = models.NMT(encoder, decoder, generator)
    return model

def init_model(model):
    params = model.state_dict()
    for k, weight in params.iteritems():
        w_name = k.split('.')[-1]
        if w_name.startswith('weight_hh'):
            tmp = torch.FloatTensor(weight.size())
            init.orthogonal(tmp, 1)
            weight.copy_(tmp)

def init_uniform(model, init_range=0.04):
    for p in model.parameters():
        p.data.uniform_(-init_range, init_range)

def train(args):
    subprocess.call(['python', './data/shuffle.py', args.datasets[0], args.datasets[1]])
    print '| build data iterators'
    train = TextIterator(args.datasets[0] + '.shuf', args.datasets[1] + '.shuf',
                         args.dicts[0], args.dicts[1],
                         n_words_source=args.n_words_src,
                         n_words_target=args.n_words_tgt,
                         batch_size=args.batch_size,
                         maxlen=100)
    valid = TextIterator(args.valid_datasets[0], args.valid_datasets[1],
                         args.dicts[0], args.dicts[1],
                         n_words_source=args.n_words_src,
                         n_words_target=args.n_words_tgt,
                         batch_size=args.batch_size,
                         maxlen=100)

    print '| build model'
    if args.n_words_src < 0:
        args.n_words_src = len(train.source_dict)
    if args.n_words_tgt < 0:
        args.n_words_tgt = len(train.target_dict)

    print '| source vocab size %d' % args.n_words_src
    print '| target vocab size %d' % args.n_words_tgt
    dicts = [train.source_dict, train.target_dict]

    print '| build criterion'
    crit = build_crit(args.n_words_tgt)

    print '| build NMT model'
    model = build_model(args)
    #init_model(model)
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    uidx = 0 # number of updates
    estop = False
    history_errs = []
    history_bleus = [0]
    for eidx in xrange(args.max_epochs):
        if eidx > 0:
            print '| shuffling training data.'
            subprocess.call(['python', './data/shuffle.py', args.datasets[0], args.datasets[1]])
        n_samples = 0
        tot_loss = 0
        n_words = 0
        ud_start = time.time()
        for x, y in train:
            optimizer.zero_grad()
            n_samples += len(x)
            x, y, lengths_x = prepare_data(x, y, maxlen=50)

            # compute loss and update model's parameters
            outputs = model(x, y[:-1], lengths_x)
            if args.max_generator_batches > 0:
                loss, df_do = memory_efficient_loss(outputs, y[1:], model.generator, crit)
                outputs.backward(df_do)
            else:
                pred = model.generator(outputs.view(-1, outputs.size(2)))
                loss = crit(pred, y[1:].view(-1))
                loss.div(outputs.size(1)).backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            uidx += 1
            tot_loss += loss
            n_words += y[1:].data.ne(0).int().sum()
            if uidx % args.report_freq == 0:
                ud = time.time() - ud_start
                fargs = [eidx, uidx, math.exp(tot_loss/n_words),
                         args.report_freq/ud]
                print("epoch {:2d} | update {:5d} | ppl {:.3f} "
                      "| speed {:.1f} b/s".format(*fargs))
                ud_start = time.time()

            if uidx % args.valid_freq == 0:
                model.eval()
                valid_nll = eval(model, valid, model.generator, crit)
                valid_ppl = math.exp(valid_nll)
                history_errs.append(valid_ppl)
                # resume training mode
                print('| update {:5d} valid ppl {:.3f}'.format(uidx, valid_ppl))
                print '| translate ...'
                infer = Beam(args, model)
                args.beam_size = 5
                out_bpe = '%s.%d.bpe' % (args.output, uidx)
                out_txt = '%s.%d.txt' % (args.output, uidx)
                infer.translate(args.valid_datasets[0], out_bpe)
                #print '| recovering from BPE
                subprocess.call("sed 's/@@ //g' {:s} > {:s}".format(out_bpe, out_txt), shell=True)
                cmd = "perl data/multi-bleu.perl {} < {}".format(args.ref, out_txt)
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
                bleu = re.search("[\d]+.[\d]+", p)
                bleu = bleu.group()
                bleu = float(bleu)
                print('| update {:5d}  BLEU {:.2f}'.format(uidx, bleu))
                if bleu >= torch.FloatTensor(history_bleus).max():
                    print '| Save checkpoint'
                    checkpoint = {
                        'params': model.state_dict(),
                        'args': args,
                        'history_errs': history_errs,
                        'history_bleus': history_bleus,
                        'dicts': dicts
                    }
                    torch.save(checkpoint, args.saveto)
                model.train()
                history_bleus.append(bleu)

            if uidx >= args.finish_after:
                print('Finishing after {:d} iterations!'.format(uidx))
                estop = True
                break

        print('Seen {:d} samples'.format(n_samples))
        if estop:
            break

train(args)
