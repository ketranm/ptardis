import argparse
import torch
import torch.nn as nn
import models
import infer
#from infer import Beam

# build args parser
parser = argparse.ArgumentParser(description='Training NMT')

parser.add_argument('--checkpoint', required=True,
                    help='saved checkpoit.')

parser.add_argument('--input', required=True,
                    help='Text file to translate.')
parser.add_argument('--output', default='trans.bpe', help='output file')
parser.add_argument('--ref', default='', help='reference file')

parser.add_argument('--beam_size', default=5, type=int,
                    help="Beam size.")
parser.add_argument('--gpus', default=[], nargs='+', type=int,
                   help="Use CUDA")


args = parser.parse_args()
args.cuda = len(args.gpus)
#print(args)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")
if args.cuda:
    torch.cuda.set_device(args.gpus[0])

checkpoint = torch.load(args.checkpoint)
saved_args =  checkpoint['args']

encoder = models.Encoder(saved_args)
decoder = models.Decoder(saved_args)
generator = nn.Sequential(
    nn.Linear(saved_args.rnn_size, saved_args.n_words_tgt),
    nn.LogSoftmax()
)
model = models.NMT(encoder, decoder, generator)

if args.cuda:
    model.cuda()
model.load_state_dict(checkpoint['params'])

# overwritten some options
saved_args.beam_size = args.beam_size
if args.beam_size == 0 and args.ref != '':
    agent = infer.MyPolicy(saved_args, model)
    #agent = infer.RefPolicy(saved_args, model)
    agent.get_scores(args.input, args.ref)
else:
    agent = infer.Beam(saved_args, model)
    agent.translate(args.input, args.output)
