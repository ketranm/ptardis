import torch
from torch.autograd import Variable
import cPickle as pkl

_INF = 1e10
class Beam(object):
    """
    Beam search class for NMT
    """
    def __init__(self, args, model):
        self.args = args
        self.tt = torch.cuda if len(args.gpus) >0  else torch
        self.model = model
        self.model.eval()
        self.dicts = [pkl.load(open(args.dicts[0], 'rb')),
                      pkl.load(open(args.dicts[1], 'rb'))]
        self.idx2w = {}
        for w, idx in self.dicts[1].iteritems():
            self.idx2w[idx] = w
        self.bos_idx = self.dicts[1]['<bos>']
        self.eos_idx = self.dicts[1]['<eos>']

    def encode_string(self, ss):
        ss = ss.split()
        ss = [self.dicts[0][w] if w in self.dicts[0] else 1
              for w in ss]
        if self.args.n_words_src > 0:
            ss = [w if w < self.args.n_words_src else 1 for w in ss]
        ss = Variable(torch.LongTensor(ss).view(-1, 1),
                        volatile=True)
        if self.args.cuda:
            ss = ss.cuda()
        return ss

    def decode_string(self, tidx):
        ts = [self.idx2w[i] for i in list(tidx)]
        return ' '.join(ts)

    def beam_search(self, input):
        """
        Args:
            input: Tensor (seqlen x 1)
        """

        k = self.args.beam_size
        completed_hyps = []

        input = input.expand(input.size(0), k)
        max_len = int(input.size(0) * 1.5)
        hypos = self.tt.LongTensor(max_len, k).fill_(2)

        init_target = self.tt.LongTensor(1, k).fill_(2)
        init_target = Variable(init_target, volatile=True)

        scores = self.tt.FloatTensor(k).fill_(-_INF)
        scores[0] = 0
        lengths = [input.size(0) for i in range(k)]

        context, enc_hidden = self.model.encoder(input, lengths)

        init_hidden = (self.model._fix_enc_hidden(enc_hidden[0]),
                       self.model._fix_enc_hidden(enc_hidden[1]))

        # alias
        decoder = self.model.decoder
        generator = self.model.generator

        init_output = None
        context = context.t()
        decoder.attn.mask = None
        for t in xrange(max_len):
            out, dec_hidden, attn = decoder(init_target, init_hidden, context, init_output=init_output)
            log_probs = generator(out.squeeze(0)).data
            scores_t, idx_t = log_probs.topk(k, 1)
            scores_t = scores_t + scores.view(-1, 1).expand_as(scores_t)

            scores, k_idx = scores_t.view(-1).topk(k)
            next_hp = k_idx.div(k)
            next_ys = idx_t.view(-1).index_select(0, k_idx)


            done_y = next_ys.eq(self.eos_idx)
            if done_y.sum() > 0 and t > 0:
                for i in range(k):
                    if next_ys[i] == 3:
                        j = next_hp[i]
                        text = self.decode_string(hypos[0:t, j])
                        completed_hyps.append((text, scores[i] / (t+1) ))
                        k -= 1
                if k > 0:
                    cont_y = next_ys.ne(self.eos_idx)
                    next_ys = next_ys.masked_select(cont_y)
                    next_hp = next_hp.masked_select(cont_y)
                    context = context[:k]
                    scores = scores.masked_select(cont_y)
            if k == 0:
                break
            hypos = hypos.index_select(1, next_hp)
            hypos[t] = next_ys
            init_target = Variable(next_ys.view(1, -1), volatile=True)
            next_hp = Variable(next_hp)
            init_output = out.squeeze(0).index_select(0, next_hp)
            init_hidden = [h.index_select(1, next_hp) for h in dec_hidden]

        if len(completed_hyps) > 0:
            completed_hyps.sort(key=lambda tup: tup[1])
            best_h = completed_hyps.pop()
            return best_h[0]
        else:
            best_s, idx = scores.topk(1)
            best_h = hypos.index_select(0, idx).view(-1)
            return self.decode_string(best_h)

    def translate(self, text_file, out_file='output.txt'):
        fw = open(out_file, 'w')
        for line in open(text_file):
            src_idx = self.encode_string(line)
            s = self.beam_search(src_idx)
            fw.write(s + '\n')
        fw.close()
