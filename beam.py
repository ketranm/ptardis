# beam search
import torch
import torch.nn as nn
from torch.autograd import Variable

logsoftmax = nn.LogSoftmax()

def find_topk(mat, k):
    """x is a 2D tensor, return results, row, col"""
    res, idx = mat.view(-1).topk(k, 0, True)
    row = idx / mat.size(1)
    col = idx - (idx/mat.size(1)) * mat.size(1)
    return res, row, col

def beam_search(model, input, beam_size, max_len, bos_idx, eos_idx):
    """beam search, currently support batch_size 1"""
    #print(input)
    #input_expand = input.expand(input.size(0), beam_size)
    input = Variable(input, requires_grad=False)
    hidden = model.init_hidden(1)
    # run the encoder
    hidden = model.init_step(input, hidden)
    input = Variable(input.data.new(1, 1).fill_(bos_idx),
                     requires_grad=False)
    hypo = input.data.new(max_len, beam_size)
    hypo_score = torch.zeros(beam_size).cuda()

    for t in range(max_len):
        output, hidden = model.step(input, hidden)
        output = logsoftmax(output)
        score, row, col = find_topk(output.data, beam_size)
        hypo_score += score
        # check if eos is fond
        if col.eq(eos_idx).sum() > 0:
            print "eos found"
            print(score)
        # next input
        hypo[t,:] = col
        input = Variable(col.view(1, -1), requires_grad=False)
        # select next hidden
        hidden = (Variable(hidden[0].data.index_select(1, row), requires_grad=False),
                  Variable(hidden[1].data.index_select(1, row), requires_grad=False))

        #print(hidden[0].data.index_select(1, col))
        #print(score, row, col)
        #break
        # compute the next input
        # compute the next hidden
    best_score, best_idx = hypo_score.topk(1, 0 , True)
    best_hypo = hypo[:, best_idx[0]]
    return best_hypo
