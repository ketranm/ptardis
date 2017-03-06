import torch
import torch.nn as nn
from torch.autograd import Variable


_INF = 1e10 #float('inf')

class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -_INF)
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn

class Encoder(nn.Module):

    def __init__(self, args):
        self.layers = args.layers
        self.hidden_size = args.rnn_size // 2

        super(Encoder, self).__init__()
        self.lut = nn.Embedding(args.n_words_src, args.emb_size, padding_idx=0)
        self.rnn = nn.LSTM(args.emb_size, self.hidden_size,
                           num_layers=args.layers,
                           dropout=args.dropout,
                           bidirectional=True)

    def forward(self, input, lengths):
        """Forward computation of the encoder. This function handles variable
        length input, lengths is a list of actual sequence lengths of input
        """
        batch_size = input.size(1)
        emb = self.lut(input)
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)
        outputs, hidden = self.rnn(packed_emb)
        return nn.utils.rnn.pad_packed_sequence(outputs)[0], hidden


class StackedLSTM(nn.Module):
    """Simple stacked LSTM layer. This implementation is more memory efficient
    than using nn.LSTM, also it is easier to extend this implementation to
    use residual connections
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_n, c_n = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))
            input = h_i
            if i != self.num_layers:
                input = self.dropout(input)
            h_n += [h_i]
            c_n += [c_i]

        h_n = torch.stack(h_n)
        c_n = torch.stack(c_n)

        return input, (h_n, c_n)


class Decoder(nn.Module):

    def __init__(self, args):
        self.layers = args.layers
        input_size = args.emb_size + args.rnn_size

        super(Decoder, self).__init__()
        self.lut = nn.Embedding(args.n_words_tgt,
                                args.emb_size,
                                padding_idx=0)
        self.rnn = StackedLSTM(args.layers, input_size, args.rnn_size,
                               args.dropout)
        self.attn = GlobalAttention(args.rnn_size)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_size = args.rnn_size

    def forward(self, input, hidden, context, mask=None, init_output=None):
        emb = self.lut(input)
        batch_size = input.size(1)
        h_size = (batch_size, self.hidden_size)

        outputs = []
        if init_output is None:
            output = Variable(emb.data.new(*h_size).zero_(),
                              requires_grad=False)
        else:
            output = init_output
        attns = []

        # set mask
        if mask is not None:
            self.attn.apply_mask(mask)
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)
            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context)
            output = self.dropout(output)
            outputs += [output]
            attns.append(attn)
        attns = torch.stack(attns)
        outputs = torch.stack(outputs)
        return outputs, hidden, attns



class NMT(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(NMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.generate = False

    def set_generate(self, enabled):
        self.generate = enabled

    def _fix_enc_hidden(self, h):
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)


    def forward(self, x, y, lengths_x):
        context, enc_hidden = self.encoder(x, lengths_x)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        x_mask = x.data.eq(0).t() # batch x seqlen
        out, dec_hidden, attn = self.decoder(y, enc_hidden,
                                             context.t(), x_mask)
        if self.generate:
            out = self.generator(out)
        return out
