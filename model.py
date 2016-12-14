import torch
import torch.nn as nn
from torch.autograd import Variable

class Sequencer(nn.Container):
    """A recurrent transducer"""

    def __init__(self, dict_size, embsize, hidsize, nlayers, dropout_rate):
        super(Sequencer, self).__init__(
            embedding = nn.sparse.Embedding(dict_size, embsize),
            rnn = nn.LSTM(embsize, hidsize, nlayers, dropout=dropout_rate)
        )

        # TODO: initialize parameters

    def forward(self, input, hidden):
        emb = self.embedding(input)
        output, hidden = self.rnn(emb, hidden)
        return output, hidden

class NMTA(nn.Container):
    """Attentional neural machine translation"""

    def __init__(self, source_size, target_size,
                embsize, hidsize, nlayers, dropout_rate):
        super(NMT, self).__init__(
            encoder = Sequencer(source_size, embsize, hidsize,
                                nlayers, dropout_rate),
            decoder = Sequencer(target_size, embsize, hidsize,
                                nlayers, dropout_rate),
            classifier = nn.Linear(hidsize, target_size)
        )

        self.hidsize = hidsize
        self.nlayers = nlayers

    def forward(self, input, ini_hidden):
        enc_output, enc_hidden = self.encoder(input[0], ini_hidden)
        dec_output, dec_hidden = self.decoder(input[1], enc_hidden)
        # now computing dot attention

        output = self.classifier(dec_output.view(-1, self.hidsize))
        output = output.view(dec_output.size(0), dec_output.size(1), -1)
        return output

    def init_hidden(self, batch_size):
        """Generate the first hidden states for encoder."""
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size,
                                    self.hidsize).zero_()),
                Variable(weight.new(self.nlayers, batch_size,
                                    self.hidsize).zero_()))


class NMT(nn.Container):
    """Simple seq2seq model, no attention"""

    def __init__(self, source_size, target_size,
                embsize, hidsize, nlayers, dropout_rate):
        super(NMT, self).__init__(
            encoder = Sequencer(source_size, embsize, hidsize,
                                nlayers, dropout_rate),
            decoder = Sequencer(target_size, embsize, hidsize,
                                nlayers, dropout_rate),
            classifier = nn.Linear(hidsize, target_size)
        )

        self.hidsize = hidsize
        self.nlayers = nlayers

    def forward(self, input, hidden):
        _, enc_hidden = self.encoder(input[0], hidden)
        dec_output, dec_hidden = self.decoder(input[1], enc_hidden)
        output = self.classifier(dec_output.view(-1, self.hidsize))
        output = output.view(dec_output.size(0), dec_output.size(1), -1)
        return output

    def init_steps(self, input, hidden):
        """run the encoder and initialize the decoder"""
        _, enc_hidden = self.encoder(input, hidden)
        return (Variable(enc_hidden[1].data), Variable(enc_hidden[2].data))

    def step(self, input, hidden):
        """one step forward of decoder"""

    def init_hidden(self, batch_size):
        """Generate the first hidden states for encoder."""
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size,
                                    self.hidsize).zero_()),
                Variable(weight.new(self.nlayers, batch_size,
                                    self.hidsize).zero_()))
