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

class NMT(nn.Container):
    def __init__(self, source_size, target_size,
                embsize, hidsize, nlayers, dropout_rate):
        super(NMT, self).__init__(
            encoder = Sequencer(source_size, embsize, hidsize, nlayers, dropout_rate),
            decoder = Sequencer(target_size, embsize, hidsize, nlayers, dropout_rate),
            classifier = nn.Linear(hidsize, target_size)
        )

        self.hidsize = hidsize

    def forward(self, input, hidden):
        enc_output, enc_hidden = self.encoder(input[1], hidden)
        dec_output, dec_hidden = self.decoder(input[2], enc_hidden)
        output = classifier(dec_output.view(-1, self.hidsize))
        output = output.view(dec_output.size(0), dec_output.size(1), -1)
