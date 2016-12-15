###############################
# Data preparation for NMT
###############################

import cPickle as pkl
from collections import defaultdict
from itertools import izip
import torch
import os
import random

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<pad>':0, '<s>':1, '</s>':2, '<unk>':3}
        self.idx2word = [0, 1, 2, 3]

    def addword(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def size(self):
        return len(self.idx2word)

    def word(self, idx):
        return self.idx2word[idx] if idx < len(self.idx2word) else 1

    def idx(self, word):
        return self.word2idx[word] if word in self.word2idx else '<pad>'

    def make_dict(self, text_file):
        with open(text_file, 'r') as f:
            for line in f:
                ws = line.strip().split()
                for w in ws:
                    self.addword(w)

class BitextIterator(object):
    """Basic Bitext Iterator."""
    def __init__(self, path, source_lang, target_lang, batch_size):
        source = os.path.join(path, 'train.' + source_lang)
        target = os.path.join(path, 'train.' + target_lang)
        source_dict = os.path.join(path, 'dict.' + source_lang)
        target_dict = os.path.join(path, 'dict.' + target_lang)

        self.source_dict = self._load_dict(source_dict, source)
        self.target_dict = self._load_dict(target_dict, target)

        data_file = os.path.join(path, 'train.pt')
        if os.path.isfile(data_file):
            self.data = torch.load(data_file)
        else:
            self.data = self.tensorize(source, target, batch_size)
            torch.save(self.data, data_file)
        self.counter = 0  # count training examples

    def tensorize(self, source, target, batch_size):
        """Convert raw text to torch Tensors."""

        max_diff = 5 # groups sentences into buckets of len 5, 10, 15, ...
        source_bucket = defaultdict(list)
        target_bucket = defaultdict(list)
        data = []

        with open(source) as sf, open(target) as tf:
            for ss, tt in izip(sf, tf):
                ss = self.numberize(self.source_dict, ss)
                tt = self.numberize(self.target_dict, tt, [2], [3])
                tl = (len(tt) + max_diff) - len(tt) % max_diff
                # add padding to the end of target
                for t in range(tl - len(tt)):
                    tt.append(1)
                # reversing the source sentences
                ss.reverse()
                source_bucket[(len(ss), len(tt))].append(ss)
                target_bucket[(len(ss), len(tt))].append(tt)
        for key in source_bucket.keys():
            ss = torch.LongTensor(source_bucket[key]).t().split(batch_size, 1)
            tt = torch.LongTensor(target_bucket[key]).t().split(batch_size, 1)

            for srcb, trgb in zip(ss, tt):
                data.append((srcb, trgb))

        return data

    def _load_dict(self, dict_file, text_file):
        """Load dictionary if it exists, otherwise create it."""

        if os.path.isfile(dict_file):
            return pkl.load(open(dict_file, 'rb'))
        else:
            dict = Dictionary()
            dict.make_dict(text_file)
            pkl.dump(dict, open(dict_file, 'wb'))
            return dict

    def numberize(self, dict, line, leftpads=None, rightpads=None):
        ids = []

        if leftpads:
            ids.extend(leftpads)
        for word in line.strip().split():
            ids.append(dict.idx(word))
        if rightpads:
            ids.extend(rightpads)
        return ids

    def next(self):
        if self.counter == len(self.data):
            random.shuffle(self.data)
            self.counter = 0
        self.counter += 1
        return self.data[self.counter - 1]
