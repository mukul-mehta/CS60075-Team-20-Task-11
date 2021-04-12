import os
import copy
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import time
import utils

from transformers import AutoTokenizer, AutoModel


class BiLSTM_CRF(nn.Module):

    def __init__(self, biluo_code, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.biluo_code = biluo_code
        self.tagset_size = len(biluo_code)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.modell = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased")

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[biluo_code[START_TAG], :] = -10000
        self.transitions.data[:, biluo_code[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def forward_scorer(self, feats):

        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)

        init_alphas[0][self.biluo_code[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):

                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)

                trans_score = self.transitions[next_tag].view(1, -1)

                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(utils.log_func(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + \
            self.transitions[self.biluo_code[STOP_TAG]]
        alpha = utils.log_func(terminal_var)
        return alpha

    def get_lstm_features(self, sentence):

        outputs = self.modell(**sentence, output_hidden_states=True)
        scibert_out = ((outputs[2][12])[0]).view(
            len(sentence["input_ids"][0]), 1, -1)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(scibert_out, self.hidden)
        lstm_out = lstm_out.view(
            len(sentence["input_ids"][0]), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def sentence_scorer(self, feats, tags):

        score = torch.zeros(1).to(device)
        tags = torch.cat(
            [torch.tensor([self.biluo_code[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.biluo_code[STOP_TAG], tags[-1]]
        return score

    def viterbi_decoder(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.biluo_code[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):

                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = utils.maxval(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + \
            self.transitions[self.biluo_code[STOP_TAG]]
        best_tag_id = utils.maxval(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.biluo_code[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):

        feats = self.get_lstm_features(sentence)
        forward_score = self.forward_scorer(feats)
        gold_score = self.sentence_scorer(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):

        lstm_feats = self.get_lstm_features(sentence)

        score, tag_seq = self.viterbi_decoder(lstm_feats)
        return score, tag_seq
