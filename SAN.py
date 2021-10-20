'''we take the SAN backbone as an ablation study, 
   and use the same object features with UpDn for simplicity.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import random
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        num_hid = opt.num_hid
        activation = opt.activation
        dropG = opt.dropG
        dropW = opt.dropW
        dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt
        self.num_layer = 2

        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding(opt.dataroot + 'glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1,
                                       bidirect=False, dropout=dropG, rnn_type='LSTM')

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.gv_net = FCNet([opt.v_dim, num_hid], dropout=dropL, norm=norm, act=activation)

        self.gv_modules = nn.ModuleList()
        self.q_modules = nn.ModuleList()
        self.att_down = nn.ModuleList()

        for _ in range(self.num_layer):
            self.gv_modules.append(nn.Linear(num_hid, num_hid))
            self.q_modules.append(nn.Linear(num_hid, num_hid))
            self.att_down.append(nn.Linear(num_hid, 1))

        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=opt.ans_dim,
                                           dropout=dropC, norm=norm, act=activation)

    def forward(self, q, gv_pos, self_sup=True):

        """Forward
        q: [batch_size, seq_length]
        gv_pos: [batch, K, v_dim]
        self_sup: use negative images or not
        return: logits, not probs
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        q_repr = self.q_net(q_emb)
        gv_pos = self.gv_net(gv_pos)

        logits_pos = self.compute_predict(q_repr, gv_pos)

        return logits_pos

    def compute_predict(self, q_repr, v):

        fea = self.san_att(v, q_repr)

        logits = self.classifier(fea)

        return logits

    def san_att(self, gv_emb, q_emb):  # [batch, 36, 1280], [batch, 1280]
        u = {}
        u[0] = q_emb
        h_A = {}
        p_I = {}

        for k in range(1, self.num_layer + 1):
            h_A[k] = torch.tanh(self.gv_modules[k-1](gv_emb) + self.q_modules[k-1](u[k-1]).unsqueeze(1))    # batch, 36, 1280
            p_I[k] = torch.softmax(self.att_down[k-1](h_A[k]).squeeze(-1), dim=-1)  # batch, 36
            fusion_fea = (p_I[k].unsqueeze(-1) * gv_emb).sum(1)   # batch, num_hid
            u[k] = u[k-1] + fusion_fea

        return u[self.num_layer]

