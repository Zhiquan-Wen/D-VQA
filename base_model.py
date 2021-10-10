"""
This code is modified from SSL-VQA's repository.
https://github.com/CrossmodalGroup/SSL-VQA
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
from torch.nn.utils.weight_norm import weight_norm
import torch
import random

class squeeze(nn.Module):
    def __init__(self):
        super(squeeze, self).__init__()
    
    def forward(self, input):
        return input.squeeze()


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

        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding(opt.dataroot + 'glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1,
                                       bidirect=False, dropout=dropG, rnn_type='GRU')

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.gv_net = FCNet([opt.v_dim, num_hid], dropout=dropL, norm=norm, act=activation)

        self.gv_att_1 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.gv_att_2 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)

        # q_only_branch
        self.q_only = FCNet([self.q_emb.num_hid, num_hid, num_hid], norm=norm, act=activation, dropout=0)
        self.q_cls = weight_norm(nn.Linear(num_hid, opt.ans_dim), dim=None)

        # v_only_branch
        self.v_only = FCNet([opt.v_dim, num_hid, num_hid], norm=norm, act=activation, dropout=0)
        self.v_cls = weight_norm(nn.Linear(num_hid, opt.ans_dim), dim=None)
       

        # q_detect_bias 
        self.q_detect = nn.Sequential(
            weight_norm(nn.Linear(num_hid, 1), dim=None),
            nn.Sigmoid()
        )

        # v detect_bias
        self.v_detect = nn.Sequential(
            weight_norm(nn.Linear(num_hid, 1), dim=None),
            nn.Sigmoid()
        )

        # q and v bias fusion weight
        self.q_and_v_bias_detect = nn.Sequential(
            weight_norm(nn.Linear(num_hid, 1)),
            squeeze(),
            nn.Softmax(dim=-1)
        )

        # debiased branch
        self.debias_only = FCNet([self.q_emb.num_hid, num_hid], norm=norm, dropout=0, act=activation)
        self.debias_cls = weight_norm(nn.Linear(num_hid, opt.ans_dim), dim=None)

        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=opt.ans_dim,
                                           dropout=dropC, norm=norm, act=activation)

        self.normal = nn.BatchNorm1d(num_hid,affine=False)

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
        batch_size = q.size(0)

        out_pos, att_gv_pos = self.compute_predict(q_repr, q_emb, gv_pos, True)

        if self_sup:
            # construct an irrelevant Q-I pair for each instance
            index_v = random.sample(range(0, batch_size), batch_size)
            gv_neg = gv_pos[index_v]
            out_neg_v, att_gv_neg_v = \
                self.compute_predict(q_repr, q_emb, gv_neg, False)
            
            index_q = random.sample(range(0, batch_size), batch_size)
            q_emb_neg = q_emb[index_q]
            q_repr_neg = q_repr[index_q]
            out_neg_q, att_q_neg_q = \
                self.compute_predict(q_repr_neg, q_emb_neg, gv_pos, False) 

            return out_pos, out_neg_v, out_neg_q, att_gv_pos, att_gv_neg_v, att_q_neg_q
        else:
            return out_pos, att_gv_pos

    def compute_predict(self, q_repr, q_emb, v, Need_debias):

        out = {}
        att_1 = self.gv_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.gv_att_2(v, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2

        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.gv_net(gv_emb)

        joint_repr = q_repr * gv_repr

        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)

        if Need_debias:
            # detach, avoid backforward propogation to train front layers
            q_emb_only = q_emb.detach()
            v_emb_only = gv_emb.detach()
            joint_emb = joint_repr.detach()

            # q_only 
            q_only_emb = self.q_only(q_emb_only)  # [batch, num_hid]
            q_only_logits = self.q_cls(q_only_emb)  # [batch, num_ans]
            q_bias_detect = self.q_detect(q_only_emb).view(q_only_emb.size(0), 1)  # [batch, 1]

            # v_only
            v_only_emb = self.v_only(v_emb_only)
            v_only_logits = self.v_cls(v_only_emb)
            v_bias_detect = self.v_detect(v_only_emb).view(v_only_emb.size(0), 1)  # [batch, 1]

            bad_q_bias = q_bias_detect * q_only_emb
            bad_v_bias = v_bias_detect * v_only_emb

            # bias weight
            bad_bias = torch.stack([bad_q_bias, bad_v_bias], dim=1)
            bias_weight = self.q_and_v_bias_detect(bad_bias)

            bad_bias_ = bad_q_bias * bias_weight[:, 0].unsqueeze(1) + bad_v_bias * bias_weight[:, 1].unsqueeze(1)

            debias_emb_raw = joint_emb - bad_bias_
            debias_emb = self.debias_only(debias_emb_raw)
            debias_logits = self.debias_cls(debias_emb)        

            out['logits'] = logits
            out['q_logits'] = q_only_logits
            out['v_logits'] = v_only_logits
            out['debias_logits'] = debias_logits
            out['fea'] = joint_repr
            out['pos_fea'] = debias_emb
            out['neg_fea'] = bad_bias_
        
        else:
            out['logits'] = logits

        return out, att_gv

