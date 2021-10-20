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
from torch.nn.utils.weight_norm import weight_norm


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


        # q_only_branch
        self.q_only = FCNet([self.q_emb.num_hid, num_hid, num_hid], norm=norm, act=activation, dropout=0)
        self.q_cls = weight_norm(nn.Linear(num_hid, opt.ans_dim), dim=None)

        self.v_only = FCNet([num_hid, num_hid, num_hid], norm=norm, act=activation, dropout=0)
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

        # q and v bias weight
        self.q_and_v_bias_detect = nn.Sequential(
            weight_norm(nn.Linear(num_hid, 1)),
            squeeze(),
            nn.Softmax(dim=-1)
        )

        # fusion_branch
        self.debias_only = FCNet([self.q_emb.num_hid, num_hid], norm=norm, dropout=0, act=activation)
        self.debias_cls = weight_norm(nn.Linear(num_hid, opt.ans_dim), dim=None)


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
        batch_size = q.size(0)

        out = self.compute_predict(q_repr, q_emb, gv_pos, True)

        if self_sup:
            # construct an irrelevant Q-I pair for each instance
            index_v = random.sample(range(0, batch_size), batch_size)
            gv_neg = gv_pos[index_v]
            out_neg_v = \
                self.compute_predict(q_repr, None, gv_neg, False)
            
            index_q = random.sample(range(0, batch_size), batch_size)
            q_repr_neg = q_repr[index_q]
            out_neg_q = \
                self.compute_predict(q_repr_neg, None, gv_pos, False) 
            return out, out_neg_v, out_neg_q
        else:
            return out

    def compute_predict(self, q_repr, q_emb, v, Need_bias=False):

        fea = self.san_att(v, q_repr)

        logits = self.classifier(fea)

        out = {}

        if Need_bias:
            # detach, avoid backforward propogation to train front layers
            q_emb_only = q_emb.detach()
            v_emb_only = v.detach()
            joint_emb = fea.detach()

            # q_only 
            q_only_emb = self.q_only(q_emb_only)  # [batch, num_hid]
            q_only_logits = self.q_cls(q_only_emb)  # [batch, num_ans]
            q_bias_detect = self.q_detect(q_only_emb).view(q_only_emb.size(0), 1)  # [batch, 1]

            # v_only
            v_emb_only = v_emb_only.mean(1)
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
            out['fea'] = fea
            out['pos_fea'] = debias_emb
            out['neg_fea'] = bad_bias_
        else:
            out['logits'] = logits

        return out

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

