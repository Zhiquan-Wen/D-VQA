# coding=utf-8
# Copyleft 2019 project LXRT.
import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

from lxrt.fc import FCNet, GTH
from lxrt.attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
from torch.nn.utils.weight_norm import weight_norm
import torch
import random

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = args.MAX_VQA_LENGTH

class squeeze(nn.Module):
    def __init__(self):
        super(squeeze, self).__init__()
    
    def forward(self, input):
        return input.squeeze()


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.lxrt_encoder.load(args.load_lxmert)


        ########### init our layers ###########
        activation = 'ReLU'
        norm = 'weight'

        # q_only_branch
        self.q_only = FCNet([hid_dim, hid_dim, hid_dim], norm=norm, act=activation, dropout=0)
        self.q_cls = weight_norm(nn.Linear(hid_dim, num_answers), dim=None)

        self.v_only = FCNet([hid_dim, hid_dim, hid_dim], norm=norm, act=activation, dropout=0)
        self.v_cls = weight_norm(nn.Linear(hid_dim, num_answers), dim=None)
       
        # q_detect_bias 
        self.q_detect = nn.Sequential(
            weight_norm(nn.Linear(hid_dim, 1), dim=None),
            nn.Sigmoid()
        )

        # v detect_bias
        self.v_detect = nn.Sequential(
            weight_norm(nn.Linear(hid_dim, 1), dim=None),
            nn.Sigmoid()
        )

        # q and v bias weight
        self.q_and_v_bias_detect = nn.Sequential(
            weight_norm(nn.Linear(hid_dim, 1)),
            squeeze(),
            nn.Softmax(dim=-1)
        )

        # fusion_branch
        self.debias_only = FCNet([hid_dim, hid_dim], norm=norm, dropout=0, act=activation)
        self.debias_cls = weight_norm(nn.Linear(hid_dim, num_answers), dim=None)
        
    def use_bias(self, lang_feats, visn_feats, l_cls, v_cls, x, out):
        # detach, avoid backforward propogation to train front layers
        q_emb_only = l_cls.detach()
        v_emb_only = v_cls.detach()
        joint_emb = x.detach()

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

        out['q_logits'] = q_only_logits
        out['v_logits'] = v_only_logits
        out['debias_logits'] = debias_logits
        out['fea'] = joint_emb  # joint_repr
        out['pos_fea'] = debias_emb
        out['neg_fea'] = bad_bias_
        return out

    def forward(self, feat, pos, sent, self_sup=False, out={}):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :param need_bias: True for train, False for test
        :param self_supp: True when epoch>pretraining_epoches
        :return: (b, num_answer) The logit of each answers.
        """
        (lang_feats, visn_feats, l_cls, v_cls), x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)
        out['logits'] = logit
        out = self.use_bias(lang_feats, visn_feats, l_cls, v_cls, x, out)
        if self_sup:
            # construct an irrelevant Q-I pair for each instance
            batch_size = lang_feats.size(0)
            # V
            index_v = random.sample(range(0, batch_size), batch_size)
            neg_v = feat[index_v]
            neg_pos = pos[index_v]
            _, x = self.lxrt_encoder(sent, (neg_v, neg_pos))
            out['logit_neg_v'] = self.logit_fc(x)
            # Q            
            index_q = random.sample(range(0, batch_size), batch_size)
            neg_sent = [sent[i] for i in index_q]
            _, x = self.lxrt_encoder(neg_sent, (feat, pos))
            out['logit_neg_q'] = self.logit_fc(x)
        return out