# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    
    parser.add_argument("--data", default="/")

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=64)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=True, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=1)
    
    parser.add_argument("--fn_type", default="softmax")
    parser.add_argument("--dump",default=False,action='store_const', const=True)
    parser.add_argument("--mutant",default=False,action='store_const', const=True)
    parser.add_argument("--nops", dest='nops', default=1)
    parser.add_argument("--MAX_VQA_LENGTH", dest='MAX_VQA_LENGTH', default=20)

    parser.add_argument('--dataroot', default=None, type=str)
    parser.add_argument('--img_root', default=None, type=str)
    parser.add_argument('--grad_clip', type=float, default=0.25,
                    help='clip gradients at this value')
    parser.add_argument('--ratio', default=1)
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--checkpoint_path', type=str, default='',
                    help='directory to store checkpointed models')
    parser.add_argument('--num_epochs', type=int, default=15,
                    help='number of epochs')

    parser.add_argument('--s_epoch', type=int, default=0,
                    help='training from s epochs')

    parser.add_argument('--pretrain_epoches', type=int, default=6,
                    help='introduce negective pairs')
    parser.add_argument('--self_loss_q', type=float, default=0.7,
                    help='self_loss_q')
    parser.add_argument('--self_loss_weight', type=float, default=3,
                    help='self_loss_weight')

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
