import os
import torch
import random
import numpy as np
import logging
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError:
    print('Import sklearn.metrics failed!')

import yaml
from easydict import EasyDict

_logger = None
_logger_fh = None
_logger_names = []

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config

def create_logger(log_file, level=logging.INFO):
    global _logger, _logger_fh
    if _logger is None:
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        _logger.setLevel(level)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
        _logger_fh = fh
    else:
        _logger.removeHandler(_logger_fh)
        _logger.setLevel(level)

    return _logger


def get_logger(name, level=logging.INFO):
    global _logger_names
    logger = logging.getLogger(name)
    if name in _logger_names:
        return logger

    _logger_names.append(name)

    return logger

def count_params(model):
    logger = get_logger(__name__)

    total = sum(p.numel() for p in model.parameters())
    conv = 0
    fc = 0
    others = 0
    for name, m in model.named_modules():
        # skip non-leaf modules
        if len(list(m.children())) > 0:
            continue
        num = sum(p.numel() for p in m.parameters())
        if isinstance(m, torch.nn.Conv2d):
            conv += num
        elif isinstance(m, torch.nn.Linear):
            fc += num
        else:
            others += num

    M = 1e6

    logger.info('total param: {:.3f}M, conv: {:.3f}M, fc: {:.3f}M, others: {:.3f}M'
                .format(total/M, conv/M, fc/M, others/M))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def detailed_metrics(output, target):
    precision_class = precision_score(target, output, average=None)
    recall_class = recall_score(target, output, average=None)
    f1_class = f1_score(target, output, average=None)
    precision_avg = precision_score(target, output, average='micro')
    recall_avg = recall_score(target, output, average='micro')
    f1_avg = f1_score(target, output, average='micro')
    return precision_class, recall_class, f1_class, precision_avg, recall_avg, f1_avg


def load_state_model(model, state):

    logger = get_logger(__name__)
    logger.info('======= loading model state... =======')
    model.load_state_dict(state, strict=False)

    state_keys = set(state.keys())
    model_keys = set(model.state_dict().keys()) 
    missing_keys = model_keys - state_keys
    for k in missing_keys:
        logger.warn(f'missing key: {k}')


def load_state_optimizer(optimizer, state):

    logger = get_logger(__name__)
    logger.info('======= loading optimizer state... =======')

    optimizer.load_state_dict(state)

def load_state_variable(variable, state):

    logger = get_logger(__name__)
    logger.info('======= loading variable state... =======')
    variable.data.copy_(state.detach())

def modify_state(state, config):
    if hasattr(config, 'key'):
        for key in config['key']:
            if key == 'optimizer':
                state.pop(key)
            elif key == 'last_iter':
                state['last_iter'] = 0
            elif key == 'ema':
                state.pop('ema')

    if hasattr(config, 'model'):
        for module in config['model']:
            state['model'].pop(module)
    return state

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def param_group_all(model, config):
    logger = get_logger(__name__)
    pgroup_normal = []
    pgroup = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    names = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    if 'conv_dw_w' in config:
        pgroup['conv_dw_w'] = []
        names['conv_dw_w'] = []
    if 'conv_dw_b' in config:
        pgroup['conv_dw_b'] = []
        names['conv_dw_b'] = []
    if 'conv_dense_w' in config:
        pgroup['conv_dense_w'] = []
        names['conv_dense_w'] = []
    if 'conv_dense_b' in config:
        pgroup['conv_dense_b'] = []
        names['conv_dense_b'] = []
    if 'linear_w' in config:
        pgroup['linear_w'] = []
        names['linear_w'] = []

    names_all = []
    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                if 'conv_dw_b' in pgroup and m.groups == m.in_channels:
                    pgroup['conv_dw_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_dw_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias(dw)'] += 1
                elif 'conv_dense_b' in pgroup and m.groups == 1:
                    pgroup['conv_dense_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_dense_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias(dense)'] += 1
                else:
                    pgroup['conv_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias'] += 1
            if 'conv_dw_w' in pgroup and m.groups == m.in_channels:
                pgroup['conv_dw_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['conv_dw_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight(dw)'] += 1
            elif 'conv_dense_w' in pgroup and m.groups == 1:
                pgroup['conv_dense_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['conv_dense_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight(dense)'] += 1

        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                pgroup['linear_b'].append(m.bias)
                names_all.append(name+'.bias')
                names['linear_b'].append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
            if 'linear_w' in pgroup:
                pgroup['linear_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['linear_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
        elif (isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d)):
            if m.weight is not None:
                pgroup['bn_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['bn_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                pgroup['bn_b'].append(m.bias)
                names_all.append(name+'.bias')
                names['bn_b'].append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1

    for name, p in model.named_parameters():
        if name not in names_all:
            pgroup_normal.append(p)

    param_groups = [{'params': pgroup_normal}]
    for ptype in pgroup.keys():
        if ptype in config.keys():
            param_groups.append({'params': pgroup[ptype], **config[ptype]})
        else:
            param_groups.append({'params': pgroup[ptype]})

        logger.info(ptype)
        for k, v in param_groups[-1].items():
            if k == 'params':
                logger.info('   params: {}'.format(len(v)))
            else:
                logger.info('   {}: {}'.format(k, v))

    for ptype, pconf in config.items():
        logger.info('names for {}({}): {}'.format(
            ptype, len(names[ptype]), names[ptype]))

    return param_groups, type2num

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    index = index.to(torch.long)
    mask[index] = 1
    return mask

def gen_uniform_80_80_20_split(data):
    skf = StratifiedKFold(10, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return torch.cat(idx[:8], 0), torch.cat(idx[:8], 0), torch.cat(idx[8:], 0)

def gen_uniform_60_20_20_split(data):
    skf = StratifiedKFold(10, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return torch.cat(idx[:6], 0), torch.cat(idx[6:8], 0), torch.cat(idx[8:], 0)

def save_load_split(data, gen_splits):
    split = gen_splits(data)
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)
    return data


# Tox21 specific (splitting graphs instead of nodes)
def save_load_split_graph(data, gen_splits):
    split = gen_splits(data)
    data.train_mask = index_to_mask(split[0], len(data))
    data.val_mask = index_to_mask(split[1], len(data))
    data.test_mask = index_to_mask(split[2], len(data))
    return data

########################
# super set seed !!!
########################
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True