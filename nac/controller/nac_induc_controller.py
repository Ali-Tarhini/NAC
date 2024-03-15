import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
import pdb

class NACInductiveController(object):

    def __init__(self, config):
        self.config = config
        self.build_sane_settings()

    def build_sane_settings(self):
        self.network_momentum = self.config.momentum
        self.network_weight_decay = self.config.weight_decay
        self.unrolled = self.config.unrolled

        # subnet
        self.subnet = self.config.get('subnet', None)

    # init model in solver
    def set_supernet(self, model):
        self.model = model

    # init logger in solver
    def set_logger(self, logger):
        self.logger = logger

    # init criterion in solver
    def set_criterion(self, criterion):
        self.criterion = criterion

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=self.config.arch_learning_rate, betas=(0.5, 0.999), weight_decay=self.config.arch_weight_decay)

    def step(self, data):
        if self.subnet != None:
            return

        self.optimizer.zero_grad()
        # changed into tox21
        self._backward_step_tox21(data)
        # self.optimizer.step()

    def _backward_step(self, data):
        inp, target = data, Variable(data.y[data.val_mask], requires_grad=False) # why use val_mask here instead of train (in the case of the default 80-80-20 split they are the same)
        logit = self.model(inp)
        loss = self.criterion(logit[data.val_mask], target)

        if getattr(self.config, 'sparse', False):
            p = getattr(self.config.sparse, 'norm', 1)
            _lambda = getattr(self.config.sparse, 'lambda', 0.001)
            self.logger.info(f'original loss = {loss}')
            if getattr(self.config.sparse, 'na_sparse', True):
                na_reg_loss = _lambda * torch.norm(self.model.na_weights, p=p)
                self.logger.info(f'na sparse loss = {na_reg_loss}')
                loss += na_reg_loss
            if getattr(self.config.sparse, 'sc_sparse', True):
                sc_reg_loss = _lambda * torch.norm(self.model.sc_weights, p=p)
                self.logger.info(f'sc sparse loss = {sc_reg_loss}')
                loss += sc_reg_loss
            if getattr(self.config.sparse, 'la_sparse', True):
                la_reg_loss = _lambda * torch.norm(self.model.la_weights, p=p)
                self.logger.info(f'la sparse loss = {la_reg_loss}')
                loss += la_reg_loss
        loss.backward()


    
    def _backward_step_tox21(self, data):
        # inp, target = data, Variable(data.y[data.val_mask], requires_grad=False) # why use val_mask here instead of train (in the case of the default 80-80-20 split they are the same)
        # logit = self.model(inp)
        # loss = self.criterion(logit[data.val_mask], target)
        
        # get_data
        inp = data[data.val_mask]
        train_loader = DataLoader(inp, batch_size=128, shuffle=True)
        
        original_loss = []
        na_sparse_loss = []
        sc_sparse_loss = []
        la_sparse_loss = []
        
        for batch in train_loader:
            logit = self.model(batch)
            target = torch.tensor(batch.y, requires_grad=False)
            loss = self.criterion(logit, target.unsqueeze(-1))

            if getattr(self.config, 'sparse', False):
                p = getattr(self.config.sparse, 'norm', 1)
                _lambda = getattr(self.config.sparse, 'lambda', 0.001)
                # self.logger.info(f'original loss = {loss}')
                original_loss.append(loss.detach().numpy()) #
                if getattr(self.config.sparse, 'na_sparse', True):
                    na_reg_loss = _lambda * torch.norm(self.model.na_weights, p=p)
                    # self.logger.info(f'na sparse loss = {na_reg_loss}')
                    na_sparse_loss.append(na_reg_loss.detach().numpy()) #
                    loss += na_reg_loss
                if getattr(self.config.sparse, 'sc_sparse', True):
                    sc_reg_loss = _lambda * torch.norm(self.model.sc_weights, p=p)
                    # self.logger.info(f'sc sparse loss = {sc_reg_loss}')
                    loss += sc_reg_loss
                    sc_sparse_loss.append(sc_reg_loss.detach().numpy()) #
                if getattr(self.config.sparse, 'la_sparse', True):
                    la_reg_loss = _lambda * torch.norm(self.model.la_weights, p=p)
                    # self.logger.info(f'la sparse loss = {la_reg_loss}')
                    loss += la_reg_loss
                    la_sparse_loss.append(la_reg_loss.detach().numpy()) #
            loss.backward()
            self.optimizer.step()
            
        if getattr(self.config, 'sparse', False):
            self.logger.info(f'original loss = {np.average(original_loss)}') 
            if getattr(self.config.sparse, 'na_sparse', True):
                self.logger.info(f'na sparse loss = {np.average(na_sparse_loss)}')
            if getattr(self.config.sparse, 'sc_sparse', True):
                self.logger.info(f'sc sparse loss = {np.average(sc_sparse_loss)}') 
            if getattr(self.config.sparse, 'la_sparse', True):    
                self.logger.info(f'la sparse loss = {np.average(la_sparse_loss)}') 


    def build_active_subnet(self, subnet_settings):
        return self.model.build_active_subnet(subnet_settings)