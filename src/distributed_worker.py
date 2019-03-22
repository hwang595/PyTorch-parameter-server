from __future__ import print_function
import numpy as np

from nn_ops import NN_Trainer
from util import *

import torch
from torch.autograd import Variable
import torch.distributed as dist

import time
from datetime import datetime
import copy
import logging
from sys import getsizeof

STEP_START_ = 1
TAG_LIST_ = [i*30 for i in range(50000)]

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ModelBuffer(object):
    def __init__(self, network):
        """
        this class is used to save model weights received from parameter server
        current step for each layer of model will also be updated here to make sure
        the model is always up-to-date
        """
        super(ModelBuffer, self).__init__()
        self.recv_buf = []
        self.layer_cur_step = []
        self.layer_shape = []
        '''
        initialize space to receive model from parameter server
        '''
        # consider we don't want to update the param of `BatchNorm` layer right now
        # we temporirially deprecate the foregoing version and only update the model
        # parameters
        for param_idx, param in enumerate(network.parameters()):
            self.recv_buf.append(torch.zeros(param.size()))


class DistributedWorker(NN_Trainer):
    def __init__(self, **kwargs):
        super(NN_Trainer, self).__init__()

        self.cur_step = 0
        self.next_step = 0 # we will fetch this one from parameter server

        self.rank = kwargs['rank']
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self._max_steps = kwargs['max_steps']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self.kill_threshold = kwargs['kill_threshold']
        self._eval_batch_size = 100
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._compress_grad = kwargs['compress_grad']
        self._gather_type = kwargs['gather_type']
        self._device = kwargs['device']

        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def build_model(self, num_classes=10):
        self.network = build_model(self.network_config, num_classes)
        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf()

        self.network.to(self._device)

    def train(self, train_loader, test_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1

        # number of batches in one epoch
        iteration_last_step=0
        iter_start_time=0
        first = True

        logger.info("Worker {}: starting training".format(self.rank))
        # start the training process
        for num_epoch in range(self.max_epochs):
            for batch_idx, (train_image_batch, train_label_batch) in enumerate(train_loader):

                iter_start_time = time.time()
                # worker exit task
                if self.cur_step == self._max_steps:
                    break
                X_batch, y_batch = train_image_batch.to(self._device), train_label_batch.to(self._device)

                # bcast communication stage
                fetch_weight_start = time.time()
                self._fetch_weight()
                fetch_weight_dur = time.time() - fetch_weight_start

                comp_start = time.time()
                self._train_init()
                loss, logits = self._forward(X_batch, y_batch)
                loss.backward()
                comp_dur = time.time() - comp_start

                prec1, prec5 = accuracy(logits.detach(), train_label_batch.long(), topk=(1, 5))
                gather_start = time.time()
                self._send_grads()
                gather_dur = time.time() - gather_start


                log_format = 'Worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Time Cost: {:.4f}, FetchWeight: {:.4f}, Computation: {:.4f}, GatherTime: {:.4f}, Acc: {:.4f}'
                logger.info(log_format.format(self.rank,
                            self.cur_step, num_epoch, batch_idx * self.batch_size, len(train_loader.dataset), 
                            (100. * (batch_idx * self.batch_size) / len(train_loader.dataset)), loss.item(), 
                            time.time()-iter_start_time, fetch_weight_dur, comp_dur, gather_dur, prec1.numpy()[0]))

                if self.cur_step%self._eval_freq == 0:
                    self._save_model(file_path=self._generate_model_path())

    def init_recv_buf(self):
        self.model_recv_buf = ModelBuffer(self.network)

    def _train_init(self):
        self.network.train()
        self.optimizer.zero_grad()

    def _forward(self, X_batch, y_batch):
        logits = self.network(X_batch)
        return self.criterion(logits, y_batch), logits

    def _fetch_weight(self):
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            dist.broadcast(self.model_recv_buf.recv_buf[layer_idx], src=0)
        self.model_update(self.model_recv_buf.recv_buf)
        # Note that at here we update the global step
        self.cur_step += 1
    
    def update_step(self):
        '''update local (global) step on worker'''
        changed = (self.cur_step != self.next_step)
        self.cur_step = self.next_step
        return changed

    def model_update(self, weights_to_update):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_counter_ = 0
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            # handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
            if "running_mean" in key_name or "running_var" in key_name or "num_batches_tracked" in key_name:
                tmp_dict={key_name: param}
            else:
                assert param.size() == weights_to_update[model_counter_].size()
                tmp_dict = {key_name: weights_to_update[model_counter_].to(self._device)}
                model_counter_ += 1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    def _send_grads(self):
        for p_index, p in enumerate(self.network.parameters()):
            # fetch the grad we need
            if self._device.type == "cuda":
                grad = p.grad.to(torch.device("cpu")).detach()
            else:
                grad = p.grad.detach()

            dist.gather(grad, [], dst=0)

    def _evaluate_model(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        for data, y_batch in test_loader:
            data, target = data.to(self._device), y_batch.to(self._device)
            
            output = self.network(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).item() # sum up batch loss

            prec1_tmp, prec5_tmp = accuracy(output.detach(), y_batch, topk=(1, 5))

            if self._device.type == 'cuda':
                prec1_counter_ += prec1_tmp.to(torch.device("cpu")).numpy()[0]
                prec5_counter_ += prec5_tmp.to(torch.device("cpu")).numpy()[0]
            else:
                prec1_counter_ += prec1_tmp.numpy()[0]
                prec5_counter_ += prec5_tmp.numpy()[0]

            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        test_loss /= len(test_loader.dataset)
        print('Test set: Step: {}, Average loss: {:.4f}, Prec@1: {} Prec@5: {}'.format(self.cur_step, 
                                                                            test_loss, prec1, prec5))

    def _generate_model_path(self):
        return self._train_dir+"model_step_"+str(self.cur_step)

    def _save_model(self, file_path):
        with open(file_path, "wb") as f_:
            torch.save(self.network.state_dict(), f_)
        return

if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    worker_fc_nn = WorkerFC_NN(comm=comm, world_size=world_size, rank=rank)
    print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))