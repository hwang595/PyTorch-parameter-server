from __future__ import print_function
import time
import copy
from sys import getsizeof
import logging
from functools import reduce

import numpy as np

from nn_ops import NN_Trainer
from util import *
from optim.adam import Adam
from optim.sgd import SGD

import torch
import torch.distributed as dist

STEP_START_ = 1

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def update_params_dist_version(param, avg_grad, learning_rate):
    '''
    update the network layer by layer
    '''
    assert param.shape == avg_grad.shape
    param -= learning_rate * avg_grad
    return param


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


class GradientAccumulator(object):
    '''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
    def __init__(self, module, world_size, mode='None'):
        super(GradientAccumulator, self).__init__()
        # we will update this counter dynamically during the training process
        # the length of this counter should be number of fc layers in the network
        # we used list to contain gradients of layers
        self.gradient_aggregate_counter = []
        self.model_index_range = []
        self.gradient_aggregator = []
        self._mode = mode
        
        for param_idx, param in enumerate(module.parameters()):
            tmp_aggregator = []
            for worker_idx in range(world_size):
                tmp_aggregator.append(torch.zeros(param.size()))
            # initialize the gradient aggragator
            self.gradient_aggregator.append(tmp_aggregator)
            self.gradient_aggregate_counter.append(0)
            self.model_index_range.append(param_idx)

    def meset_everything(self):
        self._meset_grad_counter()
        self._meset_grad_aggregator()

    def _meset_grad_counter(self):
        self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

    def _meset_grad_aggregator(self):
        '''
        reset the buffers in grad accumulator, not sure if this is necessary
        '''
        if self._mode == 'compress':
            pass
        else:
            for i, tmp_aggregator in enumerate(self.gradient_aggregator):
                for j, buf in enumerate(tmp_aggregator):
                    self.gradient_aggregator[i][j] = np.zeros(self.gradient_aggregator[i][j].shape)


class SyncReplicasMaster_NN(NN_Trainer):
    def __init__(self, **kwargs):
        super(NN_Trainer, self).__init__()
        '''master node here, no rank needed since the rank will always be 0 for master node'''
        self.world_size = kwargs['world_size']
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._timeout_threshold = kwargs['timeout_threshold']

        self._num_workers = self.world_size - 1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._expected_grad_to_recv = kwargs['kill_threshold']
        self._max_steps = kwargs['max_steps']
        self._compress_grad = kwargs['compress_grad']
        self._gather_type = kwargs['gather_type']
        self._device = kwargs['device']

        ############ will be deprecated soon #############################
        self._eval_batch_size = 1000

    def build_model(self, num_classes=10):
        self.network = build_model(self.network_config, num_classes)
        self.optimizer = SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size, self._compress_grad)
        self.init_model_shapes()
        #self.network.to(self._device)
        self.network.to(torch.device("cpu"))

    def start(self):
        for i in range(1, self._max_steps+1):
            # switch back to training mode
            self.network.train()
            self._first_grad_received = False
            enough_gradients_received = False

            logger.info("Master node is entering step: {}".format(i))

            self._bcast_weight()

            self._recv_grads()

            self._model_update()

            self.cur_step += 1

    def init_model_shapes(self):
        for param_idx, param in enumerate(self.network.parameters()):
            self._model_shapes.append(param.size())
            self._grad_aggregate_buffer.append(np.zeros(param.size()))

    def _model_update(self):
        # gradient shipped from workers are averaged and update the model
        self._grad_aggregate_buffer = [x / self._num_workers for x in self._grad_aggregate_buffer]
        self.optimizer.step(grads=self._grad_aggregate_buffer)        

    def _bcast_weight(self):
        for layer_idx, layer in enumerate(self.network.parameters()):
            layer_weight = layer.detach()
            dist.broadcast(layer_weight, src=0)

    def aggregate_gradient(self, layer_idx, gradient):
        self._grad_aggregate_buffer[layer_idx] = reduce((lambda x, y: x + y), gradient[1:])

    def _recv_grads(self):
        for layer_idx, layer in enumerate(self.network.parameters()):
            dummpy_grad = self.grad_accumulator.gradient_aggregator[layer_idx][0]
            dist.gather(dummpy_grad, self.grad_accumulator.gradient_aggregator[layer_idx], dst=0)
            self.aggregate_gradient(layer_idx=layer_idx, gradient=self.grad_accumulator.gradient_aggregator[layer_idx])

    def _generate_model_path(self):
        return self._train_dir+"model_step_"+str(self.cur_step)

    def _save_model(self, file_path):
        with open(file_path, "wb") as f_:
            torch.save(self.network.state_dict(), f_)
        return

    def _evaluate_model(self, validation_loader):
        self.network.eval()
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        # which indicate an epoch based validation is done
        while validation_loader.dataset.epochs_completed <= self._epoch_counter:
            eval_image_batch, eval_label_batch = validation_loader.next_batch(batch_size=self._eval_batch_size)
            X_batch, y_batch = Variable(eval_image_batch.float()), Variable(eval_label_batch.long())
            output = self.network(X_batch)
            prec1_tmp, prec5_tmp = accuracy(output.detach(), eval_label_batch.long(), topk=(1, 5))
            prec1_counter_ += prec1_tmp
            prec5_counter_ += prec5_tmp
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        self._epoch_counter = validation_loader.dataset.epochs_completed
        logger.info('Testset Performance: Cur Step:{} Prec@1: {} Prec@5: {}'.format(self.cur_step, prec1.numpy()[0], prec5.numpy()[0]))