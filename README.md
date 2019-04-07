# PyTorch-parameter-server
Implementation of synchronous distributed machine learning in [Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_nips14.pdf) setup using [PyTorch's distributed communication library](https://pytorch.org/docs/stable/distributed.html) i.e. `torch.distributed`.

All functionality in this repository is basically a repliaction of [ps_pytorch](https://github.com/hwang595/ps_pytorch). However, instead of using `Mpi4py`, all communications and model trainings are handled by PyTorch itself.

## Contents

1. [Motivations](#motivations)
2. [System design](#system-design)
3. [Basic usages](#basic-usages)
4. [How to prepare datasets](#prepare-datasets)
5. [How to launch a distributed task](#job-launching)
6. [Future work](#future-work)

## Motivations:
1. PyTorch provides easy-to-use APIs with dynamic computational graph
2. Altough [mpi4py](https://github.com/mpi4py/mpi4py) provides a good Python binding for any distributions of MPI and flexible communication operations, transforming data back and force (e.g. `torch.Tensor` <--> `numpy.array`) incurs heavy overheads during the entire training process.
3. PyTorch supports [NCCL](https://developer.nvidia.com/nccl) as its communication backend, which makes distributed training on GPU cluster becomes efficient and scalable.

## System Design:
1. Parameter Server: This node synchronizes all workers to enter next iteration by broadcast global step to workers and stores the global model, which will be pulled by workers at beginning of one iteration (we implement this stage using `torch.distributed.broadcast`). For a user defined frequency, Parameter Server will save the current model as checkpoint to shared file system (NFS in our system) for model evaluation.
2. workers mainly aim at sample data points (or mini-batch) from local dataset (we don't pass data among nodes to maintain data locality), computing gradients, and ship them back to Parameter Server (this stage is implemented using `torch.distributed.scatter`).
3. evaluator read the checkpoints from the shared directory, and do model evaluation. Note that: there is only testset data saved on evaluator nodes.

<div align="center"><img src="https://github.com/hwang595/PyTorch-parameter-server/blob/master/images/system_overview.jpg" height="400" width="600" ></div>

## Basic Usages
### Dependencies:
Anaconda is highly recommended for installing depdencies for this project. Assume a conda setup machine is used, you can run 
```
bash ./tools/pre_run.sh
```
to install all depdencies needed. 
### Single Machine:
The code base provided in this repository can be run on a single machine, in which multiple CPU processes will be launched and each process will be assigned a role as Parameter Server (usually process with id at 0) or worker. To do this, one can just follow the "Single-Node multi-process distributed training" part in [this tutorial](https://pytorch.org/docs/stable/distributed.html#launch-utility). We provide a script (`run_pytorch_single.sh`) to do the job for you. One can simply run
```
bash ./src/run_pytorch_single.sh
```

### Cluster Setup:
For running on distributed cluster, the first thing you need do is to launch AWS EC2 instances.
#### Launching Instances:
[This script](https://github.com/hwang595/PyTorch-parameter-server/blob/master/tools/pytorch_ec2.py) helps you to launch EC2 instances automatically, but before running this script, you should follow [the instruction](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) to setup AWS CLI on your local machine.
After that, please edit this part in `./tools/pytorch_ec2.py`
``` python
cfg = Cfg({
    "name" : "PS_PYTORCH",      # Unique name for this specific configuration
    "key_name": "NameOfKeyFile",          # Necessary to ssh into created instances
    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 8,
    "num_replicas_to_aggregate" : "8", # deprecated, not necessary
    "method" : "spot",
    # Region speficiation
    "region" : "us-west-2",
    "availability_zone" : "us-west-2b",
    # Machine type - instance type configuration.
    "master_type" : "m4.2xlarge",
    "worker_type" : "m4.2xlarge",
    # please only use this AMI for pytorch
    "image_id": "ami-xxxxxxxx",            # id of AMI
    # Launch specifications
    "spot_price" : "0.15",                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "/dir/to/NameOfKeyFile.pem",

    # NFS configuration
    # To set up these values, go to Services > ElasticFileSystem > Create new filesystem, and follow the directions.
    #"nfs_ip_address" : "172.31.3.173",         # us-west-2c
    #"nfs_ip_address" : "172.31.35.0",          # us-west-2a
    "nfs_ip_address" : "172.31.14.225",          # us-west-2b
    "nfs_mount_point" : "/home/ubuntu/shared",       # NFS base dir
```
For setting everything up on EC2 cluster, the easiest way is to setup one machine and create an AMI. Then use the AMI id for `image_id` in `pytorch_ec2.py`. Then, launch EC2 instances by running
```
python ./tools/pytorch_ec2.py launch
```
After all launched instances are ready (this may take a while), getting private ips of instances by
```
python ./tools/pytorch_ec2.py get_hosts
```
this will write ips into a file named `hosts_address`, which looks like
```
172.31.16.226 (${PS_IP})
172.31.27.245
172.31.29.131
172.31.18.108
...
```
After generating the `hosts_address` of all EC2 instances, running the following command will copy your keyfile to the parameter server (PS) instance whose address is always the first one in `hosts_address`. `local_script.sh` will also do some basic configurations e.g. clone this git repo
```
bash ./tool/local_script.sh ${PS_IP}
```
#### SSH related:
At this stage, you should ssh to the PS instance and all operation should happen on PS. In PS setting, PS should be able to ssh to any compute node, [this part](https://github.com/hwang595/PyTorch-parameter-server/blob/master/tools/remote_script.sh#L8-L22) dose the job for you by running (after ssh to the PS)
```
bash ./tools/remote_script.sh
```

## Prepare Datasets
To download, split, and transform datasets by (and `./tools/remote_script.sh` dose this for you)
```
bash ./src/data_prepare.sh
```
One can simply extend script `./src/data/data_prepare.py` to support any datasets provided by [torchvision](https://github.com/pytorch/vision).

## Job Launching
Since this project is built on MPI, tasks are required to be launched by PS (or master) instance. `launch.sh` (which will call `./src/run_pytorch_dist.sh`) wraps job-launching process up. Commonly used options (arguments) are listed as following:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `n`                     | Number of processes (size of cluster) e.g. if we have P compute node and 1 PS, n=P+1. |
| `lr`                        | Inital learning rate that will be use. |
| `momentum`                  | Value of momentum that will be use. |
| `max-steps`                       | The maximum number of iterations to train. |
| `epochs`                  | The maximal number of epochs to train (somehow redundant).   |
| `network`                  | Types of deep neural nets, currently `LeNet`, `ResNet-18/32/50/110/152`, and `VGGs` are supported. |
| `dataset` | Datasets use for training. |
| `batch-size` | Batch size for optimization algorithms. |
| `eval-freq` | Frequency of iterations to evaluation the model. |
| `enable-gpu`|Training on CPU/GPU, if CPU please leave this argument empty. |
|`train-dir`|Directory to save model checkpoints for evaluation. |

## Model Evaluation
[Distributed evaluator](https://github.com/hwang595/PyTorch-parameter-server/blob/master/src/distributed_evaluator.py) will fetch model checkpoints from the shared directory and evaluate model on validation set.
To evaluate model, you can run
```
bash ./src/evaluate_pytorch.sh
```
with specified arguments.

Evaluation arguments are listed as following:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `eval-batch-size`             | Batch size (on validation set) used during model evaluation. |
| `eval-freq`      | Frequency of iterations to evaluation the model, should be set to the same value as [run_pytorch_dist.sh](https://github.com/hwang595/ps_pytorch/blob/master/src/run_pytorch.sh). |
| `network`                        | Types of deep neural nets, should be set to the same value as [run_pytorch_dist.sh](https://github.com/hwang595/PyTorch-parameter-server/blob/master/src/run_pytorch_dist.sh). |
| `dataset`                  | Datasets use for training, should be set to the same value as [run_pytorch_dist.sh](https://github.com/hwang595/PyTorch-parameter-server/blob/master/src/run_pytorch_dist.sh). |
| `model-dir`                       | Directory to save model checkpoints for evaluation, should be set to the same value as [run_pytorch_dist.sh](https://github.com/hwang595/PyTorch-parameter-server/blob/master/src/run_pytorch_dist.sh). |

## Future work:
(Please note that this project is still in early alpha version)
1. Overlapping computation (forward prop and backprop) with communication to gain better speedup. 
2. Support async communication mode i.e. [Backup Worker](https://arxiv.org/pdf/1604.00981.pdf)

## Contact:
Any contribution to this repo is highly appreciated. 
If you encountered any issue in using the code base provided here, please feel free to start an issue or email [Hongyi Wang](https://hwang595.github.io/) at (hongyiwang@cs.wisc.edu) directly.
