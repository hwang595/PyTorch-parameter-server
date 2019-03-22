NODE_RANK="$1"
NNODE="$2"
MASTER_IP="$3"
SRC_DIR=${HOME}/ps_real_pytorch/src

echo ${MASTER_IP}
sudo /home/ubuntu/anaconda3/bin/python -m torch.distributed.launch \
--nproc_per_node=1 \
--nnodes=${NNODE} --node_rank=${NODE_RANK} --master_addr="${MASTER_IP}" --master_port=1234 \
${SRC_DIR}/distributed_nn.py \
--lr=0.1 \
--momentum=0.9 \
--max-steps=100000 \
--epochs=100 \
--network=ResNet18 \
--dataset=Cifar10 \
--batch-size=64 \
--comm-type=Bcast \
--num-aggregate=2 \
--mode=normal \
--eval-freq=2000 \
--gather-type=gather \
--compress-grad=compress \
--enable-gpu= \
--train-dir=/home/ubuntu > out_node_${NODE_RANK} 2>&1