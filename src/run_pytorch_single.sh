python -m torch.distributed.launch \
--nproc_per_node=3 \
distributed_nn.py \
--lr=0.01 \
--momentum=0.9 \
--max-steps=100000 \
--epochs=100 \
--network=LeNet \
--dataset=MNIST \
--batch-size=128 \
--comm-type=Bcast \
--num-aggregate=5 \
--mode=normal \
--eval-freq=20 \
--gather-type=gather \
--compress-grad=compress \
--enable-gpu= \
--train-dir=/home/ubuntu