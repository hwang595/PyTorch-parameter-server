KEY_PEM_NAME=HongyiScript.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`
MASTER_PUB_IP="$1"
WORKING_DIR=${HOME}/ps_real_pytorch/src

for i in $(seq 1 $DEEPLEARNING_WORKERS_COUNT);
  do
    ssh -i ${HOME}/.ssh/${KEY_PEM_NAME} deeplearning-worker${i} "cd ${WORKING_DIR}; nohup bash ${WORKING_DIR}/run_pytorch_dist.sh \"$((${i}-1))\" \"${DEEPLEARNING_WORKERS_COUNT}\" \"${MASTER_PUB_IP}\" &>/dev/null &"
  done