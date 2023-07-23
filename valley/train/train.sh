PORTS=$ARNOLD_WORKER_0_PORT 
PORT=(${PORTS//,/ })
torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST --master_port $PORT valley/train/train.py --conf $1