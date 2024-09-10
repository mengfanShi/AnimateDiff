# 设置环境变量
export RANK=0
export LOCAL_RANK=$RANK
export WORLD_SIZE=1

# 启动训练脚本
torchrun --nproc_per_node=1 train.py --config configs/training/v3/training_lora.yaml --wandb