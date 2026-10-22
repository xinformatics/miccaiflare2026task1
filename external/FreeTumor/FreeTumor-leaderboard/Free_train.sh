now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_free_all
data=all
task=freesyn
baseline_seg_dir=./baseline/model_baseline_segmentor.pt
TGAN_checkpoint=./baseline/TGAN.pt

mkdir -p $logdir

export CUDA_VISIBLE_DEVICES=0
torchrun --master_port=25023 Free_train.py \
    --data $data \
    --task $task \
    --baseline_seg_dir $baseline_seg_dir \
    --TGAN_checkpoint $TGAN_checkpoint \
    --logdir $logdir | tee $logdir/$now.txt