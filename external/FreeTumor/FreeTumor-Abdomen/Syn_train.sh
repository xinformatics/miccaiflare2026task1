now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_syn
baseline_seg_dir=./baseline/model_baseline_segmentor.pt
TGAN_checkpoint=./baseline/TGAN.pt
mkdir -p $logdir

torchrun -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=23024 \
    Syn_train.py \
    --baseline_seg_dir $baseline_seg_dir \
    --TGAN_checkpoint $TGAN_checkpoint \
    --logdir $logdir | tee $logdir/$now.txt