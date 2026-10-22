now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_free_lits
data=lits
task=freesyn
max_epochs=20
val_every=1
baseline_seg_dir=./baseline/model_baseline_segmentor.pt
TGAN_checkpoint=./baseline/TGAN.pt

mkdir -p $logdir

export CUDA_VISIBLE_DEVICES=0
torchrun --master_port=25024 Free_train.py \
    --data $data \
    --task $task \
    --max_epochs $max_epochs \
    --val_every $val_every \
    --use_ssl_pretrained \
    --baseline_seg_dir $baseline_seg_dir \
    --TGAN_checkpoint $TGAN_checkpoint \
    --logdir $logdir | tee $logdir/$now.txt
