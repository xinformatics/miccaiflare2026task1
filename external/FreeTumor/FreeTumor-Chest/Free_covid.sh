now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_free_covid
data=covid
task=freesyn
baseline_seg_dir=./baseline/model_covid_voco160k.pt
TGAN_checkpoint=./runs/logs_syn_covid/model_final.pt

mkdir -p $logdir

export CUDA_VISIBLE_DEVICES=0
torchrun --master_port=25024 Free_train.py \
    --data $data \
    --task $task \
    --use_ssl_pretrained \
    --baseline_seg_dir $baseline_seg_dir \
    --TGAN_checkpoint $TGAN_checkpoint \
    --logdir $logdir | tee $logdir/$now.txt
