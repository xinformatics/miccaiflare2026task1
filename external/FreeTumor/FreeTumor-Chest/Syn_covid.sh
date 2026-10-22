now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_syn_covid
data=covid
baseline_seg_dir=./baseline/model_covid_voco160k.pt
TGAN_checkpoint=./runs/logs_syn_covid/model_final.pt
mkdir -p $logdir

torchrun -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=23020 \
    Syn_train.py \
    --data $data \
    --baseline_seg_dir $baseline_seg_dir \
    --TGAN_checkpoint $TGAN_checkpoint \
    --logdir $logdir | tee $logdir/$now.txt