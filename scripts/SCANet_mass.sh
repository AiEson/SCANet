#!/bin/bash
dataset='mass'
method='train'
data_root=/path/to/mass/dataset
decoder_name='unetplusplus'
encoder_name='scanet-101e'
notes='bce_dice'

now=$(date +"%Y%m%d_%H%M%S")
exp=${decoder_name}'_'${encoder_name}'_'${notes}

config=configs/$dataset.yaml
train_id_path=$data_root/train.txt
test_id_path=$data_root/test.txt
val_id_path=$data_root/val.txt
save_path=exp/$dataset/$method/$exp

mkdir -p $save_path

cp ${method}.py $save_path
cp datasets/${dataset}.py $save_path

python ${method}.py --exp_name=$exp \
    --config=$config \
    --dataset=$dataset \
    --train-id-path $train_id_path \
    --test-id-path $test_id_path \
    --val_id_path $val_id_path \
    --save-path $save_path \
    --encoder_name $encoder_name \
    --decoder_name $decoder_name \
    $(if [ -n "$exchanged" ]; then echo "--data_root=$data_root"; fi) \
    --port $2 2>&1 | tee $save_path/$now.log