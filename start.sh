#!/bin/bash
#SBATCH -vv
#SBATCH -t 47:00:00
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -J mfa
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sdixit1
#SBATCH --signal INT@60

export MASTER_PORT='12334'
export MASTER_ADDR='localhost' 

encoder_name="conformer_cat" # conformer_cat | ecapa_tdnn_large | resnet34
embedding_dim=192
loss_name='amsoftmax'

dataset="vox"
num_classes=1211 # 1211 # 7205
num_blocks=6
train_csv_path="data/train.csv"

input_layer=conv2d2
pos_enc_layer_type=rel_pos # no_pos| rel_pos 
save_dir=experiment/vox1_${loss_name}
trial_path=data/vox1_test.txt

mkdir -p $save_dir
cp start.sh $save_dir
cp main.py $save_dir
cp -r module $save_dir
cp -r wenet $save_dir
cp -r scripts $save_dir
cp -r loss $save_dir
echo save_dir: $save_dir

export CUDA_VISIBLE_DEVICES=0
python3 main.py \
        --batch_size 200 \
        --num_workers 40 \
        --max_epochs 30 \
        --embedding_dim $embedding_dim \
        --save_dir $save_dir \
        --encoder_name $encoder_name \
        --train_csv_path $train_csv_path \
        --learning_rate 0.001 \
        --encoder_name ${encoder_name} \
        --num_classes $num_classes \
        --trial_path $trial_path \
        --loss_name $loss_name \
        --num_blocks $num_blocks \
        --step_size 4 \
        --gamma 0.5 \
        --weight_decay 0.0000001 \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \

# 
