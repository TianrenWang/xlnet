#!/bin/bash

#SBATCH --account=def-inkpen
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=28
#SBATCH --mem=150G
#SBATCH --time=0-01:00
#SBATCH --output=output.out


#### local path
RACE_DIR=data/RACE
INIT_CKPT_DIR=xlnet_cased_L-12_H-768_A-12

#### google storage path
GS_ROOT=${PWD}
GS_INIT_CKPT_DIR=${GS_ROOT}/${INIT_CKPT_DIR}
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/race
GS_MODEL_DIR=${GS_ROOT}/experiment/race

# TPU name in google cloud
TPU_NAME=

python3 run_race_macnetwork.py \
  --use_tpu=False \
  --use_mac=True \
  --tpu=${TPU_NAME} \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --model_config_path=${GS_INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${GS_INIT_CKPT_DIR}/spiece.model \
  --output_dir=${GS_PROC_DATA_DIR} \
  --init_checkpoint=${GS_INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${GS_MODEL_DIR} \
  --data_dir=${GS_ROOT}/${RACE_DIR} \
  --max_seq_length=640 \
  --max_qa_length=128 \
  --uncased=False \
  --do_train=True \
  --train_batch_size=8 \
  --do_eval=True \
  --eval_batch_size=32 \
  --train_steps=12000 \
  --save_steps=1000 \
  --iterations=1000 \
  --warmup_steps=1000 \
  --learning_rate=1e-6 \
  --weight_decay=0 \
  --adam_epsilon=1e-6 \
  $@