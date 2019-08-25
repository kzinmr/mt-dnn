#!/bin/bash
if [[ $# -ne 3 ]]; then
  echo "run_ner.sh <batch_size> <gpu> <model_ckpt>"
  exit 1
fi
prefix="mt-dnn-nerall"
BATCH_SIZE=$1
gpu=$2
model_ckpt=$3  # model_ckpt="${model_dir}/model_${model_num}.pt"
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="nerall"
test_datasets="nerall"
MODEL_ROOT="checkpoints"

BERT_PATH="mt_dnn_models/Japanese_L-12_H-768_A-12_E-30_BPE/"
BERT_CONFIG_PATH="mt_dnn_models/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json"
BERT_VOCAB_PATH="mt_dnn_models/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt"
DATA_DIR="data/bccwj_all_class"
TASK_DEF_PATH="experiments/japanese/japanese_task_def.yml"

optim="adamax"
grad_clipping=0
global_grad_clipping=1
# lr="5e-5"
# epochs=5

model_dir="checkpoints/${prefix}_${optim}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python evaluate.py --data_dir ${DATA_DIR} --bert_config_path ${BERT_CONFIG_PATH} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --multi_gpu_on --task_def ${TASK_DEF_PATH} --model_ckpt ${model_ckpt} --bert_vocab ${BERT_VOCAB_PATH}
