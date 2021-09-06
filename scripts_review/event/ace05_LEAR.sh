export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

DATA_DIR=/data/yangpan/workspace/dataset/event
SVAE_DIR=/data/yangpan/workspace/nlp_with_label/checkpoint
PRETRAIN_MODEL=roberta-base
LOG_DIR=/data/yangpan/workspace/SERS/logs
RESULT_DIR=/data/yangpan/workspace/SERS/results
PRETRAIN_MODEL_DIR=/data/yangpan/workspace/pretrain_models/pytorch
DATA_NAME=ace05
MODEL_TYPE=LEAR
SAVE_NAME=event/${DATA_NAME}/${PRETRAIN_MODEL}_${DATA_NAME}_${MODEL_TYPE}
# SAVE_NAME=event/${DATA_NAME}/${PRETRAIN_MODEL}_${DATA_NAME}_trigger_extraction_with_label_siamese_attn

python ../../run_trigger_extraction.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --use_auxiliary_task false \
  --evaluate_during_training true \
  --gradient_checkpointing true \
  --use_attn true \
  --data_tag LEAR \
  --dump_result false \
  --overwrite_output_dir true \
  --span_decode_strategy 'v5' \
  --overwrite_cache false \
  --task_type trigger \
  --num_train_epochs 30 \
  --gradient_accumulation_steps 16 \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 4 \
  --max_seq_length 128 \
  --learning_rate 1e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --eval_per_epoch 1 \
  --val_skip_epoch 0 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --vocab_file ${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL}/vocab.txt \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/trigger_label_map.json \
  --label_str_file ${DATA_DIR}/${DATA_NAME}/processed/trigger_annotation.txt \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 1
