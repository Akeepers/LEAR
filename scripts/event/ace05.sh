export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

DATA_DIR=/data/yangpan/workspace/dataset/event
SVAE_DIR=/data/yangpan/workspace/SERS/checkpoint
PRETRAIN_MODEL=bert-base-cased
LOG_DIR=/data/yangpan/workspace/SERS/logs
RESULT_DIR=/data/yangpan/workspace/SERS/results
PRETRAIN_MODEL_DIR=/data/yangpan/workspace/pretrain_models/pytorch
DATA_NAME=ace05
MODEL_TYPE=bert_span_ed
SAVE_NAME=event/${DATA_NAME}/${PRETRAIN_MODEL}_${DATA_NAME}_${MODEL_TYPE}_v5

python ../../run_trigger_extraction.py \
  --use_cuda true \
  --do_train false \
  --do_eval true \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --dump_result false \
  --overwrite_cache false \
  --data_tag new \
  --task_type trigger \
  --span_decode_strategy 'v5' \
  --num_train_epochs 30 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 256 \
  --learning_rate 1e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --eval_per_epoch 1 \
  --val_skip_step 1 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --vocab_file ${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL}/vocab.txt \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/trigger_label_map.json \
  --second_label_file ${DATA_DIR}/${DATA_NAME}/processed/argument_label_map.json \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 1
