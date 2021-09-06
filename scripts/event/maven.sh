export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

DATA_DIR=/data/yangpan/workspace/dataset/event
SVAE_DIR=/data/yangpan/workspace/research/SERS/checkpoint
PRETRAIN_MODEL=roberta-large
LOG_DIR=/data/yangpan/workspace/research/SERS/logs
RESULT_DIR=/data/yangpan/workspace/research/SERS/results
DATA_NAME=maven
MODEL_TYPE=bert_span_ed
SAVE_NAME=event/${DATA_NAME}/${PRETRAIN_MODEL}_${DATA_NAME}_${MODEL_TYPE}_1

python ../../run_trigger_extraction.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --eval_test false \
  --data_type maven \
  --dump_result true \
  --overwrite_cache false \
  --span_decode_strategy 'v5_fast' \
  --match_pattern "defalut" \
  --task_type trigger \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 512 \
  --max_seq_length 512 \
  --learning_rate 2e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --eval_per_epoch 1 \
  --val_skip_step 0 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/dev4test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/trigger_label_map.json \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 1
