export CUDA_VISIBLE_DEVICES=0

DATA_DIR=/data/yangpan/workspace/dataset/event
SVAE_DIR=/data/yangpan/workspace/nlp_with_label/checkpoint
PRETRAIN_MODEL=bert-base-cased
LOG_DIR=/data/yangpan/workspace/nlp_with_label/logs
RESULT_DIR=/data/yangpan/workspace/nlp_with_label/results
DATA_NAME=ace05
MODEL_TYPE=trigger_extraction_with_label_siamese
SAVE_NAME=event/${DATA_NAME}/${PRETRAIN_MODEL}_${DATA_NAME}_${MODEL_TYPE}_gnn_1

python ../../run_trigger_extraction.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --use_auxiliary_task false \
  --evaluate_during_training true \
  --use_gnn true \
  --time_steps 3 \
  --gnn_file ${DATA_DIR}/${DATA_NAME}/processed/gnn_file.npy \
  --overwrite_output_dir true \
  --overwrite_cache false \
  --task_type trigger \
  --num_train_epochs 30 \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 256 \
  --learning_rate 1e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --val_step 1739 \
  --val_skip_step 1 \
  --first_label_num 33 \
  --second_label_num 35 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/trigger_label_map.json \
  --label_emb_size 768 \
  --label_str_file ${DATA_DIR}/${DATA_NAME}/processed/trigger_annotation.txt \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 1
