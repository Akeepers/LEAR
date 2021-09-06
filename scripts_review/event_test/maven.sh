export CUDA_VISIBLE_DEVICES=0

DATA_DIR=/data/yangpan/workspace/dataset/event
SVAE_DIR=/data/yangpan/workspace/nlp_with_label/checkpoint
PRETRAIN_MODEL=bert-base-cased
LOG_DIR=/data/yangpan/workspace/nlp_with_label/logs
RESULT_DIR=/data/yangpan/workspace/nlp_with_label/results
PRETRAIN_MODEL_DIR=/data/yangpan/workspace/pretrain_models/pytorch
DATA_NAME=maven
MODEL_TYPE=trigger_extraction
SAVE_NAME=event/${DATA_NAME}/${PRETRAIN_MODEL}_${DATA_NAME}_${MODEL_TYPE}_1

python ../../run_trigger_extraction.py \
  --use_cuda true \
  --do_train false \
  --do_eval true \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --eval_test false \
  --dump_result true \
  --is_maven true \
  --overwrite_cache false \
  --task_type trigger \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 2 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 512 \
  --learning_rate 1e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --val_step 2026 \
  --val_skip_step 1 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --vocab_file ${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL}/vocab.txt \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/trigger_label_map.json \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 1
