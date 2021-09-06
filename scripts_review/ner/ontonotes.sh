export CUDA_VISIBLE_DEVICES=1

DATA_DIR=/data/yangpan/workspace/dataset/ner
SVAE_DIR=/data/yangpan/workspace/nlp_with_label/checkpoint
PRETRAIN_MODEL=bert-base-cased
RESULT_DIR=/data/yangpan/workspace/nlp_with_label/results
DATA_NAME=ontonotes5.0
MODEL_TYPE=bert_ner
SAVE_NAME=ner/${DATA_NAME}/${PRETRAIN_MODEL}_${DATA_NAME}_${MODEL_TYPE}

python ../../run_ner.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --evaluate_during_training true \
  --span_decode_strategy 'v3' \
  --overwrite_output_dir true \
  --overwrite_cache false \
  --task_type ner \
  --num_train_epochs 40 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 512 \
  --learning_rate 1e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --val_step 3619 \
  --val_skip_step 1 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/label_map.json \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 42