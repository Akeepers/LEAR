export CUDA_VISIBLE_DEVICES=0

DATA_DIR=/data/yangpan/workspace/dataset/text_classification
SVAE_DIR=/data/yangpan/workspace/nlp_with_label/checkpoint
PRETRAIN_MODEL=bert-base-uncased
RESULT_DIR=/data/yangpan/workspace/nlp_with_label/results
DATA_NAME=imdb
MODEL_TYPE=text_classification
SAVE_NAME=text_classification/${DATA_NAME}/${PRETRAIN_MODEL}_${MODEL_TYPE}

python ../../run_classifier.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case true \
  --model_name ${MODEL_TYPE} \
  --evaluate_during_training true \
  --use_auxiliary_task false \
  --overwrite_output_dir true \
  --overwrite_cache false \
  --task_type imdb \
  --num_train_epochs 20 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --max_seq_length 512 \
  --learning_rate 1e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --val_step 1562 \
  --val_skip_step 1 \
  --first_label_num 33 \
  --second_label_num 35 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/train.csv \
  --dev_set ${DATA_DIR}/${DATA_NAME}/test.csv \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/label_map.json \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 42
