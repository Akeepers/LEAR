export CUDA_VISIBLE_DEVICES=1

DATA_DIR=/data/yangpan/workspace/dataset/ner
SVAE_DIR=/data/yangpan/workspace/nlp_with_label/checkpoint
PRETRAIN_MODEL=bert-base-chinese
RESULT_DIR=/data/yangpan/workspace/nlp_with_label/results
PRETRAIN_MODEL_DIR=/data/yangpan/workspace/pretrain_models/pytorch
DATA_NAME=zh_onto4
MODEL_TYPE=bert_ner_with_label_siamese
SAVE_NAME=ner/${DATA_NAME}/${PRETRAIN_MODEL}_${MODEL_TYPE}_attn_test

python ../../run_ner.py \
  --use_cuda true \
  --do_train true \
  --do_eval false \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --use_auxiliary_task false \
  --span_decode_strategy 'v5' \
  --use_attn true \
  --is_chinese true \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --overwrite_cache false \
  --task_type ner \
  --num_train_epochs 5 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 128 \
  --learning_rate 8e-6 \
  --task_layer_lr 10 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path /${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL} \
  --eval_per_epoch 200 \
  --val_skip_step 1 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/label_map.json \
  --vocab_file ${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL}/vocab.txt \
  --label_str_file ${DATA_DIR}/${DATA_NAME}/processed/label_annotation.txt \
  --glove_label_emb_file ${DATA_DIR}/${DATA_NAME}/processed/bert_base_chinese_entity_with_annoation_uncased_768d.npy \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 42