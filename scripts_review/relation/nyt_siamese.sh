export CUDA_VISIBLE_DEVICES=1

DATA_DIR=/data/yangpan/workspace/dataset/relation
SVAE_DIR=/data/yangpan/workspace/nlp_with_label/checkpoint
PRETRAIN_MODEL_DIR=/data/yangpan/workspace/pretrain_models/pytorch
PRETRAIN_MODEL=bert-base-cased
RESULT_DIR=/data/yangpan/workspace/nlp_with_label/results
DATA_NAME=nyt
MODEL_TYPE=bert_relation_label_siamese
SAVE_NAME=relation/${DATA_NAME}/${PRETRAIN_MODEL}_${MODEL_TYPE}

python ../../run_relation.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case false \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --overwrite_cache false \
  --task_type relation \
  --num_train_epochs 40 \
  --add_attention false \
  --add_transformer false \
  --add_cls true \
  --per_gpu_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --per_gpu_eval_batch_size 8 \
  --max_seq_length 128 \
  --learning_rate 1e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name ${MODEL_TYPE} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --eval_per_epoch 1 \
  --val_skip_step 1 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/onepass_format/train.json \
  --vocab_file ${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL}/vocab.txt \
  --dev_set ${DATA_DIR}/${DATA_NAME}/onepass_format/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/onepass_format/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/onepass_format/fake_subject_label_map.json \
  --second_label_file ${DATA_DIR}/${DATA_NAME}/onepass_format/relation_label_map.json \
  --label_str_file ${DATA_DIR}/${DATA_NAME}/onepass_format/label_annotation.txt \
  --label_emb_size 768 \
  --data_dir ${DATA_DIR}/cached \
  --checkpoint ${SVAE_DIR}/${SAVE_NAME} \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 42 \
