export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

DATA_DIR=/data/yangpan/workspace/dataset/ner
SVAE_DIR=/data/yangpan/workspace/research/SERS/checkpoint
PRETRAIN_MODEL=bert-large-uncased
RESULT_DIR=/data/yangpan/workspace/research/SERS/results
DATA_NAME=ace2004
MODEL_TYPE=SERS
SAVE_NAME=ner/${DATA_NAME}/${PRETRAIN_MODEL}_${MODEL_TYPE}_analysis_label_embedding

python ../../run_ner.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case true \
  --model_name ${MODEL_TYPE} \
  --use_auxiliary_task false \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --gradient_checkpointing true \
  --overwrite_cache false \
  --eval_test true \
  --use_attn true \
  --use_label_embedding true \
  --do_add false \
  --average_pooling false \
  --task_type ner \
  --dump_result false \
  --exist_nested true \
  --weight_start_loss 5 \
  --weight_end_loss 5 \
  --weight_span_loss 1 \
  --num_train_epochs 30 \
  --gradient_accumulation_steps 16 \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 2 \
  --max_seq_length 128 \
  --learning_rate 3e-5 \
  --task_layer_lr 30 \
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
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/label_map.json \
  --label_str_file ${DATA_DIR}/${DATA_NAME}/processed/label_annotation.txt \
  --label_ann_word_id_list_file ${DATA_DIR}/${DATA_NAME}/processed/label_ann_words.txt \
  --label_ann_vocab_file ${DATA_DIR}/${DATA_NAME}/processed/label_annotation_words.json \
  --glove_label_emb_file ${DATA_DIR}/${DATA_NAME}/processed/glove_entity_with_annoation_uncased_whole_300d.npy \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --classifier_dropout_rate 0.0 \
  --seed 42
