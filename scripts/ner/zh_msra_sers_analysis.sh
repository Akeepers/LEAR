export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true

DATA_DIR=/data/yangpan/workspace/dataset/ner
SVAE_DIR=/data/yangpan/workspace/research/SERS/checkpoint
PRETRAIN_MODEL=chinese_roberta_wwm_large_ext_pytorch
RESULT_DIR=/data/yangpan/workspace/research/SERS/results
PRETRAIN_MODEL_DIR=/data/yangpan/workspace/pretrain_models/pytorch
DATA_NAME=zh_msra
MODEL_TYPE=SERS
SAVE_NAME=ner/${DATA_NAME}/${PRETRAIN_MODEL}_${MODEL_TYPE}_analysis_label_embedding

python ../../run_ner.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case false \
  --model_name ${MODEL_TYPE} \
  --span_decode_strategy 'v5' \
  --use_attn true \
  --use_label_embedding true \
  --average_pooling false \
  --do_add false \
  --is_chinese true \
  --use_random_label_emb false \
  --gradient_checkpointing false \
  --use_label_encoding false \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --overwrite_cache false \
  --task_type ner \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 128 \
  --learning_rate 7e-6 \
  --task_layer_lr 10 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path /${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL} \
  --eval_per_epoch 2 \
  --val_skip_epoch 0 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/label_map.json \
  --vocab_file ${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL}/vocab.txt \
  --label_str_file ${DATA_DIR}/${DATA_NAME}/processed/label_annotation.txt \
  --label_ann_word_id_list_file ${DATA_DIR}/${DATA_NAME}/processed/label_ann_words.txt \
  --label_ann_vocab_file ${DATA_DIR}/${DATA_NAME}/processed/label_annotation_words.json \
  --glove_label_emb_file ${DATA_DIR}/${DATA_NAME}/processed/glove_entity_with_annoation_cased_whole_200d.npy \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --seed 42
