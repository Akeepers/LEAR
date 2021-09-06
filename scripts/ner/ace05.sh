export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true

DATA_DIR=/data/yangpan/workspace/dataset/ner
SVAE_DIR=/data/yangpan/workspace/research/SERS/checkpoint
PRETRAIN_MODEL=bert-large-uncased
RESULT_DIR=/data/yangpan/workspace/research/SERS/results
PRETRAIN_MODEL_DIR=/data/yangpan/workspace/pretrain_models/pytorch
DATA_NAME=ace2005
MODEL_TYPE=bert_ner
SAVE_NAME=ner/${DATA_NAME}/${PRETRAIN_MODEL}_${MODEL_TYPE}_analysis

python ../../run_ner.py \
  --use_cuda true \
  --do_train true \
  --do_eval true \
  --do_lower_case true \
  --model_name ${MODEL_TYPE} \
  --evaluate_during_training true \
  --overwrite_output_dir true \
  --overwrite_cache false \
  --dump_result false \
  --eval_test false \
  --task_type ner \
  --data_tag emnlp \
  --weight_span_loss 1 \
  --span_decode_strategy 'v2' \
  --exist_nested true \
  --num_train_epochs 40 \
  --gradient_accumulation_steps 2 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 128 \
  --learning_rate 3e-5 \
  --task_layer_lr 20 \
  --task_save_name ${SAVE_NAME} \
  --model_name_or_path ${PRETRAIN_MODEL} \
  --eval_per_epoch 2 \
  --val_skip_step 0 \
  --output_dir ${SVAE_DIR}/${SAVE_NAME} \
  --result_dir ${RESULT_DIR}/${DATA_NAME} \
  --data_name ${DATA_NAME} \
  --train_set ${DATA_DIR}/${DATA_NAME}/processed/train.json \
  --dev_set ${DATA_DIR}/${DATA_NAME}/processed/dev.json \
  --test_set ${DATA_DIR}/${DATA_NAME}/processed/test.json \
  --vocab_file ${PRETRAIN_MODEL_DIR}/${PRETRAIN_MODEL}/vocab.txt \
  --first_label_file ${DATA_DIR}/${DATA_NAME}/processed/label_map.json \
  --data_dir ${DATA_DIR}/cached \
  --warmup_proportion 0.0 \
  --dropout_rate 0.1 \
  --classifier_dropout_rate 0.0 \
  --seed 42
