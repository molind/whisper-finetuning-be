mkdir -p logs

# Set CV_DATA_DIR to the extracted Common Voice 23.0 Belarusian directory
# e.g. ~/.mozdata/datasets/cv-corpus-23.0-2025-09-05/be
CV_DATA_DIR="${CV_DATA_DIR:?Set CV_DATA_DIR to the extracted Common Voice directory}"

.venv/bin/python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-tiny" \
	--dataset_dir="$CV_DATA_DIR" \
	--language="be" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--model_index_name="Whisper Tiny Belarusian Debug" \
	\
	--max_steps="50" \
	--max_eval_samples="16" \
	--output_dir="./output_tiny_debug_cv23" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--logging_steps="10" \
	--logging_first_step \
	--learning_rate="1e-4" \
	--warmup_steps="5" \
	--eval_strategy="steps" \
	--eval_steps="25" \
	--save_strategy="steps" \
	--save_steps="25" \
	--gradient_checkpointing \
	\
	--generation_max_length="225" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	\
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--streaming_train="False" \
	--streaming_eval="False" \
	--seed="42" \
	--push_to_hub="False" 2>&1 | tee "logs/train_tiny_debug_cv23_$(date +"%Y%m%d-%H%M%S").log"
