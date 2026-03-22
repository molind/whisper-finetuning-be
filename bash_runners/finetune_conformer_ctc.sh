mkdir -p logs

# Set CV_DATA_DIR to the extracted Common Voice Belarusian directory
# e.g. ~/.mozdata/datasets/cv-corpus-24.0-2025-12-05/be
CV_DATA_DIR="${CV_DATA_DIR:?Set CV_DATA_DIR to the extracted Common Voice directory}"

python mlx_finetune_conformer.py \
	--model nemo_models/ctc_extracted \
	--dataset-dir "$CV_DATA_DIR" \
	--output-dir ./output_mlx_conformer_ctc \
	--batch-size 4 \
	--grad-accumulation 4 \
	--iters 10000 \
	--learning-rate 1e-4 \
	--warmup-steps 500 \
	--log-every 10 \
	--eval-every 500 \
	--save-every 2000 \
	--val-samples 200 \
	--seed 42 \
	2>&1 | tee "logs/conformer_ctc_$(date +"%Y%m%d-%H%M%S").log"
