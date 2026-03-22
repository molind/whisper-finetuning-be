mkdir -p logs

# Set CV_DATA_DIR to the extracted Common Voice Belarusian directory
# e.g. ~/.mozdata/datasets/cv-corpus-24.0-2025-12-05/be
CV_DATA_DIR="${CV_DATA_DIR:?Set CV_DATA_DIR to the extracted Common Voice directory}"

python mlx_finetune_transducer.py \
	--model nemo_models/transducer_extracted \
	--dataset-dir "$CV_DATA_DIR" \
	--output-dir ./output_mlx_transducer \
	--batch-size 2 \
	--grad-accumulation 8 \
	--iters 10000 \
	--learning-rate 1e-4 \
	--warmup-steps 500 \
	--log-every 10 \
	--eval-every 500 \
	--save-every 2000 \
	--val-samples 200 \
	--seed 42 \
	2>&1 | tee "logs/transducer_$(date +"%Y%m%d-%H%M%S").log"
