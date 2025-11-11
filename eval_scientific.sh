python evaluation/Qwen2.5-VL/eval_scientific.py \
  --image_folder /data/phd/jinjiachun/codebase/samples_qwenimage/t2i_076_images/scientific_reasoning \
  --output_path csv_result/scientific \
  --prompt_json prompts/scientific_reasoning.json \
  --qs_json deepseek_evaluation_qs/evaluation_scientific.json \
  --model_name 076 \
  --num_workers 4