python evaluation/Qwen2.5-VL/eval_textual_image.py \
  --image_folder /data/phd/jinjiachun/codebase/samples_qwenimage/t2i_076_images/textual_image_design \
  --output_path csv_result/textual_image \
  --prompt_json prompts/textual_image_design.json \
  --qs_json deepseek_evaluation_qs/evaluation_textual_image.json \
  --model_name 076 \
  --num_workers 4