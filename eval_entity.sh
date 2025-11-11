python evaluation/Qwen2.5-VL/eval_entity.py \
  --image_folder /data/phd/jinjiachun/codebase/samples_qwenimage/t2i_076_images/entity_reasoning \
  --output_path csv_result/entity \
  --prompt_json prompts/entity_reasoning.json \
  --qs_json deepseek_evaluation_qs/evaluation_entity.json \
  --model_name 076 \
  --num_workers 8