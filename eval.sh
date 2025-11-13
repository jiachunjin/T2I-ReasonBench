NAME="zeroshot_qwen25"
IMAGE_PATH="qwen2.5_zero_shot_reasoning"
WORKERS=16

python evaluation/Qwen2.5-VL/eval_entity.py \
  --image_folder /data/phd/jinjiachun/codebase/samples_qwenimage/${IMAGE_PATH}/entity_reasoning \
  --output_path csv_result/entity \
  --prompt_json prompts/entity_reasoning.json \
  --qs_json deepseek_evaluation_qs/evaluation_entity.json \
  --model_name ${NAME} \
  --num_workers ${WORKERS}

python evaluation/Qwen2.5-VL/eval_idiom.py \
  --image_folder /data/phd/jinjiachun/codebase/samples_qwenimage/${IMAGE_PATH}/idiom_interpretation \
  --output_path csv_result/idiom \
  --prompt_json prompts/idiom_interpretation.json \
  --qs_json deepseek_evaluation_qs/evaluation_idiom.json \
  --model_name ${NAME} \
  --num_workers ${WORKERS}

python evaluation/Qwen2.5-VL/eval_scientific.py \
  --image_folder /data/phd/jinjiachun/codebase/samples_qwenimage/${IMAGE_PATH}/scientific_reasoning \
  --output_path csv_result/scientific \
  --prompt_json prompts/scientific_reasoning.json \
  --qs_json deepseek_evaluation_qs/evaluation_scientific.json \
  --model_name ${NAME} \
  --num_workers ${WORKERS}

python evaluation/Qwen2.5-VL/eval_textual_image.py \
  --image_folder /data/phd/jinjiachun/codebase/samples_qwenimage/${IMAGE_PATH}/textual_image_design \
  --output_path csv_result/textual_image \
  --prompt_json prompts/textual_image_design.json \
  --qs_json deepseek_evaluation_qs/evaluation_textual_image.json \
  --model_name ${NAME} \
  --num_workers ${WORKERS}