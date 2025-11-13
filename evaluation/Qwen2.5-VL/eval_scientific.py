# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
import base64
from openai import OpenAI
import os
import csv
import json
import re
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_json(text):
    # Use a regular expression to find the JSON part
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_string = match.group(0)  # Extract the matched JSON string
        try:
            # Parse the JSON string
            json_data = json.loads(json_string)
            # print("Extracted JSON:", json_data)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
    else:
        print("No JSON found in the text.")
    return json_data

key1 = "xtsrg8n95ff53v3srkhn5k11vxh1jpwjak8g"
key2 = "3st3k7qm36mv0839s869edb7eey63qommvce"
key3 = "dbk89ive7aaagkt209cgvcbyuy0whuwcifx6"

def ask_qw(mmm, processor=None, model=None):
    model_id = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
    if model_id == 1:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = random.choice([key1, key2])
        )
        completion = client.chat.completions.create(
            model="ep-j4xf6w-1762763909712128651",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]
    elif model_id == 2:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = key3
        )
        completion = client.chat.completions.create(
            model="ep-lu1b3u-1763016733960154813",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]
    elif model_id == 3:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = key3
        )
        completion = client.chat.completions.create(
            model="ep-or5pkx-1763017020326247518",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]
    elif model_id == 4:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = key3
        )
        completion = client.chat.completions.create(
            model="ep-652i88-1763017142984904743",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]
    elif model_id == 5:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = key3
        )
        completion = client.chat.completions.create(
            model="ep-ctae6w-1763017348136059709",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]
    elif model_id == 6:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = key3
        )
        completion = client.chat.completions.create(
            model="ep-gs3okx-1763017595342986713",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]
    elif model_id == 7:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = key3
        )
        completion = client.chat.completions.create(
            model="ep-fchnq4-1763017640477993503",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]
    elif model_id == 8:
        client = OpenAI(
            # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
            base_url="http://wanqing.internal/api/gateway/v1/endpoints",
            # base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
            # 从环境变量中获取您的 API Key
            api_key = key3
        )
        completion = client.chat.completions.create(
            model="ep-fk3now-1763017669714918928",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
            messages=mmm,
        )
        output_text = completion.choices[0].message.content
        print(output_text)
        
        return [output_text]


def process_single_image(image_info):
    """处理单张图片的函数，用于多进程"""
    image_name, image_folder, prompt, qs_scientific, qs_detail, qs_quality, num = image_info
    
    try:
        image_path = os.path.join(image_folder, image_name)
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_image = f"data:image/png;base64,{encoded_image_text}"
        
        # Step 1: 描述图片
        q1 = "Describe this image."   
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    },
                    {"type": "text", "text": q1}
                ]
            }
        ]
        
        out1 = ask_qw(messages)[0]
        print(f"[{image_name}] Description generated")

        # Step 2: 评估科学性
        q2 = f"""\
Based on the image and your previous description, answer the following questions: q1, q2, ...
For each question, assign a score of 1, 0.5 or 0 according to the corresponding scoring criteria: c1, c2, ...
Here are the questions and criteria: {qs_scientific}
Carefully consider the image and each question before responding, then provide your answer in json format:
{{"reason": [your detailed reasoning], "score": [s1,s2, ...]"}}"""
        
        new_messages_scientific = messages + [
            {
                "role": "assistant",
                "content": out1,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q2}
                ]
            }
        ]
        
        out2 = ask_qw(new_messages_scientific)[0]
        print(f"[{image_name}] Scientific evaluation completed")
        json_data_2 = extract_json(out2)
        score_scientific = json_data_2['score']

        # Step 3: 评估细节
        q3 = f"""\
Based on the image and your previous description, answer the following questions: q1, q2, ...
For each question, assign a score of 1, 0.5 or 0 according to the corresponding scoring criteria: c1, c2, ...
Here are the questions and criteria: {qs_detail}
Carefully consider the image and each question before responding, then provide your answer in json format:
{{"reason": [your detailed reasoning], "score": [s1,s2, ...]"}}"""

        new_messages_detail = messages + [
            {
                "role": "assistant",
                "content": out1,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q3}
                ]
            }
        ]
        
        out3 = ask_qw(new_messages_detail)[0]
        print(f"[{image_name}] Detail evaluation completed")
        json_data_3 = extract_json(out3)
        score_detail = json_data_3['score']
        
        # Step 4: 评估质量
        q4 = f"""\
Based on the image and your previous description, answer the following questions: q1, q2, ...
For each question, assign a score of 1, 0.5 or 0 according to the corresponding scoring criteria: c1, c2, ...
Here are the questions and criteria: {qs_quality}
Carefully consider the image and each question before responding, then provide your answer in json format:
{{"reason": [your detailed reasoning], "score": [s1,s2, ...]"}}"""

        new_messages_quality = messages + [
            {
                "role": "assistant",
                "content": out1,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q4}
                ]
            }
        ]
        
        out4 = ask_qw(new_messages_quality)[0]
        print(f"[{image_name}] Quality evaluation completed")
        json_data_4 = extract_json(out4)
        score_quality = json_data_4['score']
        
        # 计算平均分
        score_scientific = [float(x) for x in score_scientific]
        score_detail = [float(x) for x in score_detail]
        score_quality = [float(x) for x in score_quality]
        score_scientific_avg = sum(score_scientific) / len(score_scientific)
        score_detail_avg = sum(score_detail) / len(score_detail)
        score_quality_avg = sum(score_quality) / len(score_quality)
        
        print(f"[{image_name}] Scores: scientific={score_scientific_avg:.3f}, detail={score_detail_avg:.3f}, qual={score_quality_avg:.3f}")
        
        return {
            'success': True,
            'image_name': image_name,
            'prompt': prompt,
            'out1': out1,
            'out2': out2,
            'score_scientific': score_scientific,
            'out3': out3,
            'score_detail': score_detail,
            'out4': out4,
            'score_quality': score_quality,
            'score_scientific_avg': score_scientific_avg,
            'score_detail_avg': score_detail_avg,
            'score_quality_avg': score_quality_avg,
            'num': num
        }
    except Exception as e:
        print(f"[{image_name}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'image_name': image_name,
            'error': str(e),
            'num': num
        }


def eval(args):
    model_name = args.model_name
    image_folder = args.image_folder
    output_path = args.output_path
    num_workers = args.num_workers  # 新增：进程数参数
    
    prompt_json = args.prompt_json
    qs_json = args.qs_json

    with open(prompt_json, 'r') as file: 
        prompts = json.load(file)
    
    with open(qs_json, 'r') as file:
        qs = json.load(file)
   
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, f"{model_name}.csv")
    
    # 检查已评估的图片数量
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as csvreader: 
            reader = csv.reader(csvreader)
            lines = list(reader)
            line_count = len(lines)
    else:
        line_count = 0
    
    # 准备所有待处理的图片信息
    all_images = [f for f in os.listdir(image_folder) if f[0].isdigit() and (f[-4:]==".png" or f[-4:]==".jpg")]
    all_images = sorted(all_images)
    print(f"总共 {len(all_images)} 张图片")
    
    evaluated = max(line_count - 1, 0)
    print(f"已评估 {evaluated} 张，待评估 {len(all_images) - evaluated} 张")
    
    # 准备待处理的图片任务列表
    tasks = []
    for i in range(evaluated, len(all_images)):
        image_name = all_images[i]
        num = int(image_name[0:4]) - 1
        prompt = prompts[num]['prompt']
        
        assert qs[num]["id"] == int(image_name[0:4]), f"Image {image_name} does not match the expected id."
        
        qs_scientific = qs[num]["scientific_evaluation"]
        qs_detail = qs[num]["other_details_evaluation"]
        qs_quality = qs[num]["quality_evaluation"]
        
        tasks.append((image_name, image_folder, prompt, qs_scientific, qs_detail, qs_quality, num))
    
    if not tasks:
        print("没有待评估的图片")
        return csv_path
    
    # 如果是新文件，先写入表头
    if line_count == 0:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["id", "prompt", "answer_1", "answer_2", "score_scientific", 
                                "answer_3", "score_detail", "answer_4", "score_qual", 
                                "score_s_avg", "score_d_avg", "score_q_avg"])
            csvfile.flush()
        print("已写入表头")
    
    # 使用多进程处理，并实时写入结果
    print(f"开始使用 {num_workers} 个进程并行处理...")
    completed_count = 0
    success_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_image, task): task for task in tasks}
        
        # 每完成一个任务就立即写入CSV
        for future in as_completed(future_to_task):
            result = future.result()
            completed_count += 1
            
            if result['success']:
                success_count += 1
                # 立即写入CSV
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([
                        result['image_name'],
                        result['prompt'],
                        result['out1'],
                        result['out2'],
                        result['score_scientific'],
                        result['out3'],
                        result['score_detail'],
                        result['out4'],
                        result['score_quality'],
                        result['score_scientific_avg'],
                        result['score_detail_avg'],
                        result['score_quality_avg']
                    ])
                    csvfile.flush()
                print(f"✓ 完成并写入 {result['image_name']} ({completed_count}/{len(tasks)}) - scientific:{result['score_scientific_avg']:.3f}, detail:{result['score_detail_avg']:.3f}, qual:{result['score_quality_avg']:.3f}")
            else:
                failed_count += 1
                print(f"✗ 失败 {result['image_name']} ({completed_count}/{len(tasks)}): {result.get('error', 'Unknown error')}")
    
    print(f"\n评估完成！成功 {success_count} 张，失败 {failed_count} 张")
    print(f"结果已保存到 {csv_path}")
    return csv_path

def model_score(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        accuracy = 0
        quality = 0
        cnt = 0
        for line in lines[1:]:
            try:
                scientific_tmp = float(line[-3]) 
                detail_tmp = float(line[-2])
                qual_tmp = float(line[-1]) 
                
                accuracy+= 0.7*scientific_tmp + 0.3*detail_tmp
                quality+=qual_tmp
                cnt+=1
            except:
                continue
            
        accuracy = round(accuracy/cnt * 100, 2)
        quality = round(quality/cnt * 100, 2)
        print("number of images evaluated: ", cnt, "reasoning accuracy score: ",accuracy, "image quality score: ",quality)
        
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["reasoning accuracy score: ",accuracy, "image quality score", quality]) 
                   
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--prompt_json",
        type=str,
        default = "prompts/scientific_reasoning.json",
        help="path to the prompt",
    )
    parser.add_argument(
        "--qs_json",
        type=str,
        default = "deepseek_evaluation_qs/evaluation_scientific.json",
        help="path to the evaluation question-criterion pairs",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the T2I model to be evaluated",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="path to images",
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="csv_result/scientific", 
        help="path to store the image scores")
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of parallel workers for processing images (default: 4)",
    )
    
    args = parser.parse_args()
    
    csv_path = eval(args)
    model_score(csv_path)