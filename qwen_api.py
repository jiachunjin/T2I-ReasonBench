import os
import base64
from openai import OpenAI
from PIL import Image

# 请确保您已将 API Key 存储在环境变量 WQ_API_KEY 中
# 初始化 OpenAI 客户端，从环境变量中读取您的 API Key
client = OpenAI(
    # 如需办公网调用，请使用：https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints
    # base_url="http://wanqing.internal/api/gateway/v1/endpoints",
    base_url = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
    # 从环境变量中获取您的 API Key
    api_key = "3st3k7qm36mv0839s869edb7eey63qommvce"
)

# Single-round:
model_contents = ["描述这张图片"]
image_path = "1.png"

# 读取本地图片并编码为base64
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_image = f"data:image/png;base64,{encoded_image_text}"

mmm = [
    {
    "role": "user",
    "content": [
        {
        "type": "image_url",
        "image_url": {
            "url": base64_image
        }
        },
        {
        "type": "text",
        "text": "Describe this image."
        }
    ]
    }
]

# q1 = "Describe this image."   
# mmm = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_url",
#                 "image_url": base64_image,
#             },
#             {"type": "text", "text": q1},
#         ],
#     }
# ]

# model_contents.append(image)
completion = client.chat.completions.create(
    model="ep-j4xf6w-1762763909712128651",  # ep-j4xf6w-1762763909712128651 为您当前的智能体应用的ID
    messages=mmm,
)
print(completion.choices[0].message.content)
