#!/bin/bash

TEXT_PROMPT='你是一个监控视频分析人员，观看监控视频并对视频标注，核心任务是需要给监控视频起一个简短的标题，不要进行编造，最终输出格式：严格遵守如下JSON输出格式，确保能被解析
JSON{
"description": "描述一下视频中的主要主体和事件",    
"title": [标题长度在12个字内，格式为主体描述+行为+场景可选，不能遗漏背景中运动的物体，对于人的指代仅限于老人/男人/女人/小孩，不用添加任何额外的主体描述。对于车辆需要说明其颜色，不能遗漏任何视频中移动的车辆。对于猫咪的颜色描述仅限于白色、黑色和花猫。如果模型能够准确判断事件发生的场景，也应该将其输出；场景分为室内场景和室外场景，室内场景包括客厅、卧室、餐厅、厨房等子类，室外场景包括广场、车库入口、乡村公路等子类，如果不能准确判断子类场景，输出室内/室外即可。事件中对于车辆，需要描述其运动，包括驶入/停泊/离开，对于人需要描述其关键行为，其中经过门口是一个关键动作，如果发生应该输出，相较之下，上下楼优先级较低，如果不能准确判断上下楼不要输出，如果不能准确识别动作，输出模糊描述，例如活动即可。如果是空镜头（无运动物体），直接输出场景+静止，无事件发生即可。注意需要保证标题完整通顺，主体完整],",
    
"object": "[Type: Array[String]] 列出视频中出现的主体对象及其重要特征
"event": "[Type: Array[String]] 列出视频中出现的主体对象执行的动作或事件"
}
重要约束 (Constraints):
    严禁在JSON之外添加任何文本、解释、或任何"```json"标记。
    你的全部输出必须且只能是上述JSON结构。
    确保 title 字段的值是一个字符串 (String)，而不是包含大括号的对象
'
##########################################################################################
# 1. 定义本地图片路径
IMAGE_PATH="/home/hajimi/Qwen3.5-stuff/scripts/concurrent_test_data/0_1.jpeg"

# 2. 将本地图片转换为 Base64 编码（去除换行符）
# 注意冷知识：Linux 系统使用 base64 -w 0，而 macOS 系统必须使用 base64 -b 0 或 base64 -i
IMAGE_B64=$(base64 -w 0 "$IMAGE_PATH")

# 3. 构造 Data URI 前缀 (假设是 jpeg 格式)
DATA_URI="data:image/jpeg;base64,${IMAGE_B64}"

# 使用 jq 构造安全的 JSON Payload
PAYLOAD=$(jq -n \
  --arg prompt "$TEXT_PROMPT" \
  --arg uri "$DATA_URI" \
  '{
    temperature: 0,
    top_p: 0.95,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: $prompt },
          { type: "image_url", image_url: { url: $uri } }
        ]
      }
    ]
  }')

# 发送请求
curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"

echo -e "\n"

sleep 2
##########################################################################################
# 1. 定义本地图片路径
IMAGE_PATH="/home/hajimi/Qwen3.5-stuff/scripts/concurrent_test_data/0_20.jpeg"

# 2. 将本地图片转换为 Base64 编码（去除换行符）
# 注意冷知识：Linux 系统使用 base64 -w 0，而 macOS 系统必须使用 base64 -b 0 或 base64 -i
IMAGE_B64=$(base64 -w 0 "$IMAGE_PATH")

# 3. 构造 Data URI 前缀 (假设是 jpeg 格式)
DATA_URI="data:image/jpeg;base64,${IMAGE_B64}"

# 使用 jq 构造安全的 JSON Payload
PAYLOAD=$(jq -n \
  --arg prompt "$TEXT_PROMPT" \
  --arg uri "$DATA_URI" \
  '{
    model: "qwen3.5",
    temperature: 0,
    top_p: 0.95,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: $prompt },
          { type: "image_url", image_url: { url: $uri } }
        ]
      }
    ]
  }')

# 发送请求
curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"

echo -e "\n"

sleep 2
##########################################################################################
curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "qwen3.5",
  "temperature": 0,
  "top_p": 0.95,
  "min_p":0,
  "messages": [
    {
      "role": "user",
      "content": "你好啊？你叫什么名字？"
    }
  ]
}'

echo -e "\n"

sleep 2
##########################################################################################
curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "qwen3.5",
  "temperature": 0,
  "max_tokens": 500,
  "top_p": 0.95,
  "min_p":0,
  "messages": [
    {
      "role": "user",
      "content": "解释一下JoJo的奇妙冒险里面败者食尘能力是什么。"
    }
  ]
}'