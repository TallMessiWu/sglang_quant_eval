curl http://127.0.0.1:6969/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "max_tokens": 512,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0,
  "messages": [
    {
      "role": "user",
      "content": "JoJo的奇妙冒险里最强的替身能力是什么？"
    }
  ]
}
'