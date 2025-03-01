import openai
import os


openai_client = openai.OpenAI(api_key=)
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, AI!"}]
)

print(response.choices[0].message.content)
