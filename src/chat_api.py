import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, AI!"}]
)

print(response.choices[0].message.content)
