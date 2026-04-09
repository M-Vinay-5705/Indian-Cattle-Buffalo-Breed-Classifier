import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")


client = Groq(api_key=api_key)


try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        temperature=0.2,
    )
    print("Groq API connected successfully.")
    print("Model:", model_name)
    print("Response:", response.choices[0].message.content)
except Exception as exc:
    print("Groq API call failed.")
    print("Error:", exc)
