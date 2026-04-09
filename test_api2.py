import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def list_available_models():
    """Lists available Gemini models."""
    for m in genai.list_models():
        print(m)


if __name__ == "__main__":
    list_available_models()
