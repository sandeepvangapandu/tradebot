import os
import sys
import logging

# Configure path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.llm_client import LLMClient
from dotenv import load_dotenv

def test_groq_key():
    logging.basicConfig(level=logging.INFO)
    print("Loading .env file...")
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))
    
    key = os.environ.get("GROQ_API_KEY")
    if not key or key == "your_actual_groq_api_key_here":
        print("❌ Groq API key not found or is still the placeholder in .env!")
        return
        
    print(f"✅ Found GROQ_API_KEY (starts with {key[:8]}...)")
    
    try:
        print("Testing LLMClient connection to Groq...")
        client = LLMClient(api_key=key, model="llama3-8b-8192")
        
        response = client.invoke(
            system_prompt="You are a helpful assistant.",
            prompt="Say 'hello world' and give me a status code 200.",
        )
        print("✅ Received valid response from Groq:")
        print(response)
    except Exception as e:
        print(f"❌ Failed to connect to Groq API: {e}")

if __name__ == "__main__":
    test_groq_key()
