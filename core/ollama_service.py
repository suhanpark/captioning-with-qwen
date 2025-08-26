'''
Enhanced Ollama VLM Captioning Service with health checks
'''
import time
import requests
from typing import Optional
from ollama import generate
from .config import settings
from functools import lru_cache

MODEL: str = settings.model_name
PROMPT: str = settings.prompt

@lru_cache(maxsize=1)
def check_ollama_status() -> tuple[bool, str]:
    """Check if Ollama is running and model is available"""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama is not running. Start with: ollama serve"
        
        # Check if model is available
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if not any(MODEL in name for name in model_names):
            return False, f"Model {MODEL} not found. Pull with: ollama pull {MODEL}"
        
        return True, "Ollama ready"
    except (requests.ConnectionError, requests.Timeout):
        return False, "Cannot connect to Ollama. Start with: ollama serve"
    except Exception as e:
        return False, f"Error checking Ollama: {e}"

def read_image(file_path: str, max_size_mb: int = 10) -> Optional[bytes]:
    """Read image with size validation"""
    try:
        # Check file size
        import os
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            print(f"⚠️ Image {file_path} is large ({file_size_mb:.1f}MB)")
            
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

def get_caption(img_path: str, timeout: int = 30) -> str:
    """Get caption with health checks and timeout protection"""
    # Check Ollama status first
    is_ready, message = check_ollama_status()
    if not is_ready:
        raise ConnectionError(message)
    
    img_data = read_image(img_path)
    if not img_data:
        raise ValueError(f"Could not read image: {img_path}")
    
    res = ''
    start_time = time.time()
    
    try:
        for response in generate(
            model=MODEL,
            prompt=PROMPT,
            images=[img_data],
            stream=True,
            options={
                "temperature": 0.7,  # Lower for more consistent outputs
                "num_predict": 512,  # Limit output length
            }
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Caption generation exceeded {timeout}s")
            res += response['response']
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise
    
    return res
