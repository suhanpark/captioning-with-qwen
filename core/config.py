from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import pydantic
import yaml
import os

load_dotenv()

class Settings(BaseSettings):
    model_name: str = "qwen2.5vl:7b"
    
    @property
    def prompt(self) -> str:
        # Get path relative to the project root
        config_dir = os.path.dirname(os.path.dirname(__file__))
        prompt_path = os.path.join(config_dir, "prompt.yaml")
        with open(prompt_path, "r") as f:
            return str(yaml.safe_load(f))

    class Config:
        env_file = ".env"

settings = Settings()