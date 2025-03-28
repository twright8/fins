#!/usr/bin/env python3
"""
Test script to verify that the DeepSeek API key is correctly loaded from environment variables.
"""
import os
from dotenv import load_dotenv
from src.core.config import CONFIG

# First, check if we have the environment variable
load_dotenv()
env_api_key = os.environ.get('DEEPSEEK_API_KEY')
print(f"DeepSeek API key in environment: {'Found (hidden for security)' if env_api_key else 'Not found'}")

# Then check if it was properly loaded into the config
config_api_key = CONFIG['generation']['deepseek']['api_key']
print(f"DeepSeek API key in config: {'Found (hidden for security)' if config_api_key else 'Not found'}")

# Check if the provider is set correctly
provider = CONFIG['generation']['provider']
print(f"Current generation provider: {provider}")

# Check if we can switch to DeepSeek
deepseek_ready = bool(config_api_key and provider == 'deepseek')
print(f"DeepSeek ready to use: {'Yes' if deepseek_ready else 'No'}")

if not deepseek_ready:
    if not config_api_key:
        print("Error: DeepSeek API key not found in configuration.")
        print("Please set the DEEPSEEK_API_KEY in your .env file or config.yaml")
    
    if provider != 'deepseek':
        print(f"Provider is set to '{provider}', not 'deepseek'.")
        print("Set PROVIDER=deepseek in your .env file or update 'provider' in config.yaml")
else:
    print("Configuration looks good! DeepSeek should be usable.")
