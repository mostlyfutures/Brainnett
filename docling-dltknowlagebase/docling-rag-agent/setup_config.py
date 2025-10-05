#!/usr/bin/env python3
"""
Configuration setup utility for Docling RAG Agent.
This script helps you set up your OpenAI API key and other environment variables.
"""

import os
import sys
from pathlib import Path


def get_api_key_from_user():
    """Get API key from user input with validation."""
    print("🔑 OpenAI API Key Configuration")
    print("=" * 50)
    print("To use the Docling RAG Agent, you need an OpenAI API key.")
    print("You can get one from: https://platform.openai.com/api-keys")
    print()
    
    while True:
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ API key cannot be empty. Please enter a valid key.")
            continue
            
        if not api_key.startswith('sk-'):
            print("⚠️  Warning: OpenAI API keys typically start with 'sk-'")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        if len(api_key) < 20:
            print("⚠️  Warning: This seems too short for a valid API key")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        return api_key


def create_env_file(api_key, env_path=".env"):
    """Create .env file with the provided API key."""
    env_content = f"""# OpenAI Configuration
OPENAI_API_KEY={api_key}

# Database Configuration
# Update this with your PostgreSQL connection string
DATABASE_URL=postgresql://raguser:ragpass123@localhost:5432/postgres

# Model Selection
LLM_CHOICE=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Development Settings
LOG_LEVEL=INFO
DEBUG_MODE=false
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Configuration saved to {env_path}")


def check_existing_config():
    """Check if .env file already exists and has API key configured."""
    env_path = Path(".env")
    if not env_path.exists():
        return False
    
    try:
        with open(env_path, 'r') as f:
            content = f.read()
        
        # Check if API key is already configured
        if 'OPENAI_API_KEY=' in content and 'your-openai-api-key' not in content:
            for line in content.split('\n'):
                if line.strip().startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    if api_key and api_key != 'your-openai-api-key':
                        return True
        return False
    except Exception:
        return False


def main():
    """Main configuration function."""
    print("🤖 Docling RAG Agent Configuration Setup")
    print("=" * 50)
    
    # Check if already configured
    if check_existing_config():
        print("✅ OpenAI API key is already configured in .env file")
        
        reconfigure = input("Do you want to reconfigure? (y/n): ").strip().lower()
        if reconfigure != 'y':
            print("Configuration unchanged. You're ready to go!")
            return
    
    try:
        # Get API key from user
        api_key = get_api_key_from_user()
        
        # Create .env file
        create_env_file(api_key)
        
        print()
        print("🎉 Configuration completed successfully!")
        print()
        print("Next steps:")
        print("1. Make sure you have PostgreSQL running with PGVector extension")
        print("2. Update DATABASE_URL in .env if needed")
        print("3. Run: uv run python cli.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()