#!/usr/bin/env python3
"""
Welcome script for Docling RAG Agent in VS Code.
"""

import os
from pathlib import Path


def check_configuration():
    """Check if the project is properly configured."""
    env_file = Path(".env")
    
    if not env_file.exists():
        return False, "No .env file found"
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'OPENAI_API_KEY=' not in content:
            return False, "No OpenAI API key configured"
        
        # Check if it's the default placeholder
        for line in content.split('\n'):
            if line.strip().startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                if not api_key or api_key == 'your-openai-api-key':
                    return False, "OpenAI API key not configured"
                
        return True, "Configuration complete"
    
    except Exception as e:
        return False, f"Error checking configuration: {e}"


def main():
    """Display welcome message."""
    print("🤖 Welcome to Docling RAG Agent!")
    print("=" * 50)
    
    configured, message = check_configuration()
    
    if configured:
        print("✅", message)
        print("\n🚀 You're ready to go! Try these commands:")
        print("   • Press F5 to run the agent")
        print("   • Ctrl+Shift+P → 'Tasks: Run Task' → 'Run RAG Agent'")
        print("   • Or run: uv run python cli.py")
    else:
        print("⚠️ ", message)
        print("\n🔧 To get started:")
        print("   • Press Ctrl+Shift+P")
        print("   • Type: 'Tasks: Run Task'")
        print("   • Select: 'Configure OpenAI API Key'")
        print("   • Or run: python3 setup_config.py")
    
    print("\n📚 Documentation:")
    print("   • README.md - Complete setup guide") 
    print("   • docling_basics/ - Learning tutorials")
    print("   • sql/schema.sql - Database setup")


if __name__ == "__main__":
    main()