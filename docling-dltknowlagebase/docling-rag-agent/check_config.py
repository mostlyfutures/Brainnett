#!/usr/bin/env python3
"""
Simple environment checker that doesn't require external dependencies.
"""

import os
from pathlib import Path


def check_env_file():
    """Check .env file configuration."""
    env_path = Path('.env')
    
    if not env_path.exists():
        return False, False, "No .env file found"
    
    try:
        with open(env_path, 'r') as f:
            content = f.read()
        
        api_key = None
        db_url = None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
            elif line.startswith('DATABASE_URL='):
                db_url = line.split('=', 1)[1].strip()
        
        has_api_key = api_key and api_key != 'your-openai-api-key' and len(api_key) > 10
        has_db_config = db_url and db_url != 'postgresql://raguser:ragpass123@localhost:5432/postgres'
        
        return has_api_key, has_db_config, "Configuration check complete"
        
    except Exception as e:
        return False, False, f"Error reading .env: {e}"


def main():
    """Main check function."""
    has_api, has_db, message = check_env_file()
    
    print("🔍 Environment Configuration Status")
    print("=" * 40)
    
    if has_api and has_db:
        print("✅ Environment fully configured")
        print("   • OpenAI API key: ✅ Configured")
        print("   • Database URL: ✅ Configured")
        print("\n🚀 Ready to run: uv run python cli.py")
        
    elif has_api and not has_db:
        print("⚠️ Configuration needed: Database")
        print("   • OpenAI API key: ✅ Configured")
        print("   • Database URL: ❌ Using localhost (not configured)")
        print("\n📊 Next: Configure database connection")
        print("   Run: python3 setup_database.py")
        
    elif not has_api and has_db:
        print("⚠️ Configuration needed: API Key")
        print("   • OpenAI API key: ❌ Not configured")
        print("   • Database URL: ✅ Configured")
        print("\n🔑 Next: Configure OpenAI API key")
        print("   Run: python3 setup_config.py")
        
    else:
        print("⚠️ Configuration needed: API Key and Database")
        print("   • OpenAI API key: ❌ Not configured")
        print("   • Database URL: ❌ Using localhost (not configured)")
        print("\n🚀 Next: Run complete setup")
        print("   1. python3 setup_config.py")
        print("   2. python3 setup_database.py")


if __name__ == "__main__":
    main()