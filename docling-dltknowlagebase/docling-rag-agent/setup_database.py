#!/usr/bin/env python3
"""
Database configuration script for Docling RAG Agent.
Helps users set up their PostgreSQL database connection.
"""

import os
import re
from pathlib import Path


def print_database_options():
    """Display database setup options."""
    print("📊 Database Setup Options")
    print("=" * 60)
    print()
    
    print("🚀 Option 1: Supabase (Recommended - Free tier)")
    print("   • Go to https://supabase.com")
    print("   • Create a free account and new project")
    print("   • Go to Settings → Database")
    print("   • Copy 'Connection pooling' string")
    print("   • Format: postgresql://postgres.[ref]:[password]@[host]:5432/postgres")
    print()
    
    print("🌐 Option 2: Neon (Serverless PostgreSQL)")
    print("   • Go to https://neon.tech")
    print("   • Create a free account and database")
    print("   • Copy connection string from dashboard")
    print("   • Format: postgresql://[user]:[password]@[endpoint].neon.tech/[db]")
    print()
    
    print("🏠 Option 3: Local PostgreSQL")
    print("   • Install PostgreSQL with PGVector extension")
    print("   • brew install postgresql pgvector (macOS)")
    print("   • Create database and user")
    print("   • Format: postgresql://user:password@localhost:5432/dbname")
    print()
    
    print("📋 Option 4: Other PostgreSQL Provider")
    print("   • Any PostgreSQL service with PGVector extension")
    print("   • AWS RDS, Google Cloud SQL, Azure, etc.")
    print()


def validate_database_url(url):
    """Basic validation of PostgreSQL URL format."""
    if not url:
        return False, "Database URL cannot be empty"
    
    if not url.startswith('postgresql://'):
        return False, "Database URL must start with 'postgresql://'"
    
    # Basic regex check for PostgreSQL URL format
    pattern = r'^postgresql://[^:]+:[^@]+@[^:]+:\d+/[^/]+$'
    if not re.match(pattern, url):
        return False, "Invalid PostgreSQL URL format. Expected: postgresql://user:password@host:port/database"
    
    return True, "Valid format"


def update_env_file(database_url):
    """Update .env file with new database URL."""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found. Please run setup_config.py first to configure your OpenAI API key.")
        return False
    
    try:
        # Read existing .env file
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update DATABASE_URL line
        updated_lines = []
        database_updated = False
        
        for line in lines:
            if line.strip().startswith('DATABASE_URL='):
                updated_lines.append(f'DATABASE_URL={database_url}\n')
                database_updated = True
            else:
                updated_lines.append(line)
        
        # If DATABASE_URL wasn't found, add it
        if not database_updated:
            updated_lines.append(f'\n# Database Configuration\nDATABASE_URL={database_url}\n')
        
        # Write back to file
        with open(env_path, 'w') as f:
            f.writelines(updated_lines)
        
        print(f"✅ Database configuration updated in {env_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False


def get_database_url():
    """Get database URL from user with validation."""
    print("🔗 Enter Database Connection String")
    print("=" * 50)
    
    while True:
        print("Paste your PostgreSQL connection string:")
        print("(Format: postgresql://user:password@host:port/database)")
        print()
        
        db_url = input("Database URL: ").strip()
        
        if not db_url:
            retry = input("\n❌ No URL entered. Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            continue
        
        # Validate format
        is_valid, message = validate_database_url(db_url)
        if not is_valid:
            print(f"\n❌ {message}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            continue
        
        # Show connection details (masking password)
        masked_url = re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', db_url)
        print(f"\n📋 Connection string: {masked_url}")
        
        confirm = input("Is this correct? (y/n): ").strip().lower()
        if confirm == 'y':
            return db_url


def main():
    """Main database configuration function."""
    print("🤖 Docling RAG Agent - Database Setup")
    print("=" * 60)
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ No .env file found!")
        print("Please run the API key configuration first:")
        print("   python3 setup_config.py")
        print()
        return
    
    print_database_options()
    
    try:
        database_url = get_database_url()
        
        if not database_url:
            print("\n❌ Database configuration cancelled")
            return
        
        # Update .env file
        if update_env_file(database_url):
            print()
            print("🎉 Database configuration completed!")
            print()
            print("Next steps:")
            print("1. Set up database schema:")
            print("   • Copy and run sql/schema.sql in your database")
            print("   • Or use: psql \"$DATABASE_URL\" < sql/schema.sql")
            print()
            print("2. Add documents to documents/ folder")
            print()
            print("3. Run document ingestion:")
            print("   uv run python -m ingestion.ingest --documents documents/")
            print()
            print("4. Start the RAG agent:")
            print("   uv run python cli.py")
            print()
        
    except KeyboardInterrupt:
        print("\n\n❌ Database configuration cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()