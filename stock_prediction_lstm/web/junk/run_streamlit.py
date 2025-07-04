#!/usr/bin/env python3
"""
Streamlit runner with proper path setup
"""
import sys
import os
import subprocess

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

# Add both parent directory and project root to path
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

if __name__ == "__main__":
    print("🚀 Starting Stock AI Assistant...")
    print(f"📁 Working directory: {current_dir}")
    print(f"📦 Python path includes: {parent_dir}")
    
    # Set environment variable for Python path
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    new_pythonpath = f"{parent_dir}:{project_root}:{pythonpath}" if pythonpath else f"{parent_dir}:{project_root}"
    env['PYTHONPATH'] = new_pythonpath
    
    print("🔧 Environment configured!")
    print("🌐 Starting Streamlit server...")
    
    try:
        # Run streamlit with the proper environment
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'
        ], cwd=current_dir, env=env)
    except KeyboardInterrupt:
        print("\n👋 Stock AI Assistant stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running Stock AI Assistant: {e}")
        sys.exit(1)
