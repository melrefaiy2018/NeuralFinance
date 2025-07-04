#!/usr/bin/env python3
"""
Flask app runner with error handling
"""
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

if __name__ == "__main__":
    try:
        print("🚀 Starting Flask app with error handling...")
        
        # Import and run the Flask app
        from flask_app import app
        print("✅ Flask app imported successfully")
        print("🌐 Starting server on http://localhost:8081")
        app.run(debug=True, host='0.0.0.0', port=8081)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
