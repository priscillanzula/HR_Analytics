"""
START HR ATTRITION API SERVER
==============================
Simple script to launch the FastAPI deployment
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required = ['fastapi', 'uvicorn', 'pydantic', 'joblib', 'pandas', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def check_model_files():
    """Check if model files exist"""
    model_path = 'deployment/models'
    required_files = [
        'random_forest_smote.pkl',
        'xgboost_smote.pkl',
        'metadata.pkl',
        'preprocessor.pkl'
    ]
    
    missing = []
    for file in required_files:
        filepath = os.path.join(model_path, file)
        if not os.path.exists(filepath):
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files in {model_path}/:")
        for file in missing:
            print(f"   - {file}")
        print("\nMake sure all model files are in deployment/models/ folder")
        return False
    
    # Also check if api_smote.py exists
    api_file = os.path.join(model_path, 'api_smote.py')
    if not os.path.exists(api_file):
        print(f"❌ Missing {api_file}")
        print("Make sure api_smote.py is in deployment/models/ folder")
        return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    print("\n" + "="*60)
    print("         HR ATTRITION PREDICTION API")
    print("="*60)
    print("\n✅ Starting server...")
    print("\nAPI will be available at:")
    print("   🌐 http://localhost:8000")
    print("   📚 Documentation: http://localhost:8000/docs")
    print("   🔍 Alternative docs: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # Change to deployment/models directory to run the API
        original_dir = os.getcwd()
        os.chdir('deployment/models')
        
        subprocess.run([
            sys.executable, '-m', 'uvicorn',
            'api_smote:app',
            '--reload',
            '--host', '0.0.0.0',
            '--port', '8000'
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
    finally:
        # Return to original directory
        os.chdir(original_dir)

def main():
    print("\n" + "="*60)
    print("   HR ATTRITION API - STARTUP CHECK")
    print("="*60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return
    print("   ✅ All packages installed")
    
    # Check model files
    print("\n2. Checking model files...")
    if not check_model_files():
        return
    print("   ✅ All model files found")
    
    # Start server
    print("\n3. Starting API server...")
    start_server()

if __name__ == "__main__":
    main()
