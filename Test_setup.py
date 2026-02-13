"""
QUICK SETUP TEST
================
Verifies your HR Analytics project is set up correctly
"""

import os
import sys


def check_file_exists(filepath):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}")
    return exists


def main():
    print("\n" + "="*60)
    print("HR ANALYTICS PROJECT - SETUP VERIFICATION")
    print("="*60)

    all_good = True

    # Check deployment structure
    print("\n📁 Checking deployment/models/ folder:")
    required_deployment_files = [
        'deployment/models/api_smote.py',
        'deployment/models/random_forest_smote.pkl',
        'deployment/models/xgboost_smote.pkl',
        'deployment/models/metadata.pkl',
        'deployment/models/preprocessor.pkl',
        'deployment/models/smote_config.pkl'
    ]

    for filepath in required_deployment_files:
        if not check_file_exists(filepath):
            all_good = False

    # Check root files
    print("\n📄 Checking root folder files:")
    required_root_files = [
        'start_api.py',
        'api_client.py',
        'predict_attrition.py',
        'batch_predict.py',
        'sample_employee_data.csv',
        'README.md',
        'DEPLOYMENT_GUIDE.md',
        'ACTION_ITEMS.md'
    ]

    for filepath in required_root_files:
        if not check_file_exists(filepath):
            all_good = False

    # Check Python packages
    print("\n📦 Checking Python packages:")
    packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'joblib',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'requests'
    ]

    missing_packages = []
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (not installed)")
            missing_packages.append(package)
            all_good = False

    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ SETUP COMPLETE - READY TO USE!")
        print("\nNext steps:")
        print("1. Start API: python start_api.py")
        print("2. Test it: python api_client.py")
        print("3. Read ACTION_ITEMS.md for full instructions")
    else:
        print("⚠️  SETUP INCOMPLETE")
        if missing_packages:
            print("\nInstall missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
        print("\nSee DEPLOYMENT_GUIDE.md for help")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
