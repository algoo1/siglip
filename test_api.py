#!/usr/bin/env python3
"""
Simple test script to validate API structure.
This script checks the main.py file for proper FastAPI setup.

For full testing, run this after installing dependencies:
pip install -r requirements.txt
python test_api.py
"""

import ast
import sys
from pathlib import Path

def validate_main_py():
    """Validate that main.py has the required structure"""
    main_file = Path("main.py")
    
    if not main_file.exists():
        print("❌ main.py not found")
        return False
    
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check for required components
        tree = ast.parse(content)
        
        # Check for required imports
        required_imports = ['fastapi', 'torch', 'transformers']
        imports_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports_found.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports_found.append(node.module)
        
        missing_imports = [imp for imp in required_imports if not any(imp in found for found in imports_found)]
        
        if missing_imports:
            print(f"❌ Missing imports: {missing_imports}")
            return False
        
        # Check for FastAPI app creation
        app_created = any(
            isinstance(node, ast.Assign) and 
            any(target.id == 'app' for target in node.targets if isinstance(target, ast.Name))
            for node in ast.walk(tree)
        )
        
        if not app_created:
            print("❌ FastAPI app not created")
            return False
        
        # Check for required endpoints by searching in content
        required_endpoints = ['/embed/text', '/embed/image', '/health']
        missing_endpoints = []
        
        for endpoint in required_endpoints:
            if endpoint not in content:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing endpoints: {missing_endpoints}")
            return False
        
        print("✅ main.py structure validation passed")
        print("✅ Required imports found")
        print("✅ FastAPI app created")
        print("✅ Required endpoints defined")
        return True
        
    except Exception as e:
        print(f"❌ Error validating main.py: {e}")
        return False

def validate_requirements():
    """Validate requirements.txt exists and has required packages"""
    req_file = Path("requirements.txt")
    
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_packages = ['fastapi', 'torch', 'transformers', 'uvicorn', 'pillow']
        missing_packages = []
        
        for package in required_packages:
            if package.lower() not in content.lower():
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements.txt: {missing_packages}")
            return False
        
        print("✅ requirements.txt validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating requirements.txt: {e}")
        return False

def validate_dockerfile():
    """Validate Dockerfile exists and has proper structure"""
    dockerfile = Path("Dockerfile")
    
    if not dockerfile.exists():
        print("❌ Dockerfile not found")
        return False
    
    try:
        with open(dockerfile, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_instructions = ['FROM', 'COPY', 'RUN', 'EXPOSE', 'CMD']
        missing_instructions = []
        
        for instruction in required_instructions:
            if instruction not in content:
                missing_instructions.append(instruction)
        
        if missing_instructions:
            print(f"❌ Missing Dockerfile instructions: {missing_instructions}")
            return False
        
        if '8000' not in content:
            print("❌ Port 8000 not exposed in Dockerfile")
            return False
        
        print("✅ Dockerfile validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating Dockerfile: {e}")
        return False

def main():
    """Run all validations"""
    print("🔍 Validating SigLIP API codebase...\n")
    
    validations = [
        ("Main Application", validate_main_py),
        ("Requirements", validate_requirements),
        ("Docker Configuration", validate_dockerfile)
    ]
    
    all_passed = True
    
    for name, validation_func in validations:
        print(f"📋 {name}:")
        passed = validation_func()
        all_passed = all_passed and passed
        print()
    
    if all_passed:
        print("🎉 All validations passed! The codebase is ready for deployment.")
        print("\n📝 Next steps:")
        print("1. Push code to GitHub")
        print("2. Deploy on RunPod using the instructions in README.md")
        print("3. Test the deployed API endpoints")
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()