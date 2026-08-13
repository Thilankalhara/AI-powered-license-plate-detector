#!/usr/bin/env python3
"""
Quick start script for License Plate Detector
Verifies dependencies and helps with setup
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def check_dependencies():
    """Check if all required packages are installed"""
    required = {
        'cv2': 'opencv-python',
        'PyQt5': 'PyQt5',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'easyocr': 'easyocr',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)  # type: ignore
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return missing


def install_dependencies():
    """Install missing dependencies"""
    missing = check_dependencies()
    
    if not missing:
        print("\n✅ All dependencies installed!")
        return True
    
    print(f"\n⚠️  Missing: {', '.join(missing)}")
    response = input("\nInstall missing packages? (y/n): ")
    
    if response.lower() == 'y':
        print("\n📦 Installing packages...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
            )
            print("✅ Installation complete!")
            return True
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    return False


def download_models():
    """Download required AI models"""
    print("\n🔄 Checking models...")
    
    try:
        print("  Downloading YOLO model...")
        from ultralytics import YOLO  # type: ignore
        YOLO('yolov8n.pt')
        print("  ✅ YOLOv8 model ready")
    except Exception as e:
        print(f"  ⚠️  YOLOv8 download issue: {e}")
    
    try:
        print("  Downloading OCR model...")
        import easyocr  # type: ignore
        easyocr.Reader(['en'])
        print("  ✅ EasyOCR model ready")
    except Exception as e:
        print(f"  ⚠️  EasyOCR download issue: {e}")


def main():
    """Main setup routine"""
    print("=" * 50)
    print("🚗 License Plate Detector - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not install_dependencies():
        print("\n⚠️  Setup incomplete. Run:")
        print("   pip install -r requirements.txt")
        return
    
    # Download models
    download_models()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete! You can now run:")
    print("   python main.py")
    print("=" * 50)


if __name__ == '__main__':
    main()
