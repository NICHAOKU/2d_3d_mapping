#!/usr/bin/env python3
"""
Startup script for the 2D-3D Mapping Visualizer
This script handles setup and launches the web application
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """
    Check if Python version is compatible
    """
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """
    Check if required dependencies are installed
    """
    required_packages = [
        'flask', 'opencv-python', 'numpy', 'open3d', 
        'scipy', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTo install missing packages, run:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """
    Create necessary directories
    """
    directories = ['uploads', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created/verified")

def run_tests():
    """
    Run basic tests to verify the system
    """
    print("\n🧪 Running basic tests...")
    
    try:
        # Import and test basic functionality
        from mapping_algorithm import Image3DMapper
        print("✅ Algorithm module imported successfully")
        
        import flask
        print("✅ Flask web framework ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def start_application():
    """
    Start the Flask application
    """
    print("\n🚀 Starting the 2D-3D Mapping Visualizer...")
    print("\nThe application will be available at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

def main():
    """
    Main startup function
    """
    print("2D-3D Mapping Visualizer")
    print("========================")
    print("Starting up...\n")
    
    # Check Python version
    if not check_python_version():
        return
    
    print("✅ Python version compatible")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("✅ All dependencies installed")
    
    # Create directories
    create_directories()
    
    # Run tests
    if not run_tests():
        print("\n❌ Basic tests failed. Please check your installation.")
        return
    
    print("✅ Basic tests passed")
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()