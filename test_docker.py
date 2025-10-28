#!/usr/bin/env python3
"""
Test script to verify Docker setup
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_docker_installation():
    """Test if Docker is installed and running"""
    print("🐳 Testing Docker installation...")
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Docker not working: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker Desktop.")
        return False

def test_docker_compose():
    """Test if Docker Compose is available"""
    print("🔧 Testing Docker Compose...")
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose available: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Docker Compose not working: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose not found.")
        return False

def test_env_file():
    """Test if .env file exists and has required keys"""
    print("🔑 Testing environment configuration...")
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("📝 Please copy env.example to .env and add your API keys:")
        print("   cp env.example .env")
        return False
    
    with open(env_file) as f:
        content = f.read()
    
    required_keys = ['OPENAI_API_KEY', 'GROQ_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if f"{key}=" not in content or f"{key}=your_" in content:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing or incomplete API keys: {', '.join(missing_keys)}")
        print("📝 Please edit .env file with your actual API keys")
        return False
    
    print("✅ Environment configuration looks good!")
    return True

def test_docker_build():
    """Test if Docker image can be built"""
    print("🔨 Testing Docker build...")
    try:
        result = subprocess.run(['docker-compose', 'build'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker build successful!")
            return True
        else:
            print(f"❌ Docker build failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False

def test_application_startup():
    """Test if application starts and responds"""
    print("🚀 Testing application startup...")
    try:
        # Start the application
        subprocess.Popen(['docker-compose', 'up', '-d'])
        
        # Wait for startup
        print("⏳ Waiting for application to start...")
        time.sleep(30)
        
        # Test if application responds
        try:
            response = requests.get('http://localhost:7860', timeout=10)
            if response.status_code == 200:
                print("✅ Application is running and responding!")
                return True
            else:
                print(f"❌ Application returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Application not responding: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Application startup error: {e}")
        return False

def cleanup():
    """Clean up test containers"""
    print("🧹 Cleaning up test containers...")
    try:
        subprocess.run(['docker-compose', 'down'], capture_output=True)
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing RL-Enhanced GHG Consultant Docker Setup")
    print("=" * 60)
    
    tests = [
        test_docker_installation,
        test_docker_compose,
        test_env_file,
        test_docker_build,
        test_application_startup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Docker setup is ready!")
        print("🌐 You can now run: docker-compose up -d")
        print("🔗 Then visit: http://localhost:7860")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("📖 Check README_DOCKER.md for detailed instructions.")
    
    # Cleanup
    cleanup()

if __name__ == "__main__":
    main()
