import subprocess
import sys

def install_requirements():
    """Install all required packages from requirements.txt"""
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def check_models():
    """Check if required model files exist"""
    import os
    
    model_paths = ["yolov8n.pt", "models/yolov8n.pt"]
    video_paths = ["855564-hd_1920_1080_24fps.mp4", "video/855564-hd_1920_1080_24fps.mp4"]
    
    model_exists = any(os.path.exists(path) for path in model_paths)
    video_exists = any(os.path.exists(path) for path in video_paths)
    
    if not model_exists:
        print("⚠️ YOLO model file not found. Please download yolov8n.pt and place it in the project root or models/ folder")
        print("You can download it from: https://github.com/ultralytics/assets/releases")
    
    if not video_exists:
        print("⚠️ Sample video file not found. Please place the sample video in the project root or video/ folder")
    
    if model_exists and video_exists:
        print("✅ All required files found!")
    
    return model_exists, video_exists

def main():
    print("Crowd Safety AI Platform - Setup Script")
    print("="*50)
    
    # Install dependencies
    if install_requirements():
        print("\n✅ Setup completed successfully!")
        print("\nTo run the application:")
        print("streamlit run full.py")
        print("\nMake sure to create a .env file with your API keys if you want real API data:")
        print("WAQI_TOKEN=your_token_here")
        print("TOMTOM_API_KEY=your_token_here")
        
        # Check for required files
        print("\nChecking for required files...")
        check_models()
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()