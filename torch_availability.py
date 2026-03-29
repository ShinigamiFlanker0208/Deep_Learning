import torch

def check_gpu():
    if torch.cuda.is_available():
        print("✅ CUDA is available. You are using GPU:")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA is NOT available. You are using CPU.")

if __name__ == "__main__":
    check_gpu()

# torch cuda command pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121