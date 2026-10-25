import sys
import torch
import torchvision
import torchaudio

def main():
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"torch Version: {torch.__version__}")
    print(f"torchvision Version: {torchvision.__version__}")
    print(f"torchaudio Version: {torchaudio.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")
        
        # Test actual GPU computation
        print("\n--- Testing CUDA Tensor Computation ---")
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        y = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")
        z = x * 2.0 + y  # arithmetic on GPU
        result_cpu = z.cpu()
        print(f"Input X (CUDA): {x}")
        print(f"Input Y (CUDA): {y}")
        print(f"Result (X*2 + Y on CUDA, copied to CPU): {result_cpu}")
        print(f"Computation Successful: {result_cpu.tolist() == [12.0, 24.0, 36.0, 48.0]}")
        print("GPU computation completed with 0 errors!")
    else:
        print("CUDA is NOT available.")

if __name__ == "__main__":
    main()

