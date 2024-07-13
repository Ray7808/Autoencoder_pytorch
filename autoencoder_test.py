# import pytorch (cpu, gpu, mac chip)
import torch

# Using cuda/mac m1/cpu device
print("-"*30)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using Cuda GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using M1 GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
print("-"*30)
def main():
    print("hello")
    x = torch.rand(5, 3)
    print(x)
    pass

if __name__ == "__main__":
    main()
