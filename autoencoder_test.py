# Import relative plugins
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

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

# Hyperparameters
BATCH_SIZE = 10

# function start (main)
def main():
    """
    1.Loading MNIST dataset
    2.Testing the loaded image
    """
    
    # load required dataset(download should be True if you run the program in the first time)
    train_loader = DataLoader(datasets.MNIST('./dataset', train=True,
        download=False, transform=transforms.ToTensor()),batch_size=BATCH_SIZE, shuffle=False
    )
    for data, targets in train_loader:
        break
    plt.figure(1)
    print(data.shape)
    plt.imshow(data[5,0,:,:]*255)
    
    plt.figure(2)
    inputs = data.view(-1, 784)
    print(inputs.shape)
    plt.plot(inputs[5,:]*255)
    plt.show()


if __name__ == "__main__":
    main()
