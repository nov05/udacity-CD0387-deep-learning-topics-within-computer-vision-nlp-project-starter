import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# ====================================#
# 1. Import SMDebug framework class. #
# ====================================#
import smdebug.pytorch as smd


class Net(nn.Module):
    def __init__(self):
        #TODO: Complete this function
        super(Net, self).__init__()
        # Chaining convolutional, flatten, and fully connected layers in one Sequential block
        self.model = nn.Sequential(
            # Convolutional layers
            ## Example input for MNIST: (64, 1, 28, 28) -> (batch_size, channels, height, width)
            ## 28x28 grayscale image
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),  ## output 14x14
            nn.Dropout(p=0.5),

            # Flattening
            nn.Flatten(),

            # Fully connected layers
            # After MaxPool2d, assuming input (1, 32, 32) -> (32, 14, 14)
            nn.Linear(32*14*14, 128),   ## Input 6272, Output 128
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),

            # Output with LogSoftmax for classification
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        #TODO: Complete the forward function
        return self.model(x)


def train(model, train_loader, optimizer, epoch, hook):
    # =================================================#
    # 2. Set the SMDebug hook for the training phase. #
    # =================================================#
    hook.set_mode(smd.modes.TRAIN)
    model.train()
    print(f"👉 Train Epoch: {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader, hook):
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\n👉 Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, 
            correct, 
            len(test_loader.dataset), 
            100.*correct/len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    # TODO: Add your arguments here
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # TODO: Create your transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization for MNIST dataset
    ])

    # TODO: Add the MNIST dataset and create your data loaders
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = Net()
    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors. #
    # ======================================================#
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    # TODO: Add your optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # ===========================================================#
    # 5. Pass the SMDebug hook to the train and test functions. #
    # ===========================================================#
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, epoch, hook)
        test(model, test_loader, hook)


if __name__ == "__main__":
    main()
