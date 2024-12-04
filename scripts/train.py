## TODO: Import your dependencies.
## For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import os
import wandb
import argparse
from datetime import datetime

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow truncated images
## TODO: Import dependencies for Debugging andd Profiling
# ====================================#
# 1. Import SMDebug framework class.  #
# ====================================#
import smdebug.pytorch as smd


class Config:
    def __init__(self):
        self.debug = False


class StepCounter:
    def __init__(self):
        self.total_steps = 0
    
    def __call__(self):
        self.total_steps += 1
    
    def reset(self):
        self.total_steps = 0


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



def test(model, device, data_loader, criterion, 
         config, step_counter, hook, early_stopping,
         phase='eval'):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    if config.debug: hook.set_mode(smd.modes.EVAL)
    model.eval()
    test_loss = 0.
    correct = 0.
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.dataset)
    if phase=='eval': 
        early_stopping(test_loss)
        wandb.log({f"{phase}_loss_epoch": test_loss}, step=step_counter.total_steps)
    accuracy = 100.*correct/len(data_loader.dataset)
    print(
        "\n👉 {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            phase.upper(),
            test_loss, 
            correct, 
            len(data_loader.dataset), 
            accuracy
        )
    )
    if phase=='eval': 
        wandb.log({f"{phase}_accuracy_epoch": accuracy}, step=step_counter.total_steps)



def train(model, device, train_loader, criterion, optimizer, epoch, 
          config, step_counter, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # =================================================#
    # 2. Set the SMDebug hook for the training phase. #
    # =================================================#
    if config.debug: hook.set_mode(smd.modes.TRAIN)
    model.train()
    print(f"👉 Train Epoch: {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):
        step_counter()
        data, target = data.to(device), target.to(device)  ## inputs, labels
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        wandb.log({"train_loss": loss.item()}, step=step_counter.total_steps)
        loss.backward()
        optimizer.step()
        if batch_idx%100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )



def net(model_name, num_classes):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the number of classes
    return model



def main(args):
    config = Config()
    config.debug = args.debug
    step_counter = StepCounter()
    early_stopping = EarlyStopping()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.use_cuda else "cpu"
    print(f"👉 Device: {device}")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  
    train_dataset = datasets.ImageFolder(args.train, transform=transform)
    val_dataset = datasets.ImageFolder(args.validation, transform=transform)
    test_dataset = datasets.ImageFolder(args.test, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_dataset.targets), 
        y=train_dataset.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    ## TODO: Initialize a model by calling the net function
    model = net(args.model_name, len(train_dataset.classes))
    model.to(device)
    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors.  #
    # ======================================================#
    hook = None
    if config.debug:
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)  
    ## TODO: Create your loss and optimizer
    # criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.opt_learning_rate,
        weight_decay=args.opt_weight_decay,
    )  
    '''
    TODO: Call the train function to start training your model
          Remember that you will need to set up a way to get training data from S3
    '''
    # ===========================================================#
    # 5. Pass the SMDebug hook to the train and test functions. #
    # ===========================================================#
    for epoch in range(args.epochs):
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # loss per step
        train(model, device, train_loader, criterion, optimizer, epoch, 
              config, step_counter, hook)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")  ## loss per epoch
        test(model, device, val_loader, criterion, 
             config, step_counter, early_stopping, hook, 
             phase='eval')
        if early_stopping.early_stop:
            print("⚠️ Early stopping")
            break
    ## TODO: Test the model to see its accuracy
    print("🟢 Start testing...")
    test(model, device, test_loader, criterion, 
         config, step_counter, early_stopping, hook, 
         phase='test')
    ## TODO: Save the trained model
    path = os.path.join(args.model_dir, 'model.pth')
    with open(path, 'wb') as f:
        torch.save(model.state_dict(), f)
    print(f"Model saved at '{path}'")



if __name__=='__main__':


    parser=argparse.ArgumentParser()
    # Hyperparameters passed by the SageMaker estimator
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--opt-learning-rate', type=float, default=1e-4)
    parser.add_argument('--opt-weight-decay', type=float, default=1e-4)
    parser.add_argument('--use-cuda', type=bool, default=True)
    # Data, model, and output directories
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    ## Others
    parser.add_argument('--debug', type=str, default=False)
    wandb.init(
        # set the wandb project where this run will be logged
        project="udacity-awsmle-resnet50-dog-breeds",
        config=vars(parser.parse_args())
    )
    args, _ = parser.parse_known_args()
    main(args)
    wandb.finish()