#lets import all modules
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import os
import PIL
from PIL import Image
import argparse
#create dictionary which stores different architecture
arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}
#create a function that loads dataset directory and to datatrandformation, etc
def load_data(data_dir = "./flowers" ):
    data_dir =data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    validation_transform = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transform)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transform)
    #Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    data_loaders = [train_loader, val_loader, test_loader]
    image_datasets=[train_data, validation_data, test_data]
    return data_loaders , image_datasets

def load_model(arch="densenet121", gpu=True, learning_rate=0.001, hidden_units=[500, 200], input_size=1024, output_size=102, drop_out=0.20):
    #load_model function loads and sets up a model for training.
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
    else:
        print("model you specified is insupported, only densenet121, vgg16, and alexnet are supported.")
        exit(1) #exits with an error code
    
    # Don't train the other layers
    for param in model.parameters():
        param.requires_grad = False

    # Build the classifier
    hidden_layers_dict = {}
    
    # Loop through the hidden layers and keep building
    for idx, layer in enumerate(hidden_units):
        if(idx == 0):
            hidden_layers_dict[f'Layer {idx+1}'] = nn.Linear(input_size, layer)
        else:
            hidden_layers_dict[f'Layer {idx+1}'] = nn.Linear(hidden_units[idx-1], layer)
                    
        hidden_layers_dict[f'relu{idx+1}'] = nn.ReLU()
        hidden_layers_dict[f'dropout {idx+1}'] = nn.Dropout(p=drop_out)
    # output layer 
    final_layer_count = len(hidden_units) + 1
    hidden_layers_dict[f'Layer {final_layer_count}'] = nn.Linear(hidden_units[-1], output_size)
    hidden_layers_dict[f'output'] = nn.LogSoftmax(dim=1)
    # hidden_layers_dict stores for each layer X    :
            # Linear layer, where X is the layer number.
            # 'reluX': ReLU activation for layer X.
            # 'dropout X': Dropout layer for layer X.
    # And for the output layer:
            # 'Layer Y': Linear layer for the output, where Y is the count of hidden layers + 1.
            # 'output': LogSoftmax activation for the final output.
    # Assign the new classifier
    model.classifier = nn.Sequential(OrderedDict(hidden_layers_dict))

    # Specify the criterion
    criterion = nn.NLLLoss();

    # Specify our optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Track which device is available
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
        
    # Apply that device
    model.to(device)
    
    return model, optimizer, criterion
#let's create training function wjich trains model
def train_model(model, criterion, optimizer, epochs, data_loaders, gpu):
    # Specify params
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 15
    
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")

    # Loop through each epoch
    for epoch in range(epochs):
        for inputs, labels in data_loaders[0]:
            steps += 1
            print(steps)
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in data_loaders[1]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(data_loaders[1]):.3f}.. "
                      f"Validation accuracy: {accuracy/len(data_loaders[1]):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer, criterion  
#let us Save the model checkpoint
def save_model_to_checkpoint(path, model, optimizer, criterion, arch, input_size, output_size, hidden_layers_dict, epochs, image_idx):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {
        'arch': arch,
        'input_size': int(input_size),
        'output_size': int(output_size),
        'hidden_layers_dict': hidden_layers_dict,
        'epochs': int(epochs),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'class_to_idx': image_idx,
        'loss': criterion
    }

    torch.save(checkpoint, path + '/cli_checkpoint.pth')
#let us load the checkpoint
def load_checkpoint(args):
    # Load the data first
    data_loaders, image_datasets = load_data(args.data_dir)
    # Setup the network
    model, optimizer, criterion = load_model(arch=args.arch, gpu=args.gpu, learning_rate=args.learning_rate, hidden_units=args.hidden_units, output_size=args.output_size, input_size=args.input_size, drop_out=args.drop_out)

    # Train the network
    model, optimizer, criterion = train_model(model=model, criterion=criterion, optimizer=optimizer, epochs=args.epochs, data_loaders=data_loaders, gpu=args.gpu)
    
    # Save model
    save_model_to_checkpoint(path=args.path, model=model, optimizer=optimizer, criterion=criterion, arch=args.arch, input_size=args.input_size, output_size=args.output_size, hidden_layers_dict=args.hidden_units, epochs=args.epochs, image_idx=image_datasets[0].class_to_idx)
    
print("everything have runned successfully!")
print(torch.cuda.is_available()) #to check if gpu is enabled or not
