
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from PIL import Image
import json

def load_checkpoint(filepath):
    # Load the saved model
    savedcheckpoint = torch.load(filepath)
    
    # Init a new model based on the OG
    if savedcheckpoint["arch"] == "densenet121":
        model = models.densenet121(pretrained=True)
    elif savedcheckpoint["arch"] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif savedcheckpoint["arch"] == "alexnet":
        model = models.alexnet(pretrained=True)
    else:
        print("Unsupported model specified. only densenet121, vgg16, and alexnet are supported.")
        exit(1)
    
    # Build the classifier
    hidden_layers = {}
    
    # Loop through the hidden layers and keep building
    for idx, layer in enumerate(savedcheckpoint['hidden_layers']):
        if idx == 0:
            hidden_layers[f'Layer {idx+1}'] = nn.Linear(savedcheckpoint['input_size'], layer)
        else:
            hidden_layers[f'Layer {idx+1}'] = nn.Linear(savedcheckpoint['hidden_layers'][idx-1], layer)
            
        hidden_layers[f'relu{idx+1}'] = nn.ReLU()
        hidden_layers[f'dropout {idx+1}'] = nn.Dropout(p=0.2)

    # Final layer (output)
    final_layer_count = len(savedcheckpoint['hidden_layers']) + 1
    hidden_layers[f'Layer {final_layer_count}'] = nn.Linear(savedcheckpoint['hidden_layers'][-1], savedcheckpoint['output_size'])
    hidden_layers[f'output'] = nn.LogSoftmax(dim=1)
    
    model.classifier = nn.Sequential(OrderedDict(hidden_layers))
    
    # Load state dict
    model.load_state_dict(savedcheckpoint['model_state_dict'])
    
    # Load class to idx
    model.class_to_idx = savedcheckpoint['class_to_idx']
    
    return model, savedcheckpoint

def process_image(image):
    # Load the image
    img = Image.open(image)
   
    # Apply transformations
    tranform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = tranform(img)
    
    return image_tensor

def predict(image, model, topk=5):
    model.eval()
    img = process_image(image)
    img = img.unsqueeze_(0)
    img = img.float()
    
    probability = torch.exp(model.forward(img))
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Only the topk
    probs, labels = probability.topk(topk)
    probs = probs.detach().numpy()[0]
    labels = labels.detach().numpy()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    indexes = [idx_to_class[l] for l in labels]
    flower_names = [cat_to_name[idx_to_class[l]] for l in labels]

    return probs, flower_names

def start(args):
    model, savedcheckpoint = load_checkpoint(args.checkpoint)
    prob, names = predict(args.image_file, model)
    
    return prob, names
