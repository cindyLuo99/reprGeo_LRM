import torch
import json
import os
from torchvision import datasets
from torch.utils.data import DataLoader

def load_imagenet_label_maps(json_path):
    # Get the name of an index label
    # Load the class index file
    with open(json_path, 'r') as f:
        class_idx = json.load(f)

    # Create a dict mapping category index ("nxxxxxxxx") to label ('cat', 'frog', ...)
    idx2label = {class_idx[str(k)][0]:class_idx[str(k)][1] for k in range(len(class_idx))}

    # Create a dict mapping category index ("nxxxxxxxx") to class index (0,1,2,3,...)
    idx2number = {class_idx[str(k)][0]:k for k in range(len(class_idx))}

    # Create a dict mapping class index (0,1,2,3,...) to label ('cat', 'frog', ...)
    number2label = {k:class_idx[str(k)][1] for k in range(len(class_idx))}

    return idx2label, idx2number, number2label


def create_dataLoader(dataset_path, transform, idx2number, batch_size, num_workers):
    # Load the dataset using ImageFolder (ignores the labels for now)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Override the labels using the idx2number dictionary
    # `dataset.samples` is a list of (image_path, label), and we will update the label part
    for i, (img_path, _) in enumerate(dataset.samples):
        folder_name = os.path.basename(os.path.dirname(img_path))  # Get the folder name (n01737021, etc.)
        dataset.samples[i] = (img_path, idx2number[folder_name])  # Assign the new label from idx2number

    # Create the DataLoader with the updated labels
    val_dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)

    return val_dataloader


def get_activation(model, dataloader: DataLoader, layer_of_interest, feature_extractor, number2label, device, forward_passes=1, drop_state=True, get_accuracy=True):
    model.eval()

    activations = []
    categories = []
    allLabels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)  # Move labels to device
            print(f'Evaluating {len(images)} images in {labels[0], number2label[labels[0].item()]}')

            # Forward pass through the model and track activations
            with feature_extractor as extractor:
                outputs = model(images, drop_state=drop_state, forward_passes=forward_passes)
                features = extractor._features  # Extracted features

            # Store the activations for the layers you're interested in
            # take the last item (activation from the last pass)
            activations.append(features[layer_of_interest][-1])  
            
            # Collect the corresponding labels
            categories.extend([number2label[label.item()] for label in labels])
            allLabels.extend(labels)

            if get_accuracy:
                # Calculate accuracy
                _, predicted = torch.max(outputs[-1], 1)  # Get the predictions from the final layer
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        # Flatten activations
        activations = torch.cat(activations, dim=0)
        allLabels = torch.tensor(allLabels)

    if get_accuracy:
        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
    
    else:
        accuracy = None

    return activations, categories, allLabels, accuracy