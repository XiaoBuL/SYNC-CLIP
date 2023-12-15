import torch
from PIL import Image

def TransformAll(dataset, transform, num_shots):
    transformed_data = None
    num_labels = [0] * dataset.num_classes
    for item in dataset.train_u:
        label = item.label
        img = transform(Image.open(item.impath).convert("RGB"))
        if transformed_data is None:
            transformed_data = torch.zeros((num_shots, dataset.num_classes, img.shape[0], img.shape[1], img.shape[2]))
        transformed_data[num_labels[label], label, :, :, :] = img
        num_labels[label] += 1
    return transformed_data
