import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def feat_score_centroid(model, dataloader, device, num_classes=None):
    model = model.to(device)
    model.eval()

    features_list = []
    scores_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features and scores"):
            images, labels = images.to(device), labels.to(device)
            
            features = model.extract_features(images)  # features
            scores = model(images)  # logits
            scores = F.softmax(scores, dim=1)  # convert to probabilities
            for i in range(labels.size(0)):
                features_list.append(features[i].cpu())
                scores_list.append(scores[i].cpu())
                labels_list.append(labels[i].cpu())

    features = torch.stack(features_list)  # (N, C)
    print(f"Features shape: {features.shape}, Scores: {len(scores_list)}, Labels shape: {len(labels_list)}")
    scores = torch.stack(scores_list)      # (N, C)
    labels = torch.stack(labels_list)           # (N,)
    centroid = centroid_per_class(features, labels, num_classes=num_classes)  # (num_classes, C)
    return features, scores, labels, centroid

def centroid_per_class(features, labels, num_classes):
    centroids = torch.zeros((num_classes, features.size(1)), device=features.device)
    for c in range(num_classes):
        class_mask = (labels == c)
        centroids[c] = features[class_mask].mean(dim=0)
    return centroids