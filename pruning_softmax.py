import argparse
import os
import torch
from src.data import get_dataloader
import src.get_model as get_model
import getModel as gM
from src.feat_pruner import FeaturePruner
from src.extract_features import feat_score_centroid
import numpy as np
import random
import pickle
from config import get_data


parser = argparse.ArgumentParser(description="ImageNet pruning")

parser.add_argument("-d", "--data_path", type=str, required=True, help='Path to the dataset')
parser.add_argument("-m", "--mode", type=str, required=True, choices=["pretrained_small", "pretrained_large", "ImageNet", "Places365"], help='Mode: ImageNet or Places365')
parser.add_argument("--batch_size", type=int, default=32, help='Batch size for feature extraction')
parser.add_argument("-r1", "--rate1", type=float, default=0.1, help='Pruning rate for correctly classified samples')
parser.add_argument("-r2", "--rate2", type=float, default=0.1, help='Pruning rate for incorrectly classified samples')
parser.add_argument("-q", "--n_bins", type=int, default=10, help='num bins for pruning')
parser.add_argument("--device", type=str, default="cuda", help='Device to use for computation')
parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
# parser.add_argument("-pm", "--prune_mode", type=str, default="normal", choices=["normal", "cosine_all"], help='Mode for pruning: normal or cosine_all, used for ablation')
args = parser.parse_args()


def main(args):
    device = torch.device(args.device)

    random.seed(args.seed)  
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_folder = 'reduced_result'
    output_dir = os.path.join(output_folder, f"bins{args.n_bins}-r1-{args.rate1}-r2-{args.rate2}-{args.mode}"\
                            #   + f"-{args.prune_mode}"\
                                )
    output_file = os.path.join(output_dir, "selected_paths.txt")
    os.makedirs(output_dir, exist_ok=True)

    # ablation part
    if args.mode == "pretrained_small": 
        train_loader, val_loader, train_dataset, val_dataset = get_data(args.data_path, "ImageNet", args.batch_size, shuffle=False)
        num_classes = len(train_dataset.classes)
        # model = get_model.MobileNetV3_Pretrained(num_classes=num_classes).to(device)
        model_name_dataset = f'mobilenetv3_SoftMax_r-{args.rate1}-q-{args.n_bins}-imgnet-{args.prune_mode}'
        ckpt_path = f'clf_out/{model_name_dataset}/checkpoint.pth.tar'
        model = gM.get_model(model_name_dataset,num_class=num_classes).to(device)
        # load from trained checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
    elif args.mode == "pretrained_large":
        train_loader, val_loader, train_dataset, val_dataset = get_data(args.data_path, "ImageNet", args.batch_size, shuffle=False)
        num_classes = len(train_dataset.classes)
        model = get_model.MobileNetV3_Pretrained(num_classes=num_classes).to(device)

    # usual part
    elif args.mode == "ImageNet":
        train_loader, val_loader, train_dataset, val_dataset = get_data(args.data_path, args.mode, args.batch_size, shuffle=False)
        num_classes = len(train_dataset.classes)
        model = get_model.MobileNetV3_Pretrained(num_classes=num_classes).to(device)
        
    elif args.mode == "Places365":
        train_loader, val_loader, train_dataset, val_dataset = get_data(args.data_path, args.mode, args.batch_size, shuffle=False)
        num_classes = len(train_dataset.classes)
        model_name_dataset = f'mobilenetv3_Places365' # pretrained Places365 model
        ckpt_path = f'checkpoints/{model_name_dataset}/checkpoint.pth.tar'
        model = gM.get_model(model_name_dataset,num_class=num_classes).to(device)
        # load from trained checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    
    # feature extraction + softmax scores + centroid calculation
    if not os.path.exists(os.path.join(output_folder, f"train_features_Softmax_{args.mode}.pt")) or not os.path.exists(os.path.join(output_folder, f"val_features_Softmax_{args.mode}.pt")):
        
        train_features, train_scores, train_labels, train_centroids = feat_score_centroid(model, train_loader, device, num_classes=num_classes)
        torch.save({
            'features': train_features.cpu(),
            'scores': train_scores.cpu(),
            'labels': train_labels.cpu(),
            'centroids': train_centroids.cpu()
        }, os.path.join(output_folder, f"train_features_Softmax_{args.mode}.pt"))

        val_features, val_scores, val_labels, val_centroids = feat_score_centroid(model, val_loader, device, num_classes=num_classes)
        torch.save({
            'features': val_features.cpu(),
            'scores': val_scores.cpu(),
            'labels': val_labels.cpu(),
            'centroids': val_centroids.cpu()
        }, os.path.join(output_folder, f"val_features_Softmax_{args.mode}.pt"))
    else:
        print(f"Found existing features and scores, loading from {output_folder}")
        data = torch.load(os.path.join(output_folder, f"train_features_Softmax_{args.mode}.pt"))
        train_features, train_scores, train_labels, train_centroids = data['features'], data['scores'], data['labels'], data['centroids']
        data = torch.load(os.path.join(output_folder, f"val_features_Softmax_{args.mode}.pt"))
        val_features, val_scores, val_labels, val_centroids = data['features'], data['scores'], data['labels'], data['centroids']

    # Pruning step
    train_sample_paths = [p for p, _ in train_dataset.imgs]
    val_sample_paths = [p for p, _ in val_dataset.imgs]
    train_pruner = FeaturePruner(train_features, train_scores, train_labels, train_centroids, sample_paths=train_sample_paths, 
                                 )
    train_selected_indices = train_pruner.prune(r1=args.rate1, r2=args.rate2, n_bins=args.n_bins, mode=args.prune_mode)
    print(len(train_selected_indices))
    train_count_incorrect = train_pruner.count_incorrect
    
    # with open(f"V1_softmax_train_small_{args.prune_mode}_{args.mode}.pkl", "wb") as f:
    #         pickle.dump(train_pruner.V1_scores, f)
    
    # with open(f"V2_cosine_sim_train_small_{args.prune_mode}_{args.mode}.pkl", "wb") as f:
    #         pickle.dump(train_pruner.V2_cosine_sim, f) # technically self.all_cosine_sim for mode="cosine_all" 

    # for val
    val_pruner = FeaturePruner(val_features, val_scores, val_labels, val_centroids, sample_paths=val_sample_paths)
    val_selected_indices = val_pruner.prune(r1=args.rate1, r2=args.rate2, n_bins=args.n_bins, mode=args.prune_mode)
    print(len(val_selected_indices))
    val_count_incorrect = val_pruner.count_incorrect

    # save validation scores and cosine similarities
    # with open(f"V1_softmax_val_small_{args.prune_mode}_{args.mode}.pkl", "wb") as f:
    #         pickle.dump(val_pruner.V1_scores, f)
    # with open(f"V2_cosine_sim_val_small_{args.prune_mode}_{args.mode}.pkl", "wb") as f:
    #         pickle.dump(val_pruner.V2_cosine_sim, f)

    train_paths = [p for p, _ in train_dataset.imgs]
    val_paths   = [p for p, _ in val_dataset.imgs]
    selected_train_paths = [train_paths[i] for i in train_selected_indices]
    selected_val_paths   = [val_paths[i]   for i in val_selected_indices]
    selected_paths = selected_train_paths + selected_val_paths

    # save selected paths to a text file
    with open(output_file, "w") as f:
        for path in selected_paths:
            f.write(path + "\n")
        # f.write(f'Incorrect samples: {train_count_incorrect} + {val_count_incorrect}\n')
        f.write(f'Total samples: {len(selected_paths)}')
    print(f"Saved {len(selected_paths)} selected paths to {output_file}")
    # calculate accuracy of the model based on the scores and labels as sanity check
    predicted_labels = torch.argmax(train_scores, dim=1)
    accuracy = (predicted_labels == train_labels).float().mean().item()
    print(f"Model accuracy on the full dataset: {accuracy:.4f}")

    

if __name__ == "__main__":
    main(args)

