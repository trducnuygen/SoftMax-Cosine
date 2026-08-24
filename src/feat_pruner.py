import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F


class FeaturePruner:
    "Softmax-based pruning of features based on confidence scores and cosine similarity to class centroids."
    def __init__(self, features=None, scores=None, labels=None, centroids=None, sample_paths=None):
        self.features = features
        self.scores = scores
        self.labels = labels
        self.centroids = centroids
        self.sample_paths = sample_paths
        self.V1, self.V2, self.count_incorrect = [], [], 0
        self.unique_labels = np.unique(self.labels.cpu().numpy())
        self.count_incorrect = None
        self.V1_paths, self.V2_paths = [], []
        

        count_incorrect = 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            score = self.scores[i][label].item()  # confidence score for the true label
            feature = self.features[i]

            pred_label = self.scores[i].argmax().item()
            if pred_label == label.item():
                self.V1.append((score, label.item(), i))
                self.V1_paths.append(self.sample_paths[i] if self.sample_paths is not None else None)
            else:
                self.V2.append((feature.cpu(), label.item(), i))
                self.V2_paths.append(self.sample_paths[i] if self.sample_paths is not None else None)
                count_incorrect += 1

        self.count_incorrect = count_incorrect

        # save Softmax score for all samples in V1
        self.V1_scores = zip(self.V1_paths, np.array([x[0] for x in self.V1]))

    def _stratified_sample(self, values, indices, labels, r, n_bins=10):
        '''
        Shared core: per-class quantile binning of `values`, then sample r% from each bin.

        input:
        - values: 1D array of scalar values to bin by (e.g. softmax score, cosine similarity).
        - indices: 1D array of original sample indices, aligned with `values`.
        - labels: 1D array of class labels, aligned with `values`.
        - r: fraction to keep per bin.
        - n_bins: number of bins for quantile-based grouping.

        output:
        - selected_indices: list of indices selected across all classes.
        '''
        if r == 0:  # purposefully empty set for ablation
            return []

        selected_indices = []
        for lbl in self.unique_labels:
            cls_mask = labels == lbl
            cls_values = values[cls_mask]
            cls_indices = indices[cls_mask]

            if len(cls_values) == 0:
                continue
            elif len(cls_values) < n_bins + 1:
                selected_indices.extend(cls_indices)
                continue

            # sort samples by value
            sorted_order = np.argsort(cls_values)

            # split sorted positions into bins
            rank_bins = np.array_split(sorted_order, n_bins)

            for rank_idx in rank_bins:
                bin_indices = cls_indices[rank_idx]

                if len(bin_indices) == 0:
                    continue

                n_sample = max(1, int(len(bin_indices) * r))
                n_sample = min(n_sample, len(bin_indices))

                selected = np.random.choice(
                    bin_indices,
                    size=n_sample,
                    replace=False
                )

                selected_indices.extend(selected)

        return selected_indices
        
    
    def V1_prune(self, r, n_bins=10):
        if r == 0:
            return []

        V1 = np.array(self.V1, dtype=object)  # (score, label, index)
        scores = np.array([x[0] for x in V1])
        labels = np.array([x[1] for x in V1])
        indices = np.array([x[2] for x in V1])

        return self._stratified_sample(scores, indices, labels, r, n_bins=n_bins)

    def V2_prune(self, r, n_bins=10):
        if r == 0:
            return []

        V2 = np.array(self.V2, dtype=object)  # (feature, label, index)
        features = np.array([x[0] for x in V2])
        labels = np.array([x[1] for x in V2])
        indices = np.array([x[2] for x in V2])
        centroids = self.centroids.cpu().numpy()

        cosine_sim = np.zeros(len(V2))
        for i in range(len(V2)):
            centroid = centroids[labels[i]]
            cosine_sim[i] = np.dot(features[i], centroid) / (
                np.linalg.norm(features[i]) * np.linalg.norm(centroid) + 1e-8
            )

        self.V2_cosine_sim = zip(self.V2_paths, cosine_sim)

        return self._stratified_sample(cosine_sim, indices, labels, r, n_bins=n_bins)

    def cosine_prune_all(self, r, n_bins=10):
        if r == 0:
            return []

        features = self.features.cpu().numpy() if hasattr(self.features, 'cpu') else np.asarray(self.features)
        labels = self.labels.cpu().numpy() if hasattr(self.labels, 'cpu') else np.asarray(self.labels)
        indices = np.arange(len(labels))
        centroids = self.centroids.cpu().numpy() if hasattr(self.centroids, 'cpu') else np.asarray(self.centroids)

        cosine_sim = np.zeros(len(labels))
        for i in range(len(labels)):
            centroid = centroids[labels[i]]
            cosine_sim[i] = np.dot(features[i], centroid) / (
                np.linalg.norm(features[i]) * np.linalg.norm(centroid) + 1e-8
            )

        self.all_cosine_sim = zip(self.sample_paths, cosine_sim) if self.sample_paths is not None else None
        self.V2_cosine_sim = self.all_cosine_sim  # for saving later

        return sorted(self._stratified_sample(cosine_sim, indices, labels, r, n_bins=n_bins))

    def prune(self, r1=0.1, r2=0.1, n_bins=10, mode="cosine_all"):
        if mode != "cosine_all":
            V1_indices = self.V1_prune(r1, n_bins=n_bins)
            V2_indices = self.V2_prune(r2, n_bins=n_bins)
            print(f'number of samples in V1: {len(self.V1)}, after pruning: {len(V1_indices)}')
            print(f'number of samples in V2: {len(self.V2)}, after pruning: {len(V2_indices)}')
            final_set = []
            for idx in V1_indices:
                final_set.append(idx)

            for idx in V2_indices:
                final_set.append(idx)

            final_set = sorted(list(set(final_set)))
        elif mode == "cosine_all":
            final_set = self.cosine_prune_all(r1, n_bins=n_bins)
        return final_set