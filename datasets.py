# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:16:34 2026

@author: user
"""
# -*- coding: utf-8 -*-
"""
Ενοποιημένος κώδικας για τα Datasets: 
1. Synthetic Classification Task
2. Citeseer Node Classification
3. Custom Dataset Template
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
# import pandas as pd # Θα χρειαστεί για το MyCustomDataset

class SyntheticSetDataset(Dataset):
    def __init__(self, num_samples=2000, dim=16, k=10, epsilon=0.2, seed=42):
        super().__init__()
        np.random.seed(seed)
        
        self.num_samples = num_samples
        self.dim = dim
        self.k = k
        self.epsilon = epsilon
        
        mu_0 = np.full(dim, -0.5) 
        mu_1 = np.full(dim, 0.5)
        
        self.original_labels = np.random.randint(0, 2, size=num_samples)
        self.X = np.zeros((num_samples, dim), dtype=np.float32)
        
        for i in range(num_samples):
            if self.original_labels[i] == 0:
                self.X[i] = np.random.normal(loc=mu_0, scale=1.0)
            else:
                self.X[i] = np.random.normal(loc=mu_1, scale=1.0)
                
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(self.X)
        _, indices = nbrs.kneighbors(self.X)
        
        self.final_labels = np.zeros(num_samples, dtype=np.int64)
        self.neighborhood_sets = np.zeros((num_samples, self.k + 1, dim), dtype=np.float32)
        
        self.y_bars = np.zeros(num_samples, dtype=np.float32)
        
        for i in range(num_samples):
            neighborhood_idx = indices[i]
            self.neighborhood_sets[i] = self.X[neighborhood_idx]
            
            strict_neighbors_idx = indices[i, 1:]
            y_bar = np.mean(self.original_labels[strict_neighbors_idx])
            self.y_bars[i] = y_bar
            
            if abs(y_bar - 0.5) < self.epsilon:
                self.final_labels[i] = 1 - self.original_labels[i]
            else:
                self.final_labels[i] = self.original_labels[i]
                
        flipped_count = np.sum(self.original_labels != self.final_labels)
        print(f"[INIT SYNTHETIC] Δημιουργήθηκαν {num_samples} σημεία στις {dim} διαστάσεις.")
        print(f"[INIT SYNTHETIC] Ετικέτες που αντιστράφηκαν (Flipped): {flipped_count} ({flipped_count/num_samples*100:.1f}%)\n")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.neighborhood_sets[idx]), torch.tensor(self.final_labels[idx])


class CiteseerDataset(Dataset):
    def __init__(self, content_path='citeseer.content', cites_path='citeseer.cites'):
        super().__init__()
        
        self.node_to_idx = {}
        features_list = []
        labels_list = []
        label_map = {}
        
        print("Διαβάζεται το αρχείο .content...")
        with open(content_path, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t')
                node_id = parts[0]
                self.node_to_idx[node_id] = idx
                
                features = np.array([float(x) for x in parts[1:-1]], dtype=np.float32)
                features_list.append(features)
                
                label_str = parts[-1]
                if label_str not in label_map:
                    label_map[label_str] = len(label_map)
                labels_list.append(label_map[label_str])
                
        # Αποθήκευση όλων των δεδομένων απευθείας ως Tensors
        self.X = torch.tensor(np.array(features_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(labels_list), dtype=torch.int64)
        num_nodes = len(self.X)
        
        print("Διαβάζεται το αρχείο .cites και χτίζονται οι γειτονιές...")
        self.adj_list = {i: [] for i in range(num_nodes)}
        
        with open(cites_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    node1, node2 = parts[0], parts[1]
                    if node1 in self.node_to_idx and node2 in self.node_to_idx:
                        idx1 = self.node_to_idx[node1]
                        idx2 = self.node_to_idx[node2]
                        self.adj_list[idx1].append(idx2)
                        self.adj_list[idx2].append(idx1)
                        
        for i in range(num_nodes):
            self.adj_list[i] = list(set(self.adj_list[i]))
            
        self.max_deg = max([len(neighbors) for neighbors in self.adj_list.values()])
        print(f"Ο μέγιστος αριθμός γειτόνων (Max Degree) είναι: {self.max_deg}")
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Δυναμική κατασκευή μόνο του δείγματος που ζητείται (Lazy Loading)
        neighbors = self.adj_list.get(idx, [])
        subset_indices = [idx] + neighbors
        
        subset_features = self.X[subset_indices]
        length = len(subset_indices)
        
        padded_features = torch.zeros((1 + self.max_deg, self.X.shape[1]), dtype=torch.float32)
        padded_features[:length] = subset_features
        
        return (
            padded_features, 
            torch.tensor(length, dtype=torch.int64), 
            self.y[idx]
        )


class MyCustomDataset(Dataset):
    
    def __init__(self, file_path):
        super().__init__()
        # TODO: 1. Φόρτωση των δεδομένων (π.χ. df = pd.read_csv(file_path))
        print(f"Διαβάζεται το αρχείο {file_path}...")
        
        # TODO: 2. Εξαγωγή χαρακτηριστικών και ετικετών
        # Παράδειγμα ψευδοκώδικα:
        # self.features = df.drop('label_column', axis=1).values
        # self.labels = df['label_column'].values
        
        # TODO: 3. Διαμόρφωση των δεδομένων σε 'Σύνολα' (Sets) και Padding
        # Αν τα δεδομένα σου είναι ήδη σε σύνολα μεταβλητού μεγέθους, πρέπει να 
        # βρεις το max_len και να γεμίσεις τα μικρότερα με 0.
        
        # Προσωρινά Dummy δεδομένα για να μην χτυπάει σφάλμα ο κώδικας αν το καλέσεις
        self.X = np.zeros((100, 10, 5), dtype=np.float32)  # 100 δείγματα, max 10 στοιχεία/σύνολο, 5 features
        self.lengths = np.random.randint(1, 11, size=100)  # Τυχαία μεγέθη από 1 έως 10
        self.y = np.random.randint(0, 3, size=100)         # 3 τυχαίες κλάσεις (0, 1, 2)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]), 
            torch.tensor(self.lengths[idx]), 
            torch.tensor(self.y[idx])
        )



