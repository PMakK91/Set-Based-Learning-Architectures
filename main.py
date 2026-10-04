# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:08:17 2026

@author: user
"""



# -*- coding: utf-8 -*-
"""
@author: user
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import classification_report, confusion_matrix
import copy

from datasets import SyntheticSetDataset, CiteseerDataset
from models import DeepSets, SetTransformer, JanossyPooling, PointNetPlusPlus

def get_model(name, input_dim, output_dim):
    if name == "Deep Sets":
        return DeepSets(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    elif name == "Set Transformer":
        return SetTransformer(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, num_heads=4)
    elif name == "Janossy Pooling":
        return JanossyPooling(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, num_permutations=1)
    elif name == "PointNet++":
        return PointNetPlusPlus(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, k_neighbors=3)

def auto_find_lr(model_name, input_dim, output_dim, train_loader, device, init_lr=1e-6, max_lr=1.0, beta=0.98):
    print(f"  [LR Finder] Σάρωση συνεχούς συνάρτησης LR για το '{model_name}'...")
    model = get_model(model_name, input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    criterion = nn.CrossEntropyLoss()
    
    num_batches = len(train_loader)
    mult = (max_lr / init_lr) ** (1 / num_batches)
    
    lr = init_lr
    best_loss = 0.
    avg_loss = 0.
    losses = []
    log_lrs = []
    
    model.train()
    for batch_num, batch in enumerate(train_loader, 1):
        if len(batch) == 3:
            X, lengths, y = batch
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
        else:
            X, y = batch
            X, y = X.to(device), y.to(device)
            lengths = None
            
        optimizer.zero_grad()
        outputs = model(X, lengths=lengths)
        loss = criterion(outputs, y)
        
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
            
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
            
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        
        loss.backward()
        optimizer.step()
        
        lr *= mult
        for p in optimizer.param_groups:
            p['lr'] = lr
            
    min_loss_idx = np.argmin(losses)
    best_lr_found = (10 ** log_lrs[min_loss_idx]) / 500.0
    
    print(f"  [LR Finder] Βρέθηκε ιδανικό ομαλό LR: {best_lr_found:.6f}\n")
    return best_lr_found

def train_and_evaluate(model, train_loader, test_loader, device, lr=0.001, patience=6):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    
    
    best_loss = float('inf')
    best_weights = None     
    best_epoch = 0
    epochs_no_improve = 0
    epoch = 0
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    model.eval()
    init_test_loss, init_test_correct, init_test_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X, lengths, y = batch
                X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                lengths = None
                
            outputs = model(X, lengths=lengths)
            loss = criterion(outputs, y)
            init_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            init_test_correct += (predicted == y).sum().item()
            init_test_samples += y.size(0)
            
    avg_init_test_loss = init_test_loss / len(test_loader)
    init_test_acc = 100 * init_test_correct / init_test_samples
    print(f"  Epoch [0 ] | LR: {lr:.6f} | --- ΑΡΧΙΚΑ ΒΑΡΗ --- | Test Loss: {avg_init_test_loss:.4f} (Acc: {init_test_acc:.2f}%)")
    
    while True:
        epoch += 1
        model.train()
        total_train_loss, train_correct, train_samples = 0.0, 0, 0
        
        for batch in train_loader:
            if len(batch) == 3:
                X, lengths, y = batch
                X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                lengths = None
            
            optimizer.zero_grad()
            outputs = model(X, lengths=lengths)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y).sum().item()
            train_samples += y.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_samples
        
        model.eval()
        total_test_loss, test_correct, test_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    X, lengths, y = batch
                    X, lengths, y = X.to(device), lengths.to(device), y.to(device)
                else:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    lengths = None
                    
                outputs = model(X, lengths=lengths)
                loss = criterion(outputs, y)
                total_test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == y).sum().item()
                test_samples += y.size(0)
                
        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_samples
        
        
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)
        
        print(f"  Epoch [{epoch:<2}] | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | Test Loss: {avg_test_loss:.4f} (Acc: {test_acc:.2f}%)")
        
        
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0  
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"  [!] Τερματισμός λόγω πλατώ (Απουσία βελτίωσης για {patience} εποχές).")
            break
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"  [*] Το μοντέλο επανήλθε στα βέλτιστα βάρη (από την Εποχή {best_epoch}).")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X, lengths, y = batch
                X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                lengths = None
                
            outputs = model(X, lengths=lengths)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    final_test_acc = 100 * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    final_train_acc = train_accuracies[best_epoch - 1]
    
    print(f"\n  [>] Τελική Ακρίβεια Test Set: {final_test_acc:.2f}% | Αντίστοιχο Train Acc: {final_train_acc:.2f}%\n")
    print(classification_report(all_targets, all_preds, zero_division=0))
            
    return final_test_acc, final_train_acc, train_losses, train_accuracies, test_losses, test_accuracies, all_targets, all_preds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Εκτέλεση σε: {device}\n")

    # =========================================================================
    # ΕΠΙΛΟΓΗ DATASET (Αφαιρείς τα σχόλια από αυτό που θέλεις να τρέξεις)
    # =========================================================================
    
    # Επιλογή 1: Synthetic Dataset
    #print("Φόρτωση Synthetic Dataset...")
    #dataset = SyntheticSetDataset(num_samples=10000, dim=16, k=10, epsilon=0.4)
    
     #Επιλογή 2: Citeseer Dataset 
    print("Φόρτωση Citeseer Dataset...")
    dataset = CiteseerDataset(content_path="citeseer.content", cites_path="citeseer.cites")
    
    # =========================================================================
    # ΚΟΙΝΗ, ΑΥΤΟΜΑΤΟΠΟΙΗΜΕΝΗ ΡΟΗ (Ισχύει και για τα δύο Datasets)
    # =========================================================================
    
    # Εξάγουμε μια δειγματική εγγραφή για να εντοπίσουμε τις διαστάσεις.
    sample = dataset[0]
    
    # Η είσοδος (input_dim) είναι πάντα η τελευταία διάσταση του πρώτου στοιχείου (features).
    input_dim = sample[0].shape[-1]
    
    # Η ετικέτα (label) είναι πάντα το τελευταίο στοιχείο της πλειάδας (index -1),
    # ανεξάρτητα αν η πλειάδα έχει 2 ή 3 στοιχεία.
    all_labels = [dataset[i][-1].item() for i in range(len(dataset))]
    output_dim = len(set(all_labels))
    
    print(f"[ΑΥΤΟΜΑΤΗ ΑΝΑΓΝΩΡΙΣΗ] Χαρακτηριστικά (Input): {input_dim}, Κλάσεις (Output): {output_dim}")
    
    # Διαχωρισμός 80% - 20%
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- ΕΝΑΡΞΗ ΕΚΠΑΙΔΕΥΣΗΣ ΜΟΝΤΕΛΩΝ ---
    model_names = ["Deep Sets", "Set Transformer", "Janossy Pooling", "PointNet++"]

    results_test = {}
    results_train = {}
    histories = {}
    confusion_matrices = {}
    
    for name in model_names:
        print(f"\n{'='*50}")
        print(f" ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ: {name}")
        print(f"{'='*50}")
        
        best_lr = auto_find_lr(name, input_dim, output_dim, train_loader, device)
        model = get_model(name, input_dim, output_dim)
        
        test_acc, train_acc, tr_loss, tr_acc, te_loss, te_acc, targets, preds = train_and_evaluate(
            model, train_loader, test_loader, device, lr=best_lr, patience=6
        )
        
        results_test[name] = test_acc
        results_train[name] = train_acc
        histories[name] = {'train_loss': tr_loss, 'train_acc': tr_acc, 'test_loss': te_loss, 'test_acc': te_acc}
        confusion_matrices[name] = confusion_matrix(targets, preds)

    # --- ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ ---
    print("\n" + "=" * 65)
    print(" ΤΕΛΙΚΟΣ ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ")
    print("=" * 65)
    print(f"{'Μοντέλο':<20} | {'Test Accuracy':<15} | {'Train Accuracy':<15}")
    print("-" * 65)
    
    sorted_results = sorted(results_test.items(), key=lambda x: x[1], reverse=True)
    for name, test_acc in sorted_results:
        train_acc = results_train[name]
        print(f"{name:<20} | {test_acc:>13.2f}% | {train_acc:>13.2f}%")
    print("=" * 65)

    # --- ΓΡΑΦΗΜΑΤΑ LOSS & ACCURACY ---
    print("\nΔημιουργία Γραφημάτων...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    styles = {'Deep Sets': '-', 'Set Transformer': '--', 'Janossy Pooling': '-.', 'PointNet++': ':'}

    for name, hist in histories.items():
        epochs_range = range(1, len(hist['train_loss']) + 1)
        st = styles.get(name, '-')
        
        ax1.plot(epochs_range, hist['train_loss'], linestyle=st, label=f"{name} (Train)", linewidth=1.5)
        ax1.plot(epochs_range, hist['test_loss'], linestyle=st, label=f"{name} (Test)", linewidth=1.5, alpha=0.6)
        
        ax2.plot(epochs_range, hist['train_acc'], linestyle=st, label=f"{name} (Train)", linewidth=1.5)
        ax2.plot(epochs_range, hist['test_acc'], linestyle=st, label=f"{name} (Test)", linewidth=1.5, alpha=0.6)

    ax1.set_title('Σύγκριση Απώλειας: Train vs Test Loss', fontsize=14)
    ax1.set_xlabel('Εποχές', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Σύγκριση Ακρίβειας: Train vs Test Accuracy', fontsize=14)
    ax2.set_xlabel('Εποχές', fontsize=12)
    ax2.set_ylabel('Ακρίβεια (%)', fontsize=12)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- ΠΙΝΑΚΕΣ ΣΥΓΧΥΣΗΣ ---
    print("Δημιουργία Πινάκων Σύγχυσης...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'Πίνακας Σύγχυσης: {name}', fontsize=12)
        ax.set_xlabel('Προβλεπόμενη Κλάση', fontsize=10)
        ax.set_ylabel('Πραγματική Κλάση', fontsize=10)

    plt.tight_layout()
    plt.show()