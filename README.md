# Set-Based Learning with Emergent Label Dynamics



This repository contains the implementation of four Permutation-Invariant architectures (Deep Sets, Set Transformer, Janossy Pooling, PointNet++) for node and set classification problems, featuring dynamic architectural adaptation based on input data.



**Author:**

**Panagiotis Makris (AM: 24023 ΤΝΕΛ)**



## 1. Setup

The code is written in Python. To run the experiments, you need to install PyTorch and the standard data science libraries.

```bash
pip install torch numpy pandas scikit-learn matplotlib
```


## 2. Dataset Selection (Citeseer vs. Synthetic)

The training pipeline is fully automated. Switching between the real graph (Citeseer) and the artificial set (Synthetic Task) is done by commenting/uncommenting the respective lines in the main execution script (main.py):

```python
# Option 1: Synthetic Dataset
# dataset = SyntheticSetDataset(num_samples=10000, dim=16, k=10, epsilon=0.4)

# Option 2: Citeseer Dataset
dataset = CiteseerDataset(content_path="citeseer.content", cites_path="citeseer.cites")
```
## 3. Synthetic Dataset Parameterization (Changing Dimensions)

If you select the synthetic dataset, you can experiment with the "Curse of Dimensionality" by freely modifying the number of samples or the spatial dimensions. The class generates the data dynamically during initialization:

```python
# Example: 10,000 samples in a 128-dimensional space (instead of 16)
train_dataset = SyntheticDataset(num_samples=10000, num_dimensions=128, epsilon=0.111)
```
## 4. Using Custom Data (Bring Your Own Data)

The codebase architecture is data-agnostic. The neural networks dynamically extract the input_dim (number of features) and num_classes (number of classification categories) directly from the data.

To train the models on your own data:

1. Use the provided MyCustomDataset class, which inherits from torch.utils.data.Dataset.
2. Pass your feature matrix (X) and label matrix (y) as torch.Tensor objects.
3. The pipeline will automatically adjust the input/output layer sizes of the models (Deep Sets, etc.) without requiring any changes to the neural network code.

```python
# Example of loading custom data
my_X = torch.tensor(custom_features_array, dtype=torch.float32)
my_y = torch.tensor(custom_labels_array, dtype=torch.long)

custom_dataset = MyCustomDataset(my_X, my_y)
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
```