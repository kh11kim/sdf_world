import numpy as np
import torch.utils.data as data

class NumpyDataset(data.Dataset):
    """
    A class for creating a dataset from numpy arrays.
    """
    def __init__(self, data:np.ndarray, label:np.ndarray):
        """
        Initialize the dataset with the given data and labels.
        
        Inputs:
            data - A numpy array containing the data points.
            label - A numpy array containing the labels for each data point.
        """
        super().__init__()
        self.size = data.shape[0]
        self.data = data
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    """Collates a batch of samples into a single numpy array.
    
    Args:
        batch (list): A list of samples to be collated.
        
    Returns:
        A single numpy array containing the collated samples.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

# data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate)