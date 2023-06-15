import numpy as np
import torch.utils.data as data

class NumpyDataset(data.Dataset):
    def __init__(self, x:np.ndarray, y:np.ndarray):
        """
        Inputs:
            size - Number of data points we want to generate
            seed - The seed to use to create the PRNG state with which we want to generate the data points
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        assert isinstance(x, np.ndarray)
        self.size = x.shape[0]
        self.data = x
        self.label = y

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

# class ManipDataSet(data.Dataset):
#     def __init__(self, manip_map, xyz_grids, qtn_grids):
#         self.manip_map = manip_map
#         self.xyz_grids = xyz_grids
#         self.qtn_grids = qtn_grids
    
#     def __len__(self):
#         return np.prod(self.manip_map.shape)

#     def __getitem__(self, idx):
#         qtn_idx = idx % len(self.qtn_grids)
#         xyz_idx = (idx //len(self.qtn_grids)) % len(self.xyz_grids)
#         x = np.hstack([self.xyz_grids[xyz_idx], self.qtn_grids[qtn_idx]])
#         y = self.manip_map.flatten()[idx]
#         return x, y

# def manip_collate_fn(batch):
#     x = [sample[0] for sample in batch]
#     y = [sample[1] for sample in batch]
#     return np.array(x), np.array(y)

# dataset = NumpyDataset(x, y)
# data_loader = data.DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=numpy_collate)