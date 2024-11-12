import copy
import numpy as np


class PreprocessDataset:
    
    def __init__(self, datasets, verbose=True):

        self.datasets = datasets
        self.verbose = verbose

    @staticmethod
    def reshape_nd_array(nd_array):
        return nd_array.reshape(nd_array.shape[0], -1)
    
    @staticmethod
    def normalize_array(nd_array):
        return nd_array / 255.0

    @staticmethod
    def one_hot_encode_nd(labels, num_classes):
        """
        One-hot encodes an n-dimensional array of class labels.
        """
        flat_labels = labels.flatten()                                # Flatten the input array to 1D
        one_hot = np.zeros((flat_labels.size, num_classes))           # Create an array of zeros with shape (num_samples, num_classes)
        one_hot[np.arange(flat_labels.size), flat_labels] = 1         # Set the appropriate indices to 1
        return one_hot

    def apply_preprocessing(self, flatten=True):
        """
        Applies preprocessing to all datasets
        """
        for part in ['train', 'test', 'val']:
            part_tensors = self.datasets[part]['tensors']
            
            # Add OHE labels arrays to data partitions
            part_labels = self.datasets[part]['labels']
            num_classes = np.unique(self.datasets[part]['labels']).shape[0]
            part_ohe_labels = self.one_hot_encode_nd(labels=part_labels, num_classes=num_classes)
            self.datasets[part]['ohe_labels'] = part_ohe_labels

            # Reshape the tensors
            if flatten:
                part_reshaped_tensors = self.reshape_nd_array(part_tensors)
            else:
                part_reshaped_tensors = part_tensors
            
            # Normalize the tensors if normalization is enabled
            part_reshaped_normalised_tensors = self.normalize_array(part_reshaped_tensors)

            print(f"--------------------------- {part.title()} Data Processed-----------------------------------")
            if flatten:
                print(f"Flattened `{part}` tensor from shape {part_tensors.shape} to {part_reshaped_tensors.shape}")
            print(f"Normalised `{part}` from (min,max) of {part_tensors.min(), part_tensors.max()} to "
                  f"{part_reshaped_normalised_tensors.min(), part_reshaped_normalised_tensors.max()}")
            print(f"Added One hot encoded label arrays for `{part}`")
            self.datasets[part]['tensors'] = part_reshaped_normalised_tensors
