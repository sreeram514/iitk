import warnings
import numpy as np
from torchvision import datasets
from model_utils.preprocess import PreprocessDataset


def get_test_set():
    with warnings.catch_warnings(action="ignore"):
        # Load Training dataset from torchvision as suggested in assignment
        test_dataset = datasets.FashionMNIST(root='./data', download=True, train=False)

        # Extract arrays and labels
        test_tensors = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()
        test_dataset_shape = test_tensors.shape

        # Divide all pixels in train and test datasets by 255 to scale the values between 0-1
        assert test_dataset_shape[0] == test_labels.shape[0]
        print("------------------------- TEST DATA ---------------------------------")
        print(f"Test dataset contains {format(test_dataset_shape[0], ',')} images.")
        print(f"Each image is of size {test_dataset_shape[1]}x{test_dataset_shape[2]}")
        print(f"Each image is labelled one of: {test_dataset.classes}")
        print(f"Each image pixel values are between {test_tensors.min()} and {test_tensors.max()}")
        print("---------------------------------------------------------------------")
        return test_tensors, test_labels


def split_validation_set_from_train_set(train_tensors, train_labels):
    total_train_size = train_tensors.shape[0]
    train_size = int(0.88 * total_train_size)
    val_size = int(0.12 * total_train_size)
    np.random.seed(42)  # Set random seed for reproducibility
    indices = np.arange(total_train_size)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    # Create validation set
    val_tensors = train_tensors[val_indices]
    val_labels = train_labels[val_indices]
    # Create training set
    train_tensors = train_tensors[train_indices]
    train_labels = train_labels[train_indices]
    print("------------------------- TRAIN DATA After Split -----------------------")
    print(f"Final Training dataset shape: {(train_tensors.shape[0], ',')} images of "
          f"{train_tensors.shape[1]}x{train_tensors.shape[2]}")
    print(f"Final Validation dataset shape: {(val_tensors.shape[0], ',')} images of "
          f"{val_tensors.shape[1]}x{val_tensors.shape[2]}")
    print("---------------------------------------------------------------------")

    return train_tensors, train_labels, val_tensors, val_labels


def get_train_set():
    with warnings.catch_warnings(action="ignore"):
        # Load Training dataset from torchvision as suggested in assignment
        train_dataset = datasets.FashionMNIST(root='./data', download=True, train=True)

        # Extract arrays and labels
        train_tensors = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        train_dataset_shape = train_tensors.data.shape
        class_to_name_map = train_dataset.class_to_idx
        assert train_tensors.shape[0] == train_labels.shape[0]  # Ensuring same sizes for train inputs and outputs
        print("-------------------------TRAIN DATA---------------------------------")
        print(f"Available Training dataset contains {format(train_dataset_shape[0], ',')} images.")
        print(f"Each image is of size {train_dataset_shape[1]}x{train_dataset_shape[2]}")
        print(f"Each image is labelled one of: {train_dataset.classes}")
        print(f"Each image pixel values are between {train_tensors.min()} and {train_tensors.max()}")
        print("---------------------------------------------------------------------")
        return train_tensors, train_labels, class_to_name_map


def prepare_dataset_dict():
    train_tensors, train_labels, class_to_name_map = get_train_set()
    train_tensors, train_labels, val_tensors, val_labels = split_validation_set_from_train_set(train_tensors, train_labels)
    test_tensors, test_labels = get_test_set()
    dataset_dict = {"train": {'tensors': train_tensors, 'labels': train_labels},
                    "val": {'tensors': val_tensors, 'labels': val_labels},
                    "test": {'tensors': test_tensors, 'labels': test_labels},
                    "target_class_name_map": class_to_name_map}
    return dataset_dict


def pre_process_dataset(dataset_dict, flatten=True):
    preprocessor = PreprocessDataset(datasets=dataset_dict, verbose=True)
    preprocessor.apply_preprocessing(flatten=flatten)
    return preprocessor.datasets