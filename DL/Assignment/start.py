import warnings
import joblib
import numpy as np
from torchvision import datasets
from model_utils.preprocess import PreprocessDataset
from model_utils.mlp import ManualMLP
import matplotlib.pyplot as plt


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


def data_preparation():
    train_tensors, train_labels, class_to_name_map = get_train_set()
    train_tensors, train_labels, val_tensors, val_labels = split_validation_set_from_train_set(train_tensors, train_labels)
    test_tensors, test_labels = get_test_set()
    dataset_dict = {"train": {'tensors': train_tensors, 'labels': train_labels},
                    "val": {'tensors': val_tensors, 'labels': val_labels},
                    "test": {'tensors': test_tensors, 'labels': test_labels},
                    "target_class_name_map": class_to_name_map}
    return dataset_dict


def pre_process_dataset(dataset_dict):
    preprocessor = PreprocessDataset(datasets=dataset_dict, verbose=True)
    preprocessor.apply_preprocessing()
    return preprocessor.datasets


def mlp_train(dataset_dict, hidden_layer_sizes, learning_rate, num_epochs,
              activation_function, initialisation_function):
    # Datasets
    train_x = dataset_dict['train']['tensors']
    train_y = dataset_dict['train']['ohe_labels']
    val_x = dataset_dict['val']['tensors']
    val_y = dataset_dict['val']['ohe_labels']
    # I/O Shapes
    input_size = train_x.shape[1]
    output_size = np.unique(dataset_dict['train']['labels']).shape[0]

    layers_str = '_'.join(map(str, hidden_layer_sizes))
    model_name = (f"mlp__layers__{layers_str}__lr_{learning_rate}__epoch_{num_epochs}__"
                  f"activation_{activation_function}__initiation_{initialisation_function}")
    print(f"Model ID: {model_name}")

    mlp = ManualMLP(input_size=input_size, hidden_layers=hidden_layer_sizes, output_size=output_size,
                    learning_rate=learning_rate, epochs=num_epochs,
                    initialisation_function=initialisation_function, activation_function=activation_function)
    probabilities,activations, z_values = mlp.forward(train_x)  # Calculates weights values using weights
    mlp.backward(train_x, train_y, probabilities, activations)
    one_pass_ce = mlp.cross_entropy_loss(y_pred=probabilities, y_true=train_y)  # TODO:NEWW why to calculate this ??
    training_loss, validation_loss = mlp.train(x_train=train_x, y_train=train_y, x_val=val_x, y_val=val_y)
    model_state = mlp.save_model_state()
    trained_dict = {"weights_n_biases": model_state,
                    "initialise": {'input_size': input_size,
                                   'hidden_layers': hidden_layer_sizes,
                                   'output_size': output_size
                                   },
                    "loss": {'train': training_loss, 'val': validation_loss},
                    "epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "model_name": model_name
                    }
    return trained_dict


def compare_training_and_validation_loss(trained_dict):
    fig, ax = plt.subplots(figsize=(14, 4))
    plt.title(f"{trained_dict['model_name']}")
    plt.plot(trained_dict['loss']['train'], label='Train Loss')
    plt.plot(trained_dict['loss']['val'], label='Validation Loss')
    plt.grid(linestyle=':')
    plt.legend()