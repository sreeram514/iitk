import os

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_utils.cnn_backbone import CNNBackbone, one_hot_encode_nd
from model_utils.mlp import ManualMLP


class CustomImageDataset(Dataset):

    def __init__(self, images, labels):
        # Convert numpy arrays to PyTorch tensors
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.labels = torch.tensor(labels, dtype=torch.long)  # Long for class labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_data_to_cnn(dataset_dict):
    torch_datasets = {}
    for part in tqdm(['train', 'val', 'test']):
        _dataset = CustomImageDataset(images=dataset_dict[part]['tensors'],
                                      labels=dataset_dict[part]['labels'])
        _loader = DataLoader(_dataset, batch_size=64, shuffle=True)
        torch_datasets[part] = {'dataset': _dataset, 'loader': _loader}

    return torch_datasets


def cnn_train(torch_datasets, input_size, hidden_layers, learning_rate, num_epochs, activation_function,
              initialisation_function):

    output_size = 10
    train_losses = []
    val_losses = []
    train_loader = torch_datasets['train']['loader']
    val_loader = torch_datasets['val']['loader']
    mlp_model = ManualMLP(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size,
                          learning_rate=learning_rate, epochs=num_epochs,
                          initialisation_function=initialisation_function, activation_function=activation_function)
    cnn_model = CNNBackbone(input_channels=1)
    print(f"CNN Model details : {cnn_model}")

    for epoch in range(num_epochs):
        cnn_model.train()
        total_train_loss = cnn_training_loss(train_loader, cnn_model, mlp_model, output_size)
        # Record average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Validation phase
        cnn_model.eval()  # Set the CNN model to evaluation mode
        total_val_loss = cnn_validation_loss(val_loader, cnn_model, mlp_model, output_size)
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}  | '
              f'Validation Loss: {avg_val_loss:.4f}')
    mlp_model.model_name = f"CNN_16_32_64_128_256_{mlp_model.model_name}"
    print(f"Model ID: {mlp_model.model_name}")
    model_state = mlp_model.save_model_state()
    cnn_trained_dict = {"weights_n_biases": model_state,
                        "initialise": {'input_size': input_size,
                                       'hidden_layers': hidden_layers,
                                       'output_size': output_size
                                       },
                        "loss": {'train': train_losses, 'val': val_losses},
                        "epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "model_name": mlp_model.model_name,
                        "cnn_state": cnn_model.state_dict()
                        }
    return cnn_trained_dict, mlp_model, cnn_model


def cnn_training_loss(train_loader, cnn_model, mlp_model, output_size=10):
    total_train_loss = 0
    for images, labels in train_loader:
        # Forward pass through CNN
        features = cnn_model(images)  # Get the flattened features
        # Convert labels to one-hot encoding
        one_hot_labels = one_hot_encode_nd(labels=labels.numpy(), num_classes=output_size)
        # Forward pass through MLP
        # probabilities, _, _ = mlp_model.forward(features.detach().numpy(), dropout_details=[1, 0.25])
        probabilities, _, _ = mlp_model.forward(features.detach().numpy())
        # Compute loss
        loss = mlp_model.cross_entropy_loss(probabilities, one_hot_labels)
        total_train_loss += loss
        # Backward pass through MLP
        gradients = mlp_model.backward(features.detach().numpy(), one_hot_labels, probabilities, _)
        # Update weights for MLP
        mlp_model.update_weights(gradients)
    return total_train_loss


def cnn_validation_loss(val_loader, cnn_model, mlp_model, output_size=10):
    total_val_loss = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for val_images, val_labels in val_loader:
            # Forward pass through CNN
            val_features = cnn_model(val_images)  # Get the flattened features
            # Convert validation labels to one-hot encoding
            val_one_hot_labels = one_hot_encode_nd(labels=val_labels.numpy(), num_classes=output_size)
            # Forward pass through MLP
            val_probabilities, _, _ = mlp_model.forward(val_features.detach().numpy())
            # Compute validation loss
            val_loss = mlp_model.cross_entropy_loss(val_probabilities, val_one_hot_labels)
            total_val_loss += val_loss
    return total_val_loss


def cnn_test(dataset_dict, mlp_model, cnn_model):
    cnn_model.eval()
    test_x = dataset_dict['test']['tensors']
    test_y = dataset_dict['test']['ohe_labels']
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_x_tensor = test_x_tensor.view(-1, 1, 28, 28)
    with torch.no_grad():
        # Forward pass through CNN
        cnn_features = cnn_model(test_x_tensor)  # Pass through CNN backbone
        test_probabilities, _, _ = mlp_model.forward(cnn_features.numpy())
    test_predictions = np.argmax(test_probabilities, axis=1)
    y_true = np.argmax(test_y, axis=1)
    cm = confusion_matrix(y_true, test_predictions)
    # Step 3: Calculate metrics
    print("------------ Test Confusion Matrix ------------")
    print(cm)
    print("------------------------------------------------")

    # Display a classification report which includes precision, recall, and F1-score
    print("------------------------------------------------")
    print("\nClassification Report:")
    print(classification_report(y_true, test_predictions))
    print("------------------------------------------------")


def save_model_to_a_file(trained_dict, cnn):
    models_path = os.path.join(os.getcwd(), 'models')
    mlp_models_path = os.path.join(models_path, 'mlp')
    os.makedirs(mlp_models_path, exist_ok=True)
    joblib.dump(value=trained_dict, filename=os.path.join(mlp_models_path, cnn.model_name), compress=3)
    print(f"Model saved in {mlp_models_path}/{cnn.model_name}")