import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from model_utils.mlp import ManualMLP


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

    mlp = ManualMLP(input_size=input_size, hidden_layers=hidden_layer_sizes, output_size=output_size,
                    learning_rate=learning_rate, epochs=num_epochs,
                    initialisation_function=initialisation_function, activation_function=activation_function)
    print(f"Model ID: {mlp.model_name}")
    probabilities, activations, z_values = mlp.forward(train_x)  # Calculates weights values using weights
    mlp.backward(train_x, train_y, probabilities, activations)
    one_pass_ce = mlp.cross_entropy_loss(y_pred=probabilities, y_true=train_y)  # TODO:NEWW why to calculate this ??
    training_loss, validation_loss = mlp.train(x_train=train_x, y_train=train_y, x_val=val_x, y_val=val_y)
    model_state = mlp.save_model_state()
    trained_dict = {"model_state": model_state,
                    "initialise": {'input_size': input_size,
                                   'hidden_layers': hidden_layer_sizes,
                                   'output_size': output_size
                                   },
                    "loss": {'train': training_loss, 'val': validation_loss},
                    "epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "model_name": mlp.model_name,
                    }
    # save_model_to_a_file(trained_dict, mlp)
    return trained_dict, mlp


def mlp_test(mlp_model, dataset_dict):
    test_x = dataset_dict['test']['tensors']
    test_y = dataset_dict['test']['ohe_labels']
    test_probabilities = mlp_model.predict(input_x=test_x)
    test_predictions = np.argmax(test_probabilities, axis=1)     # Convert probabilities to class labels
    y_true = np.argmax(test_y, axis=1)     # Since y_test is one-hot encoded, convert it to class labels
    cm = confusion_matrix(y_true, test_predictions)
    # Calculate metrics
    print("------------ Test Confusion Matrix ------------")
    print(f"------------ f{mlp_model.model_name} ------------")
    print(cm)
    print("------------------------------------------------")

    # Display a classification report which includes precision, recall, and F1-score
    print("------------------------------------------------")
    print("\nClassification Report:")
    print(classification_report(y_true, test_predictions))
    print("------------------------------------------------")


def save_model_to_a_file(trained_dict, mlp):
    models_path = os.path.join(os.getcwd(), 'models')
    mlp_models_path = os.path.join(models_path, 'mlp')
    os.makedirs(mlp_models_path, exist_ok=True)
    joblib.dump(value=trained_dict, filename=os.path.join(mlp_models_path, mlp.model_name), compress=3)
    print(f"Model saved in {mlp_models_path}/{mlp.model_name}")


def compare_training_and_validation_loss(trained_dict):
    fig, ax = plt.subplots(figsize=(14, 4))
    plt.title(f"{trained_dict['model_name']}")
    plt.plot(trained_dict['loss']['train'], label='Train Loss')
    plt.plot(trained_dict['loss']['val'], label='Validation Loss')
    plt.grid(linestyle=':')
    plt.legend()