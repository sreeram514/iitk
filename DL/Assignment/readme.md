# Deep Learning Model for Classification

#### 1. Introduction
This assignment involves building a deep learning model for classification tasks using 2 major components:
- **Multilayer Perceptron (MLP) model** for classification.
- **Convolutional Neural Network (CNN)** backbone for feature extraction.

The task involves the following steps:
1. **Data Processing:** Preprocess data and prepare it for the models.
2. **MLP Model:** Implement an MLP from scratch with a forward and backward pass.
3. **CNN Backbone:** Design a CNN to extract features, which are then passed to the MLP for classification.

The goal is to train a hybrid model combining CNN and MLP, experiment with different hyperparameters, and evaluate the model performance.


#### 2. Directory Structure
The directory is organized as follows:


```plaintext
├── data                       # Fashion MNIST downloaded data 
├── mlp_implementation.ipynb   # Notebook for executing MLP model 
├── cnn_implementation.ipynb   # Notebook for executing CNN model
├── data_preparation.py        # Code for data loading and pre-processing
├── mlp.py                     # Implementation of the MLP model with training functions
├── cnn.py                     # CNN backbone model and integration with MLP
├── mode_utils                 # Core
    ├──  cnn_backbone.py       # CNN backbone model definition
    ├──  mlp.py                # Manual MLP implementation
├── models                     # Final Saved Models 
    ├── mlp
       ├── CNN_16_32_64_128_256_mlp__layers__256_128_64_32_16__lr_0.01__epoch_50__activation_relu__initiation_Xavier
       ├── mlp__layers__512_256_128_64_32_16__lr_0.1__epoch_350__activation_relu__initiation_He  
├── requirements.txt           # List of Python dependencies
└── README.md                  # Documentation for the assignment
```

#### 3. Setting up First Time 

To set up the environment and install required dependencies, 
```bash
pip3 install -r requirements.txt
```

#### 4. Run `mlp_implementation.ipynb` notebook to verify the results of MLP for Classification (Q1, Q2 in Assignment)  

#### 4. Run `cnn_implementation.ipynb` notebook to verify the results of CNN (Q3 in Assignment)
