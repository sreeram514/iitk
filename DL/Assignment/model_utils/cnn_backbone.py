from torch.nn.modules import Conv2d, MaxPool2d, BatchNorm2d
from torch.nn.modules.module import Module
from torch.nn.functional import relu, softmax, sigmoid
import numpy as np


class CNNBackbone(Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = Conv2d(input_channels, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.bn2 = BatchNorm2d(32)
        self.conv3 = Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.bn3 = BatchNorm2d(64)
        self.conv4 = Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.bn4 = BatchNorm2d(128)
        self.conv5 = Conv2d(128, 256, kernel_size=2, stride=1, padding=1)
        self.bn5 = BatchNorm2d(256)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(relu(self.bn1(self.conv1(x))))
        x = self.pool(relu(self.bn2(self.conv2(x))))
        x = self.pool(relu(self.bn3(self.conv3(x))))
        x = self.pool(relu(self.bn4(self.conv4(x))))
        x = self.pool(relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)  # Flatten the tensor before returning
        return x


def one_hot_encode_nd(labels, num_classes):
    """
    One-hot encodes an n-dimensional array of class labels.

    Parameters:
    ----------
    labels : numpy.ndarray
        n-dimensional array of class labels.

    num_classes : int
        The total number of unique classes.

    Returns:
    -------
    numpy.ndarray
        2D array of shape (num_samples, num_classes) with one-hot encoded labels.
    """
    flat_labels = labels.flatten()                                # Flatten the input array to 1D
    one_hot = np.zeros((flat_labels.size, num_classes), dtype=int)   # Create an array of zeros with shape (num_samples, num_classes)
    one_hot[np.arange(flat_labels.size), flat_labels] = 1         # Set the appropriate indices to 1
    return one_hot
