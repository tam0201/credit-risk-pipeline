from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score

# Abstract base class for machine learning models
class AbstractModel(ABC):
    """
    An abstract base class that defines a standard interface for machine learning models.
    """

    @abstractmethod
    def train_model(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        Parameters:
            X_train: Training feature data.
            y_train: Training target data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Makes predictions using the trained model.

        Parameters:
            X: Data to make predictions on.

        Returns:
            Predictions for the input data.
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves the model to the specified path.

        Parameters:
            path: File path to save the model to.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model from the specified path.

        Parameters:
            path: File path to load the model from.
        """
        pass

# Implementation of a neural network model
class BinaryClassifierNN(nn.Module, AbstractModel):
    """
    A binary classification neural network model suitable for credit default risk prediction.

    Neural networks are powerful function approximators that can model complex non-linear
    relationships, making them ideal for the intricate patterns often found in credit
    default risk data. This model can capture the interactions between various borrower
    attributes and learn from high-dimensional datasets typical in finance.

    Attributes:
        model (nn.Sequential): The sequential model comprising linear layers and activations.

    Methods:
        train_model: Trains the neural network using a DataLoader.
        predict: Predicts binary labels for the input data.
        predict_proba: Predicts class probabilities for the input data.
        save: Saves the model state dictionary to a file.
        load: Loads the model state dictionary from a file.
        forward: Implements the forward pass of the neural network.
    """
    def __init__(self, input_size, hidden_layers, output_size=1):
        """
        Initializes the neural network with one hidden layer and one output neuron with a sigmoid activation.

        Parameters:
            input_size: Number of input features.
            hidden_layers: List containing the number of neurons for each hidden layer.
            output_size: Number of output neurons (defaults to 1 for binary classification).
        """
        super(BinaryClassifierNN, self).__init__()

        layers = []
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(input_size if i == 0 else hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def train_model(self, train_loader, epochs=5, learning_rate=0.001):
        """
        Trains the neural network model with the given training data loader.

        Parameters:
            train_loader: DataLoader containing the training batches.
            epochs: Number of epochs to train for.
            learning_rate: Learning rate for the optimizer.
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train()  # Set the model to training mode
        for epoch in range(epochs):
            for X_train, y_train in train_loader:
                optimizer.zero_grad()
                output = self.model(X_train)
                loss = criterion(output, y_train.float().view(-1, 1))
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def predict(self, X):
        """
        Predicts binary labels for the input data.

        Parameters:
            X: Data to make predictions on.

        Returns:
            Binary predictions for the input data.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            probabilities = self.model(X)
        return probabilities.cpu().round()

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Parameters:
            X: Data to make predictions on.

        Returns:
            Class probabilities for the input data.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            probabilities = self.model(X)
        return probabilities.cpu().numpy()  # Convert to numpy array

    def save(self, path):
        """
        Saves the neural network model state dictionary to the specified path.

        Parameters:
            path: File path to save the model to.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads the neural network model state dictionary from the specified path.

        Parameters:
            path: File path to load the model from.
        """
        self.model.load_state_dict(torch.load(path))

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
            x: Input data forthe forward pass.

        Returns:
            Output after passing through the network (probability for binary classification).
        """
        return self.model(x)

# Implementation of a gradient boosting model
class GradientBoostingBinaryClassifier(AbstractModel):
    """
    A gradient boosting binary classifier using XGBoost, tailored for credit default risk prediction.

    Gradient boosting models like XGBoost are robust and powerful, capable of handling the various
    types of data imperfections (such as outliers and irrelevant features) commonly found in credit
    risk datasets. The model performs well with non-linear relationships and can provide insights
    into feature importance, aiding in the interpretability and understanding of factors that
    contribute to credit risk. XGBoost's regularization features help in preventing overfitting,
    ensuring the model generalizes well to new, unseen data. And it ability to handle missing values
    and outliers reduces the need for extensive data cleaning and feature engineering.

    This is the preffered model for the credit default risk prediction problem.

    Attributes:
        model (XGBClassifier): The XGBoost classifier model.

    Methods:
        train_model: Trains the XGBoost model using provided feature and target data.
        predict: Predicts binary labels for the input data.
        predict_proba: Predicts class probabilities for the input data.
        save: Saves the XGBoost model using pickle.
        load: Loads the XGBoost model using pickle.
    """
    def __init__(self, **kwargs):
        """
        Initializes the XGBoost classifier with the given keyword arguments.

        Parameters:
            **kwargs: Arbitrary keyword arguments that are passed to the XGBClassifier.
        """
        self.model = XGBClassifier(**kwargs)

    def train_model(self, X_train, y_train):
        """
        Trains the XGBoost model using the provided training data.

        Parameters:
            X_train: Training feature data.
            y_train: Training target data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predicts binary labels for the input data using the trained XGBoost model.

        Parameters:
            X: Data to make predictions on.

        Returns:
            Binary predictions for the input data.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data using the trained XGBoost model.

        Parameters:
            X: Data to make predictions on.

        Returns:
            Class probabilities for the input data.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]  # Return probability for the positive class
        else:
            raise NotImplementedError("The underlying model does not support probability predictions.")

    def save(self, path):
        """
        Saves the XGBoost model to the specified path using pickle.

        Parameters:
            path: File path to save the model to.
        """
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path):
        """
        Loads the XGBoost model from the specified path using pickle.

        Parameters:
            path: File path to load the model from.
        """
        self.model = pickle.load(open(path, 'rb'))
