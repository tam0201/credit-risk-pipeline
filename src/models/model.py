from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score


# Abstract base class
class AbstractModel(ABC):

    @abstractmethod
    def train_model(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

class NeuralNetworkModel(nn.Module, AbstractModel):
    def __init__(self, input_size, hidden_layers, output_size=1):
        super(NeuralNetworkModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], output_size),
            nn.Sigmoid()
        )

    def train_model(self, train_loader, epochs=5, learning_rate=0.001):
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
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            probabilities = self.model(X)
        return probabilities.cpu().round()

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            probabilities = self.model(X)
        return probabilities.cpu().numpy()  # Convert to numpy array

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def forward(self, x):
        # Forward pass through the network
        return self.model(x)
    
# Gradient Boosting Model
class GradientBoostingModel(AbstractModel):

    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("The underlying model does not support probability predictions.")
        
    def save(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))
    