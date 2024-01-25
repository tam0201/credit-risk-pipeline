import argparse
import yaml
import pandas as pd
import torch
import pickle
import pathlib
import scipy
from sklearn.model_selection import train_test_split

from data_preparation.preprocessor import Preprocessor
from models.model import GradientBoostingModel, NeuralNetworkModel
from evaluation.validator import validate_gbm, validate_nn
from data_preparation.data_loader import DataLoaderFactory

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path, header=1):
    """Load the dataset from a given CSV file path."""
    return pd.read_csv(file_path, header=header) # adjust accordingly to data

def preprocess_data(data, preprocessor, config):
    """Preprocess the data using the provided Preprocessor object."""
    X = data.drop(columns=[config['data']['target_column']])
    y = data[config['data']['target_column']]
    
    # If y is not None, fit_transform will return a tuple (X_resampled, y_resampled)
    processed = preprocessor.fit_transform(X, y)
    
    # Check if the result includes resampled y
    if isinstance(processed, tuple):
        X_preprocessed, y_resampled = processed
    else:
        X_preprocessed = processed
        y_resampled = y

    # Check if X_preprocessed is a sparse matrix and convert to a DataFrame if necessary
    if scipy.sparse.issparse(X_preprocessed):
        X_preprocessed = pd.DataFrame.sparse.from_spmatrix(
            X_preprocessed, columns=preprocessor.get_feature_names_out()
        )
    else:
        X_preprocessed = pd.DataFrame(
            X_preprocessed, columns=preprocessor.get_feature_names_out()
        )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y_resampled, test_size=0.2, random_state=42
    )
    
    # Save train and test data
    processed_data_dir = pathlib.Path(config['data']['processed_data_path'])
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_data_dir / 'X_train.csv', index=False)
    y_train.to_csv(processed_data_dir / 'y_train.csv', index=False)
    X_test.to_csv(processed_data_dir / 'X_test.csv', index=False)
    y_test.to_csv(processed_data_dir / 'y_test.csv', index=False)
             
    return X_train, X_test, y_train, y_test

def train_nn_model(model, train_loader, config):
    """Train the model using the training set."""
    model_type = config['model']['type']
    if model_type == 'nn':
        epochs = config['training']['epochs']
        learning_rate = config['training']['learning_rate']
        model.train_model(train_loader, epochs=epochs, learning_rate=learning_rate)
    else:
        raise ValueError("Model type not recognized in train_model function.")
    

def predict(config, model, X_test=None, y_test=None, **kwargs):
    """Generate predictions using the trained model."""
    if config['model']['type'] == 'nn':
        data_loader = DataLoaderFactory().create_dataloader(X_test, y_test, config['model']['type'])
        model.eval()  # Set the model to evaluation mode
        predictions = []

        with torch.no_grad():
            for X_batch, _ in data_loader:
                outputs = model(X_batch)
                predictions.append(outputs)

        # Convert list of tensors to a single tensor
        predictions = torch.cat(predictions, dim=0)
    elif config['model']['type'] == 'gbm':
        predictions = model.predict_proba(X_test)
    else: 
        raise ValueError("Model type not recognized in predict function.")
    print(predictions)
    # Save the predictions to a CSV file
    pathlib.Path(config['data']['predict_data_path']).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(config['data']['predict_data_path'], index=False)
    
def validate(config, model, X_test=None, y_test=None, **kwargs):
    """Validate the trained model using the test dataset."""
    if config['model']['type'] == 'nn':
        data_loader = DataLoaderFactory().create_dataloader(X_test, y_test, config['model']['type'])
        validate_nn(model, data_loader=data_loader)
    elif config['model']['type'] == 'gbm':
        validate_gbm(model, X_test, y_test)
    else:
        raise ValueError("Model type not recognized in validate function.")
    
def main(config_path, stage):
    config = load_config(config_path)
    
    if stage == 'preprocess':
        data = load_data(config['data']['raw_data_path'], header=1)
        num_features = config['features']['numerical']
        cat_features = config['features']['categorical']
        preprocessor = Preprocessor(num_features, cat_features)
        preprocess_data(data, preprocessor, config)
    elif stage == 'train':
        X_train = load_data(config['data']['processed_data_path'] + '/X_train.csv', header=0)
        y_train = load_data(config['data']['processed_data_path'] + '/y_train.csv', header=0)
        dataloader_factory = DataLoaderFactory(batch_size=config['training']['batch_size'])
        train_loader = dataloader_factory.create_dataloader(X_train, y_train, config['model']['type'])
        
        model_type = config['model']['type']
        if model_type == 'nn':
            input_size = X_train.shape[1]
            hidden_layers = config['model']['hidden_layers']
            model = NeuralNetworkModel(input_size=input_size, hidden_layers=hidden_layers)
            train_nn_model(model, train_loader, config)
        elif model_type == 'gbm':
            model = GradientBoostingModel()
            model.train_model(X_train, y_train)
        # Save the trained model
        pathlib.Path(config['model']['model_path']).parent.mkdir(parents=True, exist_ok=True)
        model.save(config['model']['model_path'])
        
    elif stage == 'predict':
        X_test= load_data(config['data']['test_data_path'] + '/X_test.csv', header=0)
        y_test = load_data(config['data']['test_data_path'] + '/y_test.csv', header=0)
        if config['model']['type'] == 'nn':
            model = NeuralNetworkModel(X_test.shape[1], hidden_layers=config['model']['hidden_layers'])
            model.load(config['model']['model_path'])
            
        elif config['model']['type'] == 'gbm':
            model = pickle.load(open(config['model']['model_path'], 'rb'))
        predict(config, model, X_test, y_test)
        
    elif stage == 'validate':
        # Assuming that test data is already preprocessed and saved
        X_test= load_data(config['data']['test_data_path'] + '/X_test.csv', header=0)
        y_test = load_data(config['data']['test_data_path'] + '/y_test.csv', header=0)
        if config['model']['type'] == 'nn':
            model = NeuralNetworkModel(X_test.shape[1], hidden_layers=config['model']['hidden_layers'])
            model.load(config['model']['model_path'])
            
        elif config['model']['type'] == 'gbm':
            model = pickle.load(open(config['model']['model_path'], 'rb'))
        validate(config, model, X_test=X_test, y_test=y_test)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the machine learning pipeline stages based on the provided configuration file.')
    parser.add_argument('--stage', type=str, choices=['preprocess', 'train', 'predict', 'validate'], help='Pipeline stage to run')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config_path, args.stage)