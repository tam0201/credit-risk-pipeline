import argparse
import yaml
import pandas as pd
import torch
import pickle
import pathlib
import scipy
from sklearn.model_selection import train_test_split

from data_preparation.preprocessor import Preprocessor
from models.model import BinaryClassifierNN, GradientBoostingBinaryClassifier
from evaluation.validator import validate_gbm, validate_nn
from data_preparation.data_loader import DataLoaderFactory

def get_versioned_model_path(config):
    """Generate the versioned model path by replacing placeholders with the actual version."""
    model_path_template = config['model']['model_path']
    return model_path_template.replace("${version}", config['model']['version']).replace("${name}", config['model']['name'])

def load_config(config_path):
    """
    Load the YAML configuration file.
    
    Parameters:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration settings as a dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path, header=1):
    """
    Load the dataset from a given CSV file path.
    
    Parameters:
        file_path (str): Path to the CSV file containing the data.
        header (int): Row number to use as the column names.
    
    Returns:
        DataFrame: Pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path, header=header) # adjust accordingly to data

def preprocess_data(data, preprocessor, config):
    """
    Preprocess the data using the provided Preprocessor object according to the configuration.
    
    This involves dropping the target column from the dataset, applying the preprocessing
    transformations, and then splitting the processed data into training and test sets.
    
    Parameters:
        data (DataFrame): The raw data to preprocess.
        preprocessor (Preprocessor): The Preprocessor object to use for data preprocessing.
        config (dict): Configuration settings.
    
    Returns:
        tuple: A tuple containing the split data (X_train, X_test, y_train, y_test).
    """
    X = data.drop(columns=[config['data']['target_column']])
    y = data[config['data']['target_column']]
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    processed = preprocessor.fit_transform(X_train, y_train)
    
    # Check if the result includes resampled y
    if isinstance(processed, tuple):
        X_train_preprocessed, y_train_resampled = processed
    else:
        X_train_preprocessed = processed
        y_train_resampled = y_train

    # Check if X_preprocessed is a sparse matrix and convert to a DataFrame
    if scipy.sparse.issparse(X_train_preprocessed):
        X_train_preprocessed = pd.DataFrame.sparse.from_spmatrix(
            X_train_preprocessed, columns=preprocessor.get_feature_names_out()
        )
    else:
        X_train_preprocessed = pd.DataFrame(
            X_train_preprocessed, columns=preprocessor.get_feature_names_out()
        )

    # Save train and test data
    processed_data_dir = pathlib.Path(config['data']['processed_data_path'])
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    X_train_preprocessed.to_csv(processed_data_dir / 'X_train.csv', index=False)
    y_train_resampled.to_csv(processed_data_dir / 'y_train.csv', index=False)
    X_test.to_csv(processed_data_dir / 'X_test.csv', index=False)
    y_test.to_csv(processed_data_dir / 'y_test.csv', index=False)
             
    return X_train, X_test, y_train, y_test

def train_nn_model(model, train_loader, config):
    """
    Train the neural network model using the training DataLoader.
    
    Parameters:
        model (BinaryClassifierNN): The neural network model to train.
        train_loader (DataLoader): DataLoader containing the training data batches.
        config (dict): Configuration settings containing the number of epochs and learning rate.
    
    Raises:
        ValueError: If the model type in the configuration is not 'nn'.
    """

    model_type = config['model']['type']
    if model_type == 'nn':
        epochs = config['training']['epochs']
        learning_rate = config['training']['learning_rate']
        model.train_model(train_loader, epochs=epochs, learning_rate=learning_rate)
    else:
        raise ValueError("Model type not recognized in train_model function.")
    

def predict(config, model, X_test=None, y_test=None, **kwargs):
    """
    Generate predictions using the trained model based on the configuration.
    
    The function supports both neural network and gradient boosting models.
    
    Parameters:
        config (dict): Configuration settings specifying the model type and prediction paths.
        model (AbstractModel): The trained model for making predictions.
        X_test (DataFrame, optional): Test feature data for prediction. Required for GBM.
        y_test (Series, optional): Test target data for DataLoader creation. Required for NN.
    
    Raises:
        ValueError: If the model type in the configuration is not recognized.
    """
    if config['model']['type'] == 'nn':
        data_loader = DataLoaderFactory().create_dataloader(X_test, y_test, config['model']['type'])
        model.eval()  
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
    pathlib.Path(config['data']['predict_data_path']).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(config['data']['predict_data_path'], index=False)
    
def validate(config, model, X_test=None, y_test=None, **kwargs):
    """
    Validate the trained model using the test dataset.
    
    The validation method depends on the model type specified in the configuration.
    
    Parameters:
        config (dict): Configuration settings specifying the model type.
        model (AbstractModel): The trained model to validate.
        X_test (DataFrame, optional): Test feature data for validation. Required for GBM.
        y_test (Series, optional): Test target data for DataLoader creation. Required for NN.
    
    Raises:
        ValueError: If the model type in the configuration is not recognized.
    """
    if config['model']['type'] == 'nn':
        data_loader = DataLoaderFactory().create_dataloader(X_test, y_test, config['model']['type'])
        validate_nn(model, data_loader=data_loader)
    elif config['model']['type'] == 'gbm':
        validate_gbm(model, X_test, y_test)
    else:
        raise ValueError("Model type not recognized in validate function.")
    
def main(config_path, stage):
    """
    Main function that orchestrates the machine learning pipeline according to the given stage.
    
    The pipeline comprises several stages: 'preprocess', 'train', 'predict', and 'validate'.
    
    Parameters:
        config_path (str): Path to the configuration file.
        stage (str): The current stage of the pipeline to run.
    """
    config = load_config(config_path)
    versioned_model_path = get_versioned_model_path(config)
    
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
            model = BinaryClassifierNN(input_size=input_size, hidden_layers=hidden_layers)
            train_nn_model(model, train_loader, config)
        elif model_type == 'gbm':
            model = GradientBoostingBinaryClassifier()
            model.train_model(X_train, y_train)
        # Save the trained model
        pathlib.Path(config['model']['model_path']).parent.mkdir(parents=True, exist_ok=True)
        model.save(versioned_model_path)
        
    elif stage == 'predict':
        X_test= load_data(config['data']['test_data_path'] + '/X_test.csv', header=0)
        y_test = load_data(config['data']['test_data_path'] + '/y_test.csv', header=0)
        if config['model']['type'] == 'nn':
            model = BinaryClassifierNN(X_test.shape[1], hidden_layers=config['model']['hidden_layers'])
            model.load(versioned_model_path)
            
        elif config['model']['type'] == 'gbm':
            model = pickle.load(open(versioned_model_path, 'rb'))
        predict(config, model, X_test, y_test)
        
    elif stage == 'validate':
        X_test= load_data(config['data']['test_data_path'] + '/X_test.csv', header=0)
        y_test = load_data(config['data']['test_data_path'] + '/y_test.csv', header=0)
        if config['model']['type'] == 'nn':
            model = BinaryClassifierNN(X_test.shape[1], hidden_layers=config['model']['hidden_layers'])
            model.load(versioned_model_path)
            
        elif config['model']['type'] == 'gbm':
            model = pickle.load(open(versioned_model_path, 'rb'))
        validate(config, model, X_test=X_test, y_test=y_test)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the machine learning pipeline stages based on the provided configuration file.')
    parser.add_argument('--stage', type=str, choices=['preprocess', 'train', 'predict', 'validate'], help='Pipeline stage to run')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config_path, args.stage)