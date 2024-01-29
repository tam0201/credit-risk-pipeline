from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessor:
    """
    Preprocessor class for handling preprocessing of numerical and categorical data.
    
    This class creates a preprocessing pipeline that includes imputation and scaling
    for numerical features and imputation and one-hot encoding for categorical features.
    It also incorporates the SMOTE technique to handle class imbalance in the dataset.
    
    Attributes:
        num_features (list): List of names of numerical features to be processed.
        cat_features (list): List of names of categorical features to be processed.
        preprocessor (ColumnTransformer): A sklearn ColumnTransformer that contains
                                          the preprocessing pipelines for both numerical
                                          and categorical data.
    """
    
    def __init__(self, num_features, cat_features):
        """
        Initializes the Preprocessor with lists of numerical and categorical features.
        
        Parameters:
            num_features (list): List of numerical feature column names.
            cat_features (list): List of categorical feature column names.
        """
        self.num_features = num_features  # List of numerical feature column names
        self.cat_features = cat_features  # List of categorical feature column names
        self.preprocessor = None
        self.init_preprocessor()

    def init_preprocessor(self):
        """
        Initializes the preprocessing pipelines for numerical and categorical features.
        
        For numerical features, median imputation is used to handle missing values due to
        its robustness against outliers. Outliers can significantly affect the mean, but the median
        remains unaffected, making it a better choice for imputation in many datasets.
        
        Standard scaling is applied to numerical features to normalize the data, ensuring that each
        feature contributes equally to the distance computations in many machine learning algorithms,
        such as k-NN, SVMs, and neural networks. This is crucial since features on larger scales can
        disproportionately influence the model's learning process.
        
        For categorical features, the most frequent value is used to impute missing data. This is a
        simple and effective strategy that assumes the mode (most frequent value) is a good estimate
        for the missing category, based on its prevalence in the data.
        
        One-hot encoding is applied to transform categorical variables into a format that can be
        provided to machine learning algorithms. It encodes categorical data into binary vectors,
        ensuring that the model does not attribute any inherent order to the categories. The
        `handle_unknown='ignore'` option allows the encoder to handle unseen categories during
        training by ignoring them and not throwing an error, which enhances the robustness of the model.
        
        The numerical and categorical pipelines are combined into a full preprocessing pipeline using
        a `ColumnTransformer`. This allows parallel processing of different feature types and is
        particularly useful for datasets with heterogeneous data types.
        """
        # Strategy for numeric features:
        # 1. Use median imputation for missing values as it's robust to outliers.
        # 2. Scale features to standardize them, which is important for many ML models.
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with the median
            ('scaler', StandardScaler())  # Scale features to have 0 mean and unit variance
        ])

        # Strategy for categorical features:
        # 1. Use the most frequent value to impute missing values, a common practice for categorical data.
        # 2. One-hot encode categorical variables; unknown categories will be ignored.
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode with ignore strategy for unseen labels during training
        ])

        # Combine the numeric and categorical pipelines into a full preprocessing pipeline
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, self.num_features),
            ('cat', cat_pipeline, self.cat_features)
        ])

    def fit_transform(self, X, y=None):
        """
        Fits the preprocessing pipeline to the data and transforms it, with optional handling of class imbalance.
        
        This method first applies the predefined preprocessing steps to the feature data, which includes
        imputation and scaling for numerical features, as well as imputation and one-hot encoding for
        categorical features. These steps prepare the data for machine learning models by addressing
        missing values and standardizing the feature scale.
        
        If the target array 'y' is provided and class imbalance is detected, the SMOTE (Synthetic Minority
        Over-sampling Technique) algorithm is employed. SMOTE works by creating synthetic samples from
        the minority class, which helps to balance the class distribution and improve the performance of
        classifiers on imbalanced datasets.
        
        It is crucial that SMOTE is applied only after preprocessing to prevent data leakage and ensure
        that the synthetic samples are created in the context of the preprocessed feature space. This
        method ensures that the same preprocessing steps are applied consistently to both the original
        and synthetic samples, maintaining the integrity of the feature space.
        
        Parameters:
            X (array-like): Feature data to be preprocessed.
            y (array-like, optional): Target data to be used for SMOTE. Default is None.
        
        Returns:
            Tuple[array-like, array-like]: The transformed feature data and target data after applying SMOTE
                                        if 'y' is provided. Otherwise, just the transformed feature data.
        """
        # Fit the preprocessing pipeline to the feature data and transform it
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        if y is not None:
            # If target data is provided, apply SMOTE to address class imbalance
            # SMOTE should only be applied on the training set to avoid introducing bias
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)
            
            # Store the fitted preprocessor and SMOTE for consistent transformation of new data
            self.fitted_preprocessor = self.preprocessor
            self.fitted_smote = smote
            
            # Return the resampled feature data and target data
            return X_resampled, y_resampled
        else:
            # If no target data is provided, return only the transformed feature data
            return X_preprocessed

    def transform(self, X):
        """
        Transforms the data using the fitted preprocessing pipeline.
        
        Should be used for new data that follows the same distribution as the training data.
        
        Parameters:
            X (array-like): Feature data to be transformed.
        
        Returns:
            array-like: Transformed feature data.
        """
        # This method transforms the data using the already fitted preprocessing pipeline.
        # It should be used on new data that follows the same distribution as the training data.
        if hasattr(self, 'fitted_pipeline'):
            # If a pipeline with SMOTE has been fitted, use it for transformations
            return self.fitted_pipeline.named_steps['preprocessor'].transform(X)
        else:
            # If SMOTE was not applied, use the standard preprocessing pipeline
            return self.preprocessor.transform(X)

    def fit(self, X, y=None):
        """
        Fits the preprocessing pipeline or a pipeline with SMOTE included to the data.
        
        Parameters:
            X (array-like): Feature data to be preprocessed.
            y (array-like, optional): Target data for SMOTE if required. Default is None.
        """
        # Fit the preprocessing pipeline or a pipeline including SMOTE depending on whether 'y' is provided.
        if y is not None:
            # Create and fit a pipeline with SMOTE followed by preprocessing steps for future transformation
            pipeline = ImbPipeline(steps=[('smote', SMOTE()), ('preprocessor', self.preprocessor)])
            pipeline.fit_resample(X, y)
            self.fitted_pipeline = pipeline
        else:
            # If 'y' is not provided, only fit the preprocessing pipeline
            self.preprocessor.fit(X)
        
    def get_feature_names_out(self):
        """
        Retrieves the output feature names after transformation.
        
        This is particularly useful after one-hot encoding categorical variables, as it can
        increase the number of features and change their names.
        
        Returns:
            array-like: Feature names after transformation.
        """
        # Retrieve the feature names output by the preprocessing pipeline.
        # This is useful for understanding the resulting features, especially after one-hot encoding.
        return self.preprocessor.get_feature_names_out()