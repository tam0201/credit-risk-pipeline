from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        self.preprocessor = None
        self.init_preprocessor()

    def init_preprocessor(self):
        """Initialize preprocessing pipelines for both numerical and categorical data."""
        # Pipeline for numeric features
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline for categorical features
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Full preprocessing pipeline
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, self.num_features),
            ('cat', cat_pipeline, self.cat_features)
        ])

    # Assuming you have already defined num_pipeline, cat_pipeline, and self.preprocessor

    def fit_transform(self, X, y=None):
        """Fit the preprocessing pipeline on the data, transform it, and then apply SMOTE."""
        if y is not None:
            print("SMOTE")
            # First fit and transform the data using the preprocessing pipeline
            X_preprocessed = self.preprocessor.fit_transform(X)
            
            # Then apply SMOTE to the preprocessed data
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)
            
            # Save the fitted preprocessor and SMOTE for future calls to transform()
            self.fitted_preprocessor = self.preprocessor
            self.fitted_smote = smote
            return X_resampled, y_resampled
        else:
            return self.preprocessor.fit_transform(X)

    def transform(self, X):
        """Transform the data using the fitted preprocessing pipeline."""
        # Ensure transformation is performed with the fitted pipeline that includes SMOTE
        if hasattr(self, 'fitted_pipeline'):
            # Use the pipeline that was fitted during fit_transform (which includes SMOTE and preprocessing)
            return self.fitted_pipeline.named_steps['preprocessor'].transform(X)
        else:
            # If the pipeline was not fitted with SMOTE (i.e., fit_transform was not called with y), then just transform
            return self.preprocessor.transform(X)

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline on the data."""
        if y is not None:
            # Create a pipeline that includes SMOTE followed by the ColumnTransformer
            pipeline = ImbPipeline(steps=[('smote', SMOTE()), ('preprocessor', self.preprocessor)])
            pipeline.fit_resample(X, y)
            # Save the fitted pipeline for future calls to transform()
            self.fitted_pipeline = pipeline
        else:
            self.preprocessor.fit(X)
    
    def get_feature_names_out(self):
        """Get output feature names for transformation after preprocessing."""
        # If your preprocessor is a ColumnTransformer, you can call get_feature_names_out directly
        return self.preprocessor.get_feature_names_out()