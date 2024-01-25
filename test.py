from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

import pandas as pd

# Create a dummy classification dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=5, n_clusters_per_class=1, n_samples=100, random_state=10)

# Convert to a DataFrame to have column names
X = pd.DataFrame(X, columns=['num1', 'num2', 'num3', 'cat1', 'cat2'])
y = pd.Series(y)

# Define numerical and categorical features
num_features = ['num1', 'num2', 'num3']
cat_features = ['cat1', 'cat2']

# Define ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

# Define ImbPipeline with SMOTE and the preprocessor
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
preprocessed_X = preprocessor.fit_transform(X_resampled, y_resampled)
print(X_resampled.shape, y_resampled.shape, preprocessed_X.shape)