from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from io import BytesIO
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report

file_id = "1ZY9Qv5nmDJ0yzffr5qCdHPrWMfbiBf5t"
download_url = f"https://drive.google.com/uc?id={file_id}"

df = pd.read_csv(download_url)

# Data overview function
def data_overview(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print('List of columns:', df.columns)
    print('\nShape of data:', df.shape)
    print('\nData info:', df.info())
    print('\nFive-point summary:', df.describe().T)
    print('\nMissing values:', df.isna().sum())
    print('\nNull values:', df.isnull().sum())
    print('\nDuplicated records:', df.duplicated().sum())
    return df

df = data_overview(df)

# Custom transformers
class NullChecker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.null_columns_ = X.columns[X.isnull().any()].tolist()
        return self
    
    def transform(self, X):
        return X.fillna(0)

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.le = LabelEncoder()
        self.obj_col_list = X.select_dtypes(include=['object']).columns.tolist()
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in self.obj_col_list:
            X_encoded[col] = self.le.fit_transform(X[col])
        return X_encoded

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = StandardScaler()
        self.num_col_list = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.scaler.fit(X[self.num_col_list])
        return self
    
    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.num_col_list] = self.scaler.transform(X[self.num_col_list])
        return X_scaled

class DataSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, target_feature, test_size=0.25, random_state=1):
        self.target_feature = target_feature
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        y = X[self.target_feature]
        X = X.drop(columns=[self.target_feature])
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

class ModelTrainer(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, criterion='gini', bootstrap=True, model='RandomForest'):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            bootstrap=bootstrap
        ) if model == 'RandomForest'

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class ModelSaver(BaseEstimator, TransformerMixin):
    def __init__(self, filename):
        self.filename = filename
    
    def fit(self, X, y):
        return self
    
    def transform(self, X, y):
        with open(self.filename, "wb") as file:
            pkl.dump(y, file)
        return X

# Pipeline setup
pipeline = Pipeline([
    ('null_checker', NullChecker()),
    ('label_encoder', LabelEncoderTransformer()),
    ('scaling', StandardScalerTransformer()),
    ('splitter', DataSplitter(target_feature='fertilizer_name')),
    ('model_trainer', ModelTrainer(n_estimators=50, max_depth=10, 
                                   min_samples_split=2, min_samples_leaf=1, 
                                   criterion='gini', bootstrap=True)),
    ('model_saver', ModelSaver(filename='final_rf_model.pkl'))
])

# Fitting and transforming the data through the pipeline
df_transformed = pipeline.named_steps['null_checker'].transform(df)
df_transformed = pipeline.named_steps['label_encoder'].transform(df_transformed)
df_transformed = pipeline.named_steps['scaling'].transform(df_transformed)

X_train, X_test, y_train, y_test = pipeline.named_steps['splitter'].transform(df_transformed)
pipeline.named_steps['model_trainer'].fit(X_train, y_train)
pipeline.named_steps['model_saver'].transform(X_test, pipeline.named_steps['model_trainer'].model)

y_pred = pipeline.named_steps['model_trainer'].predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
