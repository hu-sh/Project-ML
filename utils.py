import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch

def load_monk_data(file_path):
    columns = ["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    df = pd.read_csv(file_path, sep=' ', names=columns, skipinitialspace=True)
    df = df.drop(columns=["id"], errors='ignore')
    
    y = df["class"].values
    X_raw = df.drop(columns=["class"])
    
    return X_raw, y

def load_cup_data(file_path):
    df = pd.read_csv(file_path, comment='#', header=None)
    
    df = df.iloc[:, 1:] 
    
    X = df.iloc[:, 0:12].values
    
    y = df.iloc[:, 12:].values
    
    return X, y


def get_encoder(X_train):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train)
    return encoder
