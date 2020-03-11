import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import wandb
import sys
import logging

import pathlib
path = str(pathlib.Path().absolute().parent)
sys.path.insert(0, path)

from training.utils import default_settings, athena_to_s3, pull_data, preprocess_dataset

MODEL_PATH = 'training/model_saves/'


def process_data(data):

    enc = OneHotEncoder(drop='first')
    categorical_features = ['color', 'condition', 'carrier', 'model']
    enc.fit(data[categorical_features])
    x_prescaled = enc.transform(data[categorical_features]).toarray()

    y_prescaled = data.price.values.reshape(-1, 1)
    x_prescaled = np.concatenate([x_prescaled, data.storage.values.reshape(-1, 1)], axis=1)

    features_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    x_scaled = features_scaler.fit_transform(x_prescaled)
    y_scaled = target_scaler.fit_transform(y_prescaled)

    #pickle.dump(enc, open(MODEL_PATH + 'one_hot_encoder.pkl', 'wb'))
    #pickle.dump(features_scaler, open(MODEL_PATH + 'features_scaler.pkl', 'wb'))
    #pickle.dump(target_scaler, open(MODEL_PATH + 'target_scaler.pkl', 'wb'))

    return x_scaled, y_scaled


if __name__ == '__main__':

    wandb.init(project="phone_price")

    client, session, params = default_settings()
    s3_filename = athena_to_s3(session, params)
    df_all = pull_data(client, params, s3_filename)
    data = preprocess_dataset(df_all)

    X, y = process_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    model = LinearRegression()
    model.fit(X, y)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    #pickle.dump(model, open(MODEL_PATH + 'model.pkl', 'wb'))

    wandb.log({"Train MSE": mean_squared_error(y_train, y_train_pred), "Test MSE": mean_squared_error(y_test, y_pred)})
