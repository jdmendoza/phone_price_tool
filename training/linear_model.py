import numpy as np
import boto3
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle
import wandb
import logging

from training.utils import default_settings, athena_to_s3, pull_data, preprocess_dataset, save_to_s3

client, session, params = default_settings()
s3 = boto3.resource('s3')


def get_pkl(fname):
    return pickle.loads(s3.Bucket("phone-pricing-models").Object(fname).get()['Body'].read())


def process_data(data, dry_run=True):

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

    if not dry_run:
        pickle.dump(enc, open('one_hot_encoder.pkl', 'wb'))
        pickle.dump(features_scaler, open('features_scaler.pkl', 'wb'))
        pickle.dump(target_scaler, open('target_scaler.pkl', 'wb'))

        save_to_s3(client, params, 'one_hot_encoder.pkl')
        save_to_s3(client, params, 'features_scaler.pkl')
        save_to_s3(client, params, 'target_scaler.pkl')

    return x_scaled, y_scaled


def retrain_model(data):
    X, y = process_data(data, dry_run=False)

    model_init = Lasso()
    params_ = {'alpha': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.005, 0.0001, 0]}
    clf = GridSearchCV(model_init, params_, 'neg_mean_squared_error')
    clf.fit(X, y)

    model_to_upload = Lasso(alpha=clf.best_params_['alpha'])
    model_to_upload.fit(X, y)

    pickle.dump(model_to_upload, open('model.pkl', 'wb'))
    save_to_s3(client, params, 'model.pkl')

    return clf.best_params_['alpha']


if __name__ == '__main__':

    wandb.init(project="phone_price")

    s3_filename = athena_to_s3(session, params)
    df_all = pull_data(client, params, s3_filename)
    data = preprocess_dataset(df_all)

    X, y = process_data(data, dry_run=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    model_init = Lasso()
    params_ = {'alpha': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.005, 0.0001, 0]}
    clf = GridSearchCV(model_init, params_, 'neg_mean_squared_error')
    clf.fit(X_train, y_train)

    model = Lasso(alpha=clf.best_params_['alpha'])
    model.fit(X, y)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    pickle.dump(model, open('model.pkl', 'wb'))
    save_to_s3(client, params, 'model.pkl')

    wandb.log({"Train MSE": mean_squared_error(y_train, y_train_pred), "Test MSE": mean_squared_error(y_test, y_pred)})
