import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import boto3
import datetime
import sys

import pathlib
path = str(pathlib.Path().absolute().parent)
sys.path.insert(0, path)

from training.utils import default_settings, athena_to_s3, pull_data, preprocess_dataset, process_data
from training.linear_model import retrain_model


def get_pkl(fname):
    return pickle.loads(s3.Bucket("phone-pricing-models").Object(fname).get()['Body'].read())


class ModelTester:

    def __init__(self, model_under_test, input_data, target):
        self.metrics_table = boto3.resource('dynamodb').Table('model_metrics')

        self.model_under_test = model_under_test
        self.input_data = input_data
        self.y_true = target
        self.y_pred = self.model_under_test.predict(input_data)

        # Metrics to compare and save
        self.prev_mse = -1
        self.mse = -1

    def calculate_metrics(self):
        self.mse = mean_squared_error(self.y_true, self.y_pred)

    def save_metrics(self):
        store_data_in_table(table=self.metrics_table, name='model_metrics', metric='mse', value=self.mse)

    def get_prev_metrics(self):
        # pull the most recent mse
        self.prev_mse = -1

    def run(self):
        self.get_prev_metrics()
        self.calculate_metrics()
        self.save_metrics()

        check_mse = self.mse < self.prev_mse

        if all([check_mse]):
            retrain_model(self.input_data, self.y_true)


def store_data_in_table(table, name, metric, value):
    now = datetime.datetime.now()

    try:
        table.put_item(
           Item={
                'id': now.strftime("%Y%m%d%H%M%S"),
                'name': name,
                'metric': metric,
                'value': value,
            }
        )

        return True

    except:

        return False


if __name__ == "__main__":

    s3 = boto3.resource('s3')

    enc = get_pkl('one_hot_encoder.pkl')
    model = get_pkl('model.pkl')
    features_scaler = get_pkl('features_scaler.pkl')
    target_scaler = get_pkl('target_scaler.pkl')

    client, session, params = default_settings()
    s3_filename = athena_to_s3(session, params)
    df_all = pull_data(client, params, s3_filename)

    data = preprocess_dataset(df_all)
    test_data = data[data['date'] > (datetime.datetime.now() - datetime.timedelta(days=14))]
    X, y = process_data(test_data)

    Tester = ModelTester(model, X, y)
    Tester.run()
