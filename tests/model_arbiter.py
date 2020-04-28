import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import boto3
import datetime
from decimal import Decimal
import sys
import logging
logging.basicConfig(level=logging.INFO)

from training.utils import default_settings, athena_to_s3, pull_data, preprocess_dataset
from training.linear_model import process_data, retrain_model


def get_pkl(fname):
    return pickle.loads(s3.Bucket("phone-pricing-models").Object(fname).get()['Body'].read())


class ModelTester:

    def __init__(self, lookback_days=14):
        logging.info('Initializing ModelTester')

        self.model_under_test = get_pkl('model.pkl')
        print(self.model_under_test.coef_, self.model_under_test.intercept_)
        self.lookback_days = lookback_days

        # Pull most recent samples
        self.raw_data = None
        self.get_test_data()

        # Get data model ready
        self.input_data = None
        self.y_true = None
        self.transform_data()
        self.y_pred = self.model_under_test.predict(self.input_data)

        # Metrics to compare and save
        self.metrics_table = boto3.resource('dynamodb').Table('model_metrics')
        self.prev_mse = -1
        self.mse = -1

    def get_test_data(self):
        lookback_date = (datetime.datetime.now() - datetime.timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        s3_filename = athena_to_s3(session, params, lookback_date)
        self.raw_data = pull_data(client, params, s3_filename)

    def transform_data(self):
        enc = get_pkl('one_hot_encoder.pkl')
        features_scaler = get_pkl('features_scaler.pkl')
        target_scaler = get_pkl('target_scaler.pkl')

        categorical_features = ['color', 'condition', 'carrier', 'model']

        x_prescaled = enc.transform(self.raw_data[categorical_features]).toarray()
        x_prescaled = np.concatenate([x_prescaled, self.raw_data.storage.values.reshape(-1, 1)], axis=1)
        y_prescaled = self.raw_data.price.values.reshape(-1, 1)

        x_scaled = features_scaler.transform(x_prescaled)
        y_scaled = target_scaler.transform(y_prescaled)

        self.input_data = x_scaled
        self.y_true = y_scaled

    def calculate_metrics(self):
        self.mse = mean_squared_error(self.y_true, self.y_pred)

    def save_metrics(self, metric, value):
        logging.info('Saving to DynamoDB')
        store_data_in_table(table=self.metrics_table, name='model_metrics', metric=metric, value=value)

    def get_prev_metrics(self):
        # pull the most recent mse, currently hardcoded for mse
        response = self.metrics_table.scan()
        self.prev_mse = sorted([(x['id'], x['metric'], x['value']) for x in response['Items']])[-1][-1]
        logging.info('previous_mse = {}'.format(self.prev_mse))

    def run(self):
        self.get_prev_metrics()
        self.calculate_metrics()

        check_mse = True  # self.mse < self.prev_mse
        logging.info('CheckMSE =  {}, old mse is {}, and current mse is {}'.format(check_mse, self.prev_mse, self.mse))

        if all([check_mse]):
            logging.info('Retraining model')
            retrain_model(self.raw_data)
            self.save_metrics('mse', self.mse)
            # compare with new retrained model


def store_data_in_table(table, name, metric, value):
    now = datetime.datetime.now()

    try:
        table.put_item(
           Item={
                'id': now.strftime("%Y%m%d%H%M%S"),
                'name': name,
                'metric': metric,
                'value': Decimal(str(value)),
            }
        )

        return True

    except:

        return False


if __name__ == "__main__":

    s3 = boto3.resource('s3')
    client, session, params = default_settings()

    Tester = ModelTester(lookback_days=30)
    Tester.run()
