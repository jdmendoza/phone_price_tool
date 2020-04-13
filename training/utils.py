import boto3
import pandas as pd
import re
import time
from io import StringIO
from botocore.exceptions import ClientError
import logging


def default_settings():
    params = {
        'region': 'us-east-2',
        'database': 'phones',
        'bucket': 'phone-pricing-models',
        'path': 'temp/athena/output',
        'query': 'SELECT * FROM sold'
    }

    client = boto3.client("s3")
    session = boto3.Session()

    return client, session, params


def athena_query(client, params):
    response = client.start_query_execution(
        QueryString=params["query"],
        QueryExecutionContext={
            'Database': params['database']
        },
        ResultConfiguration={
            'OutputLocation': 's3://' + params['bucket'] + '/' + params['path']
        }
    )
    return response


def athena_to_s3(session, params, date_filter=None):
    if date_filter:
        params["query"] += " WHERE date > date '{}' ".format(date_filter)

    client_athena = session.client('athena', region_name=params["region"])
    execution = athena_query(client_athena, params)
    execution_id = execution['QueryExecutionId']
    state = 'RUNNING'

    while state in ['RUNNING', 'QUEUED']:
        response = client_athena.get_query_execution(QueryExecutionId=execution_id)
        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']

            if state == 'FAILED':
                return False
            elif state == 'SUCCEEDED':
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*\/(.*)', s3_path)[0]
                return filename
        time.sleep(2)

    return False


def pull_data(client, params, filename):
    data = client.get_object(Bucket=params['bucket'], Key=params['path']+'/'+filename)
    data_df = pd.read_csv(StringIO(data['Body'].read().decode('utf-8')))

    return data_df


def save_to_s3(client, params, file_to_save, object_name=None):
    if object_name is None:
        object_name = file_to_save

    try:
        client.upload_file(file_to_save, params['bucket'], object_name)

    except ClientError as e:
        logging.error(e)
        return False

    return True


def preprocess_dataset(dataset):
    df_deduped = dataset.dropna().drop_duplicates().reset_index(drop=True)
    subset = ['color', 'condition', 'date', 'listing_url', 'price',
              'storage', 'url', 'carrier', 'model']
    return df_deduped.drop_duplicates(subset=subset)


if __name__ == "__main__":
    client, session, params = default_settings()
    s3_filename = athena_to_s3(session, params)
    df = pull_data(client, params, s3_filename)
