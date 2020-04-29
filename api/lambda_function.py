import pickle
import numpy as np
import json
import boto3

s3 = boto3.resource('s3')


def get_pkl(fname):
    return pickle.loads(s3.Bucket("phone-pricing-models").Object(fname).get()['Body'].read())


enc = get_pkl('one_hot_encoder.pkl')
model = get_pkl('model.pkl')
features_scaler = get_pkl('features_scaler.pkl')
target_scaler = get_pkl('target_scaler.pkl')


def lambda_handler(event, context):
    if "body" in event:
        event = event["body"]

        if event is not None:
            event = json.loads(event)

        else:
            event = {}

    if "model" in event:
        new_phone = {"model": event["model"], "condition": event["condition"],
                     "storage": event["storage"], "carrier": event["carrier"], "color": event["color"]}

        x_encoded = enc.transform(np.array([new_phone['color'], new_phone['condition'],
                                            new_phone['carrier'], new_phone['model']]).reshape(1, -1))

        x_concat = np.concatenate([x_encoded.toarray(), np.array(new_phone['storage']).reshape(1, 1)], axis=1)

        x_scaled = features_scaler.transform(x_concat)

        prediction = target_scaler.inverse_transform(model.predict(x_scaled).reshape(1, 1))

        return {"body": str(round(prediction[0][0], 2))}

    return {"body": "No parameters"}
