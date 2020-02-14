import pickle
import pandas as pd
import numpy as np
import json

pickle_path = 'training/model_saves/'

enc = pickle.load(open(pickle_path + 'one_hot_encoder.pkl', 'rb'))
model = pickle.load(open(pickle_path + 'model.pkl', 'rb'))
features_scaler = pickle.load(open(pickle_path + 'features_scaler.pkl', 'rb'))
target_scaler = pickle.load(open(pickle_path + 'target_scaler.pkl', 'rb'))


def lambda_handler(event, context):
    if "body" in event:
        event = event["body"]

        if event is not None:
            event = json.loads(event)
            print(event)

        else:
            event = {}

    if "model" in event:
        new_phone = {"model": event["model"], "condition": event["condition"],
                     "storage": event["storage"], "carrier": event["carrier"], "color": event["color"]}

        x_encoded = enc.transform(np.array([new_phone['color'], new_phone['condition'],
                                            new_phone['carrier'], new_phone['model']]).reshape(1, -1))

        x_scaled = np.concatenate([x_encoded.toarray(), np.array(new_phone['storage']).reshape(1, 1)], axis=1)

        prediction = target_scaler.inverse_transform(model.predict(x_scaled))

        return {"body": str(prediction[0][0])}

    return {"body": "No parameters"}


if __name__ == '__main__':

    event = {"body": json.dumps({'color': 'White', 'condition': 'Good', 'storage': 128,
                                 'carrier': 'att', 'model': 'apple-iphone-8-plus'})}
    print(lambda_handler(event, None))
