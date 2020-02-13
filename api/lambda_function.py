import pickle
import pandas as pd
import numpy as np
import json

pickle_path = '' #add path

enc = pickle.load(open(pickle_path + 'one_hot_encoder.plk', 'rb'))
model = pickle.load(open(pickle_path + 'model.pickle', 'rb'))
features_scaler = pickle.load(open(pickle_path + 'features_scaler.plk', 'rb'))
target_scaler = pickle.load(open(pickle_path + 'target_scaler.plk', 'rb'))


def lambda_handler(event, context):
    if "body" in event:
        event = event["body"]

        if event is not None:
            event = json.loads(event)

        else:
            event = {}

    if "model" in event:
        new_phone = {"model": event["model"], "condition": event["condition"],
                     "storage": event["storage"], "carrier": event["carrier"]}

        x_encoded = enc.transform(np.array([new_phone['color'], new_phone['condition'],
                                            new_phone['carrier'], new_phone['model']]).reshape(1, -1))

        x_scaled = np.concatenate([x_encoded.toarray(), np.array(new_phone['storage']).reshape(1, 1)], axis=1)

        prediction = str(target_scaler.inverse_transform(model.predict(x_scaled)))

        return {"body": prediction}

    return {"body": "No parameters"}
