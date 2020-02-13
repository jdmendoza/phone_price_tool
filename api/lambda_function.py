import pickle
import pandas as pd
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
        new_phone = { "model": event["model"], "condition": event["condition"],
        "storage": event["storage"], "carrier": event["carrier"] }

        #Apply encoding and scaling

        new_x = pd.DataFrame.from_dict(new_phone, orient="index").transpose()
        prediction = str(model.predict(new_phone)) #Apply inverse scaling

        return {"body": prediction}

    return {"body": "No parameters"}