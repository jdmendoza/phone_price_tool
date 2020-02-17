import requests
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    sample_phone = {"color": "White",  "condition": "Good", "storage": 128,
                  "carrier": "att", "model": "apple-iphone-8-plus"}

    result = requests.post("https://uoa1yghdj0.execute-api.us-east-2.amazonaws.com/default/price_predict",
                           json=sample_phone)

    if result.status_code == 200:
        logging.info("Successful, Sent: {} \nReceived: {}".format(str(sample_phone), result.text))

    else:
        logging.error("Error")