import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

#import wandb
#wandb.init(project="phone_price")

data = pd.read_csv('/Users/jdmendoza/Desktop/phone_price_tool/training/dataset/a7bad4b1-20c1-4049-88c0-a9db77bb56e4.csv')
data = data.dropna().drop_duplicates().reset_index(drop=True)

enc = OneHotEncoder()
categorical_features = ['color', 'condition', 'carrier', 'model']
enc.fit(data[categorical_features])
X = enc.transform(data[categorical_features]).toarray()

y = data.price.values.reshape(1, -1)
X = np.concatenate([X, data.storage.values.reshape(-1, 1)], axis=1)

features_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled = features_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled.T, test_size=0.2, random_state=32)

model = LinearRegression()
model.fit(X_scaled, y_scaled.T)
y_pred = model.predict(X_test)

#wandb.sklearn.plot_regressor(model, X_train, X_test, y_train.reshape(-1,1), y_test.reshape(-1,1), 'LinearRegression')

print("R^2 " + str(model.score(X_test, y_test)))

model_path = 'model_saves/'

pickle.dump(enc, open(model_path + 'one_hot_encoder.plk', 'wb'))
pickle.dump(model, open(model_path + 'model.plk', 'wb'))
pickle.dump(features_scaler, open(model_path + 'features_scaler.plk', 'wb'))
pickle.dump(target_scaler, open(model_path + 'target_scaler.plk', 'wb'))

