import os

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import common.Config as cfg
from preprocessing.ParameterGenerator import ParameterGenerator


def train():
    pg = ParameterGenerator()
    x, y = pg.convert_to_np()

    mms = MinMaxScaler()
    mms.fit(x)

    x = mms.transform(x)

    shuffle_index = np.random.permutation(x.shape[0])
    x, y = x[shuffle_index], y[shuffle_index]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    SGDClsr = SGDClassifier(random_state=42)
    SGDClsr.fit(x_train, y_train)

    print('The score is {}'.format(SGDClsr.score(x_test, y_test)))
    joblib.dump(SGDClsr, cfg.OUTPUT_MODEL_SDGCLASSIFIER)
    joblib.dump(mms, cfg.OUTPUT_MODEL_SDGCLASSIFIER_SCALER)


# input should be a np.array [[]].
def predict(x):
    if not os.path.exists(cfg.OUTPUT_MODEL_SDGCLASSIFIER):
        print('The model is not found at {}'.format(cfg.OUTPUT_MODEL_SDGCLASSIFIER))
    elif not os.path.exists(cfg.OUTPUT_MODEL_SDGCLASSIFIER_SCALER):
        print('The scaler is not found at {}'.format(cfg.OUTPUT_MODEL_SDGCLASSIFIER_SCALER))
    else:

        mms = joblib.load(cfg.OUTPUT_MODEL_SDGCLASSIFIER_SCALER)
        SGDClsr = joblib.load(cfg.OUTPUT_MODEL_SDGCLASSIFIER)

        x = mms.transform(x)

        result = SGDClsr.predict(x)

        print(result)


# train()

pg = ParameterGenerator()
x = np.array([pg.convert_predict_input_to_np('John Doll')])

predict(x)
