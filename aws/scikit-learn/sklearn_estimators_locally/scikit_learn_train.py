"""
File: aws_basic_scikitlearn_model
Date: 2/13/2020
Author: Quinn Lanners
Description: Basic training script used to train a scikit learn model on the IRIS training set and practice deploying in various ways to AWS Sagemaker.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Put the data pre-processing and training in the __main__ block for Sagemaker to read correctly
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    #Part of file to load data and perform preprocessing
    col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = pd.read_csv(args.train)

    train_X = data[[c for c in data.columns if c != 'label']]
    train_Y = data[['label']]
    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    encoder = LabelEncoder().fit(train_Y)
    train_Y_enc = encoder.transform(train_Y)

    #Train the model
    model = LogisticRegression().fit(train_X, train_Y_enc)

    #Save the model to the specified location
    joblib.dump(clf, os.path.join(args.model_dir, "scikit_learn_model.joblib"))

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    return np.array(request_body.split(','))


def predict_fn(input_data, model):
    scaled_input = scaler.fit(input_data)
    prediction = model.predict(scaled_input)
    return np.array([prediction])    


def output_fn(prediction, content_type):
    return  encoder.inverse_transform(prediction[0])