"""
File: aws_basic_scikitlearn_model
Date: 2/13/2020
Author: Quinn Lanners
Description: Basic training script used to train a Scikit-learn model on the IRIS training set and practice deploying to AWS Sagemaker.
"""

import argparse
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Dictionary to convert labels to indices
LABEL_TO_INDEX = {
    'Iris-virginica': 0,
    'Iris-versicolor': 1,
    'Iris-setosa': 2
}

# Dictionary to convert indices to labels
INDEX_TO_LABEL = {
    0: 'Iris-virginica',
    1: 'Iris-versicolor',
    2: 'Iris-setosa'
}

"""
__main__

In order for AWS to train the model when you call the API, you 
put the data pre-processing and training in the __main__ block.

Note: You can create seperate jobs to pre-process and train the data. This model's
preprocessing is simple enough to contain in the training script for simplicity
sake.
"""
if __name__ =='__main__':
    # Create a parser object to collect the environment variables that are in the 
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    data = pd.read_csv(os.path.join(args.train,'train.csv'), index_col=0, engine="python")

    # Seperate input variables and labels.
    train_X = data[[c for c in data.columns if c != 'label']]
    train_Y = data[['label']]

    # Convert labels from text to indices
    train_Y_enc = train_Y['label'].map(LABEL_TO_INDEX)

    # Create training pipeline. In this simple example, we just scale the input features and
    # then perform logistic regression.
    pipe = Pipeline([('scale', StandardScaler()), ('log_regression', LogisticRegression())])

    #Train the model using the fit method
    model = pipe.fit(train_X, train_Y_enc)

    #Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


"""
model_fn
    model_dir: (sting) specifies location of saved model

This function is used by AWS Sagemaker to load the model for deployment. 
It does this by simply loading the model that was saved at the end of the 
__main__ training block above and returning it to be used by the predict_fn
function below.
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to 
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are 
only going to accept text/csv and raise an error for all other formats.
"""
def input_fn(request_body, request_content_type):
    if content_type == 'text/csv':
        samples = []
        for r in request_body.split('|'):
            samples.append(list(map(float,r.split(','))))
        return np.array(samples)
    else:
        raise ValueError("Thie model only supports text/csv input")

"""
predict_fn
    input_data: (numpy array) returned array from input_fn above 
    model (sklearn model) returned model loaded from model_fn above

This function is used by AWS Sagemaker to make the prediction on the data
formatted by the input_fn above using the trained model.
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned

This function reformats the predictions returned from predict_fn to the final
format that will be returned as the API call response.

Note: While we don't use content_type in this example, oftentimes you will use
that argument to handle different expected return types.
"""
def output_fn(prediction, content_type):
    return '|'.join([INDEX_TO_LABEL[t] for t in prediction])