import os
import json
import messenger
import parse_dataset
import model_info
import diagnostics
import sys
import json
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
#from timeout import timeout
import time
import timeout_decorator
import traceback
storage_location = parse_dataset.read_json("settings.json")["storage_location"]
dataset_filename = storage_location+sys.argv[1] #"../tmp/59cd43757068cd4193000001_1506627154_mnist_small.csv"
manifest_filename = storage_location+sys.argv[2] #"../tmp/59cd43757068cd4193000001_1506627154_mnist_small_manifest.json"
stated_input_column = sys.argv[3]
run_speed = sys.argv[4]
run_multiplier = 0.5
if run_speed == "2":
    run_multiplier = 1.0

def rescale(data):
    maxval = max(data)
    return [el/maxval for el in data]

dataset_id = dataset_filename.split("/")[-1].split("_")[0]
diagnostic_image_path = storage_location+"/public/images/"+dataset_id+"/"
try:
    os.mkdir(diagnostic_image_path)
except: 
    pass

messenger.send_update(dataset_id, {"status": "loading_dataset"})
parsed_dataset, manifest = parse_dataset.parse(dataset_filename, manifest_filename)
x = parsed_dataset[0]
y = parsed_dataset[1]
if len(set(y)) == 2 and sorted(set(y)) != [0,1]:
    y = rescale(y)

messenger.send_update(dataset_id, {"status": "dataset_read", "dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id})
label_type = parse_dataset.label_type(y, stated_input_column)
score_type = "accuracy"
if label_type == "Ordinal":
    models = model_info.fast_ordinal_models()
    score_type = "r2"


import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.regression
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x[0:10000], y[0:10000], random_state=1)
if label_type == "Ordinal":
    automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120, per_run_time_limit=30,
    tmp_folder='tmp/autoslearn_regression_example_tmp',
    output_folder='tmp/autosklearn_regression_example_out')
    automl.fit(X_train, y_train, dataset_name=dataset_id,
    feat_type=feature_types)
else:
    automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=10, per_run_time_limit=30,
    tmp_folder='/tmp/autoslearn_cv_example_tmp',
    output_folder='/tmp/autosklearn_cv_example_out',
    delete_tmp_folder_after_terminate=False,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5})

automl.fit(X_train.copy(), y_train.copy(), dataset_name=dataset_id)
automl.refit(X_train.copy(), y_train.copy())

    print(automl.show_models())

    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
