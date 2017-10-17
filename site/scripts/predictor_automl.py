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
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

storage_location = parse_dataset.read_json("settings.json")["storage_location"]
dataset_filename = storage_location+sys.argv[1] #"../tmp/59cd43757068cd4193000001_1506627154_mnist_small.csv"
manifest_filename = storage_location+sys.argv[2] #"../tmp/59cd43757068cd4193000001_1506627154_mnist_small_manifest.json"
stated_input_column = sys.argv[3]
prev_acc = float(sys.argv[4])
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

tpot = None
if label_type == "Ordinal"
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, max_eval_time_mins=40, scoring='r2')
else:
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, max_eval_time_mins=40)

@timeout_decorator.timeout(2400)
def train_tpot_model(x,y, tpot):
    X_train, X_test, y_train, y_test = train_test_split(x[0:10000], y[0:10000], train_size=0.8, test_size=0.2)
    tpot.fit(np.array(X_train), np.array(y_train))
    return tpot

train_tpot_model(x,y, tpot)
current_best_model = [None, prev_acc]
if tpot.fitted_pipeline_ != None:
    model = tpot.fitted_pipeline_
    scores = cross_val_score(model, x, y, cv=10, scoring=score_type)
    if current_best_model[1] < np.mean(scores):
        current_best_model = [model, np.mean(scores)]


diagnostics.store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path, percent)
