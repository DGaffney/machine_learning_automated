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
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from timeout import timeout
storage_location = parse_dataset.read_json("settings.json")["storage_location"]
dataset_filename = storage_location+sys.argv[1] #"../tmp/59cd43757068cd4193000001_1506627154_mnist_small.csv"
manifest_filename = storage_location+sys.argv[2] #"../tmp/59cd43757068cd4193000001_1506627154_mnist_small_manifest.json"
stated_input_column = sys.argv[3]
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
conversion_pipeline = parsed_dataset[2]
messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "status": "dataset_read"})
models = model_info.model_list()
label_type = parse_dataset.label_type(y, stated_input_column)
score_type = "accuracy"
if label_type == "Ordinal":
    models = model_info.model_ordinal_list()
    score_type = "r2"

@timeout(120)
def try_model(model):
    messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "label_type": label_type, "status": "running_models", "percent": ((i/float(len(models)))*0.75), "model_running": str(model), "best_model": [str(current_best_model[0]), current_best_model[1]]})
    clf = GridSearchCV(model_info.models()[model](), model_info.hyperparameters()[model], cv=5)
    try:
        results= clf.fit(x, y)
    except:
        messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "status": "model_error", "model_error": "grid search error in "+str(model), "percent": (i/float(len(models)))*0.75})
    try:
        best_model = results.best_estimator_
        scores = cross_val_score(best_model, x, y, cv=10, scoring=score_type)
    except:
        messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "status": "model_error", "model_error": str(model), "percent": (i/float(len(models)))*0.75})
    if len(scores) != 0:
        if np.abs(current_best_model[-1] - np.mean(scores)) < 0.05 or current_best_model[0] == None:
            best_performing_models.append(best_model)
        if current_best_model[-1] < np.mean(scores):
            current_best_model = [best_model, np.mean(scores)]
            diagnostics.store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path)

@timeout(120)
def try_ensemble_model(models):
    try:
        model = VotingClassifier([(str(el), el) for el in models], voting="soft")
        scores = cross_val_score(model, x, y, cv=10, scoring=score_type)
    except:
        model = VotingClassifier([(str(el), el) for el in models])
        scores = cross_val_score(model, x, y, cv=10, scoring=score_type)
    if current_best_model[-1] < np.mean(scores):
        current_best_model = [model, np.mean(scores)]
        diagnostics.store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path)
    

i = 1
current_best_model = [None, -10000000.0]
scores = []
best_performing_models = []
for model in models:
    i += 1

if current_best_model == [None, -10000000.0]:
    best_performing_models = []
    label_type = "Categorical"
    score_type = "accuracy"
    models = model_info.model_list()
    i = 1
    current_best_model = [None, -10000000.0]
    for model in models:
        try_model(model)
        i += 1

if len(best_performing_models) > 1:
    for model_count, run_count in enumerate(diagnostics.get_run_counts_by_size(best_performing_models, 50)[0]):
        model_count += 2
        for i in range(int(run_count)):
            models = list(diagnostics.random_combination(best_performing_models, model_count))
            try_ensemble_model(models)

diagnostics.store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path)