dataset_filename = "../tmp/59f372ecdb8009537e000017_1509126903_2010-02-22.csv__-_Count_Pct.csv"
manifest_filename = "../tmp/59f372ecdb8009537e000017_1509126903_2010-02-22__-_Count_Pct_manifest.json"
stated_input_column = "Float"
prev_acc = "0.96"
run_speed = "0"
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
prev_acc = float(sys.argv[5])
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

ensemble_model_count = 10
conversion_pipeline = parsed_dataset[2]
messenger.send_update(dataset_id, {"status": "dataset_read", "dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id})
models = model_info.fast_models()
label_type = parse_dataset.label_type(y, stated_input_column)
score_type = "accuracy"
if label_type == "Ordinal":
    models = model_info.fast_ordinal_models()
    score_type = "r2"

model_run_count = float(len(models)+ensemble_model_count)
i = 1
current_best_model = [None, prev_acc]
best_performing_models = []
review = False
#@timeout_decorator.timeout(120)#@timeout(120)
def try_model(model, current_best_model, i):
    percent = (i/model_run_count)*run_multiplier
    messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "label_type": label_type, "status": "running_models", "percent": percent, "model_running": str(model), "best_model": [str(current_best_model[0]), current_best_model[1]]})
    scores = []
    try:
        scores = cross_val_score(model, x, y, cv=10, scoring=score_type)
    except ValueError:
        messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "status": "model_error", "model_error": str(model), "percent": percent})
    except TypeError:
        messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "status": "model_error", "model_error": str(model), "percent": percent})
    if np.abs(current_best_model[-1] - np.mean(scores)) < 0.05 or current_best_model[0] == None:
        best_performing_models.append(model)
    if current_best_model[-1] < np.mean(scores):
        current_best_model = [model, np.mean(scores)]
        diagnostics.store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path, percent, score_type, review)
    i += 1
    return current_best_model

#@timeout_decorator.timeout(120)#@timeout(120)
def try_ensemble_model(models, current_best_model, i):
    percent = (i/model_run_count)*run_multiplier
    try:
        try:
            model = VotingClassifier([(str(el), el) for el in models], voting="soft")
            scores = cross_val_score(model, x, y, cv=10, scoring=score_type)
        except AttributeError:
            model = VotingClassifier([(str(el), el) for el in models])
            scores = cross_val_score(model, x, y, cv=10, scoring=score_type)
    except ValueError:
        return current_best_model
    if current_best_model[-1] < np.mean(scores):
        current_best_model = [model, np.mean(scores)]
        diagnostics.store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path, percent, score_type, review)
    i += 1
    return current_best_model

@timeout_decorator.timeout(2400)#@timeout(10)
def run(models, current_best_model, best_performing_models, i, x, y, label_type, score_type, dataset_filename, manifest_filename, storage_location, conversion_pipeline, diagnostic_image_path):
    for model in models:
        current_best_model = try_model(model, current_best_model, i)
    if current_best_model == [None, -10000000.0]:
        best_performing_models = []
        label_type = "Categorical"
        score_type = "accuracy"
        models = model_info.fast_models()
        i = 1
        current_best_model = [None, -10000000.0]
        for model in models:
            current_best_model = try_model(model, current_best_model, i)
    if len(best_performing_models) > 1:
        for model_count, run_count in enumerate(diagnostics.get_run_counts_by_size(best_performing_models, ensemble_model_count)[0]):
            model_count += 2
            for ik in range(int(run_count)):
                models = list(diagnostics.random_combination(best_performing_models, model_count))
                current_best_model = try_ensemble_model(models, current_best_model, i)
    diagnostics.store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path, run_multiplier*1.0, score_type, True)

error = None
try:
    run(models, current_best_model, best_performing_models, i, x, y, label_type, score_type, dataset_filename, manifest_filename, storage_location, conversion_pipeline, diagnostic_image_path)
except:
    error = sys.exc_info()
    message = None
    if "message" in dir(error[1]):
        message = error[1].message
    else:
        message = error[1].__str__()
    print(json.dumps({"error": True, "error_type": str(error[0]), "message": message, "traceback": traceback.format_tb(error[2])}))