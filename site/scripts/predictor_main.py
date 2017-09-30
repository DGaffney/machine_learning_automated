import os
import json
import messenger
from sklearn.externals import joblib
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

i = 1
current_best_model = [None, -10000000.0]
for model in models:
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

    if current_best_model[-1] < np.mean(scores):
        current_best_model = [best_model, np.mean(scores)]
    i += 1

final_model = current_best_model[0]
final_model.fit(x, y)
joblib.dump(final_model, storage_location+'ml_models/'+dataset_id+".pkl")
print json.dumps({"label_type": label_type, "model_params": current_best_model[0].get_params(), "model_name": current_best_model[0].__class__.__name__, "dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "model_path": storage_location+'ml_models/'+dataset_id+".pkl", "status": "complete", "conversion_pipeline": conversion_pipeline, "presumed_label_type": label_type, "best_model": [str(current_best_model[0]), current_best_model[1]], "diagnostic_results": diagnostics.generate_diagnostics(x, y, current_best_model, label_type, dataset_id, diagnostic_image_path)})