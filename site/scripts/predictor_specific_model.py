import sys
sys.argv = ["", "tmp/59f92673db80095895000065_1509541884_2010-02-22.csv__-_Log_Count.csv",
"tmp/59f92673db80095895000065_1509541884_2010-02-22__-_Log_Count_manifest.json" ,
"Float" ,
"tmp/59f8fe4cdb80093a83000051_1509542560_.json"]
import os
import json
import messenger
import parse_dataset
import model_info
import diagnostics
import sys
import json
from sklearn.model_selection import cross_val_score
import numpy as np

storage_location = parse_dataset.read_json("settings.json")["storage_location"]
dataset_filename = storage_location+sys.argv[1]
manifest_filename = storage_location+sys.argv[2]
stated_input_column = sys.argv[3]
model_filename = sys.argv[4]
model_data = parse_dataset.read_json(storage_location+"/"+model_filename)
dataset_id = dataset_filename.split("/")[-1].split("_")[0]
diagnostic_image_path = storage_location+"/public/images/"+dataset_id+"/"
try:
    os.mkdir(diagnostic_image_path)
except: 
    pass

def rescale(data):
    maxval = max(data)
    return [el/maxval for el in data]

messenger.send_update(dataset_id, {"status": "loading_dataset"})
parsed_dataset, manifest = parse_dataset.parse(dataset_filename, manifest_filename)
x = parsed_dataset[0]
y = parsed_dataset[1]
if len(set(y)) == 2 and sorted(set(y)) != [0,1]:
    y = rescale(y)

label_type = parse_dataset.label_type(y, stated_input_column)
score_type = "accuracy"
if label_type == "Ordinal":
    score_type = "r2"

conversion_pipeline = parsed_dataset[2]
messenger.send_update(dataset_id, {"status": "dataset_read", "dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id})
model = model_info.models()[model_data['internal_name']]()
model.set_params(**model_data['params'])
messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "label_type": label_type, "status": "running_models", "percent": 0.5, "model_running": str(model), "best_model": [str(model), 0]})
scores = []
try:
    scores = cross_val_score(model, x, y, cv=10, scoring=score_type)
except ValueError:
    messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "status": "model_error", "model_error": str(model), "percent": 0.6})
except TypeError:
    messenger.send_update(dataset_id, {"dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "status": "model_error", "model_error": str(model), "percent": 0.6})

diagnostics.store_model([model, np.mean(scores)], x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path, 0.85, score_type, True)