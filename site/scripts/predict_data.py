import sys
import pickle
import json
from sklearn.externals import joblib
storage_location = parse_dataset.read_json("settings.json")["storage_location"]
dataset_id = sys.argv[1]
json_path = sys.argv[2]
obs = json.dumps(open(json_path).read())
model = joblib.load(storage_location+"ml_models/"+dataset_id+".pkl") 
print json.dumps(model.predict(obs).tolist())
