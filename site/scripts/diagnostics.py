import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import messenger
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier.classification_report import ClassificationReport
from yellowbrick.classifier.confusion_matrix import ConfusionMatrix
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import ManualAlphaSelection
from yellowbrick.regressor.residuals import ResidualsPlot
from yellowbrick.classifier.class_balance import ClassBalance
from yellowbrick.regressor.residuals import PredictionError
from yellowbrick.features.rankd import Rank1D, Rank2D
from sklearn.model_selection import train_test_split
from yellowbrick.features.pca import PCADecomposition
import json
from scipy.special import comb
import random
import review_model

def generate_diagnostics(x, y, current_best_model, label_type, dataset_id, diagnostic_image_path, percent_complete):
    #messenger.send_update(dataset_id, {"status": "validating_model", "percent": percent_complete, "best_model": [str(current_best_model[0]), current_best_model[1]]})
    if label_type == "Binary":
        return generate_binary_diagnostics(x, y, current_best_model, label_type, diagnostic_image_path)
    elif label_type == "Categorical":
        return generate_categorical_diagnostics(x, y, current_best_model, label_type, diagnostic_image_path)
    elif label_type == "Ordinal":
        print label_type
        print "HELLO"
        return generate_ordinal_diagnostics(x, y, current_best_model, label_type, diagnostic_image_path)

def generate_ordinal_diagnostics(x, y, current_best_model, label_type, diagnostic_image_path):
    x = np.array(x)
    y = np.array(y)
    kf = KFold(n_splits=10, shuffle=True)
    guesses = []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        model = current_best_model[0].fit(X_train, y_train)
        for guess in zip(y_test.tolist(), model.predict(X_test).tolist()):
            guesses.append(guess)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    if "VotingClassifier" not in str(current_best_model[0].__class__):
        visualizer = ResidualsPlot(current_best_model[0])
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.poof(outpath=diagnostic_image_path+"/residuals_plot.png")
        plt.clf()
        visualizer = PredictionError(current_best_model[0])
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.poof(outpath=diagnostic_image_path+"/prediction_error.png")
        plt.clf()
    visualizer = PCADecomposition(scale=True, center=False, col=y, proj_dim=2)
    visualizer.fit_transform(x,y)
    visualizer.poof(outpath=diagnostic_image_path+"/pca_2.png")
    plt.clf()
    visualizer = PCADecomposition(scale=True, center=False, col=y, proj_dim=3)
    visualizer.fit_transform(x,y)
    visualizer.poof(outpath=diagnostic_image_path+"/pca_3.png")
    plt.clf()
    return {"mse": mean_squared_error(*np.array(guesses).transpose()), "r2": r2_score(*np.array(guesses).transpose()), "mae": median_absolute_error(*np.array(guesses).transpose()), "evs": explained_variance_score(*np.array(guesses).transpose()), "rmse": np.sqrt(mean_squared_error(*np.array(guesses).transpose()))}

def generate_binary_diagnostics(x, y, current_best_model, label_type, diagnostic_image_path):
    x = np.array(x)
    y = np.array(y)
    kf = KFold(n_splits=10, shuffle=True)
    guesses = []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = current_best_model[0].fit(X_train, y_train)
        for guess in zip(y_test.tolist(), model.predict(X_test).tolist()):
            guesses.append(guess)
    conmat = {}
    if len(set(y)) == 2:
        tn, fp, fn, tp = confusion_matrix(*np.array(guesses).transpose()).ravel()
        conmat = {"tn": int(tn), "tp": int(tp), "fn": int(fn), "fp": int(fp)}
    else:
        for val in list(set(y)):
            fp = len([el for el in guesses if el[0] == val and el[1] != val])
            tp = len([el for el in guesses if el[0] == val and el[1] == val])
            tn = len([el for el in guesses if el[0] != val and el[1] != val])
            fn = len([el for el in guesses if el[0] != val and el[1] == val])
            conmat[str(val)] = {"tn": tn, "tp": tp, "fn": fn, "fp": fp}
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=int(np.random.random()*100))
    current_count = 0
    while current_count < 1000 and (sorted(set(y_train)) != sorted(set(y_test)) or sorted(set(y_test)) != sorted(set(y))):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=int(np.random.random()*100))
        current_count += 1
    #pickle.dump( [current_best_model, list(set(y)), X_train, X_test, y_train, y_test], open( str(int(np.random.random()*1000000))+"_binary.pkl", "wb" ) )
    #visualizer = ROCAUC(current_best_model[0], classes=list(set(y)))
    #visualizer.fit(X_train, y_train) 
    #visualizer.score(X_test, y_test)
    #visualizer.poof(outpath=diagnostic_image_path+"/roc_auc.png")
    #plt.clf()
    visualizer = ClassificationReport(current_best_model[0], classes=list(set(y)))
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.poof(outpath=diagnostic_image_path+"/classification_report.png")
    plt.clf()
    cm = ConfusionMatrix(current_best_model[0], classes=list(set(y)))
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.poof(outpath=diagnostic_image_path+"/confusion_matrix.png")
    plt.clf()
    visualizer = ClassBalance(current_best_model[0], classes=list(set(y)))
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.poof(outpath=diagnostic_image_path+"/class_balance.png")
    plt.clf()
    visualizer = PCADecomposition(scale=True, center=False, col=y, proj_dim=2)
    visualizer.fit_transform(x,y)
    visualizer.poof(outpath=diagnostic_image_path+"/pca_2.png")
    plt.clf()
    visualizer = PCADecomposition(scale=True, center=False, col=y, proj_dim=3)
    visualizer.fit_transform(x,y)
    visualizer.poof(outpath=diagnostic_image_path+"/pca_3.png")
    plt.clf()
    #"auc": roc_auc_score(*np.array(guesses).transpose()) NEEDS TO BE FIXED.
    return {"accuracy": current_best_model[1], "confusion_matrix": conmat}

def generate_categorical_diagnostics(x, y, current_best_model, label_type, diagnostic_image_path):
    return generate_binary_diagnostics(x, y, current_best_model, label_type, diagnostic_image_path)


def store_model(current_best_model, x, y, dataset_id, label_type, dataset_filename, storage_location, manifest_filename, conversion_pipeline, diagnostic_image_path, percent, score_type, review=False):
    if current_best_model[0] == None:
        return None
    final_model = current_best_model[0]
    final_model.fit(x, y)
    if review == True:
        model_review = review_model.run_all(x, y, current_best_model[0], conversion_pipeline["internal_headers"], score_type)
    else:
        model_review = {}
    joblib.dump(final_model, storage_location+'ml_models/'+dataset_id+".pkl")
    model_params = {}
    if current_best_model[0].__class__.__name__ == "VotingClassifier":
        model_params = current_best_model[0].get_params()
        model_params["estimators"] = {}
        for mmn,mm in current_best_model[0].get_params()["estimators"]:
            model_params["estimators"][mmn] = mm.get_params()
        for key in model_params.keys():
            # try:
            #     gz = json.dumps(model_params[key])
            # except:
            model_params[key] = str(model_params[key])
    elif current_best_model[0].__class__.__name__ == "Pipeline":
        model_params = current_best_model[0].get_params()
        model_params['steps'] = [(el[0], str(el[1])) for el in params['steps']]
        for key in [el[0] for el in params['steps']]:
            model_params[key] = str(params[key])
    else:
        model_params = current_best_model[0].get_params()
    print(json.dumps({"model_review": model_review, "model_found": "true", "label_type": label_type, "model_params": json.dumps(model_params), "model_name": current_best_model[0].__class__.__name__, "dataset_filename": dataset_filename, "storage_location": storage_location, "manifest_filename": manifest_filename, "dataset_id": dataset_id, "model_path": storage_location+'ml_models/'+dataset_id+".pkl", "status": "complete", "conversion_pipeline": conversion_pipeline, "presumed_label_type": label_type, "best_model": [str(current_best_model[0]), current_best_model[1]], "diagnostic_results": generate_diagnostics(x, y, current_best_model, label_type, dataset_id, diagnostic_image_path, percent)}))

def total_combos_possible(best_performing_models):
    total_counts = []
    for i in range(len(best_performing_models))[1:]:
        total_counts.append(comb(len(best_performing_models), i+1, exact=False))
    return total_counts

def get_run_counts_by_size(best_performing_models, run_count):
    probs = [1.0/((i**1.8+1)) for i,el in enumerate(best_performing_models)][1:]
    prob_sum = sum(probs)
    allocated = np.random.multinomial(run_count, [p/prob_sum for p in probs], size=1).tolist()[0]
    total_possible = total_combos_possible(best_performing_models)
    total_set = 0
    to_run = []
    new_probs = []
    for i,val in enumerate(total_possible):
        if allocated[i] > val:
            total_set += val
            to_run.append(val)
            new_probs.append(0.0)
        else:
            total_set += allocated[i]
            to_run.append(allocated[i])
            new_probs.append(probs[i])
    new_prob_sum = sum(new_probs)
    if new_prob_sum > 0:
        new_probs = [el/new_prob_sum for el in new_probs]
        for i,val in enumerate(np.random.multinomial(run_count-total_set, new_probs, size=1).tolist()[0]):
            to_run[i] += val
    return to_run, sum(total_possible)

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


#visualizer = Rank2D(features=x, algorithm='pearson')
#visualizer.fit(x, y)
#visualizer.transform(x)
#visualizer.poof(outpath=diagnostic_image_path+"/pearson_rank2d.png")
#plt.clf()
#visualizer = Rank1D(features=x, algorithm='shapiro')
#visualizer.fit(x, y)
#visualizer.transform(x)
#visualizer.poof(outpath=diagnostic_image_path+"/shapiro_rank1d.png")
#plt.clf()
#visualizer = Rank2D(features=x, algorithm='covariance')
#visualizer.fit(x, y)
#visualizer.transform(x)
#visualizer.poof(outpath=diagnostic_image_path+"/covar_rank2d.png")
#plt.clf()
