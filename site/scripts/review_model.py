import sys
import pickle
import json
import parse_dataset
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import r2_score
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
import numpy as np
#storage_location = parse_dataset.read_json("../settings.json")["storage_location"]
#dataset_id = "59f1200cdb800917260003b6"
#model = joblib.load(storage_location+"ml_models/"+dataset_id+".pkl") 
#dataset_filename = "../tmp/problem_705393.csv"
#manifest_filename = "../tmp/problem_705393.json"
#stated_input_column = "Float"
#parsed_dataset, manifest = parse_dataset.parse(dataset_filename, manifest_filename)
#x = parsed_dataset[0]
#y = parsed_dataset[1]
#conversion_pipeline = parsed_dataset[2]
def run_func(method, x, y, model, names, score_type):
    try:
        return method(x, y, model, names, score_type)
    except ValueError:
        return [[0, el] for el in names]
    
def run_all(x, y, model, names, score_type):
    sets = []
    sets.append(["univariate_test", run_func(univariate_test, x, y, model, names, score_type)])
    #sets.append(["mean_decrease_accuracy", run_func(mean_decrease_accuracy, x, y, model, names, score_type)])
    sets.append(["stability_test", run_func(stability_test, x, y, model, names, score_type)])
    #sets.append(["recursive_feature_elimination_test", run_func(recursive_feature_elimination_test, x, y, model, names, score_type)])
    #sets.append(["recursive_feature_elimination_cross_validation_test", run_func(recursive_feature_elimination_cross_validation_test, x, y, model, names, score_type)])
    sets.append(["regularization_test", run_func(regularization_test, x, y, model, names, score_type)])
    sets.append(["lin_regress_test", run_func(lin_regress_test, x, y, model, names, score_type)])
    sets.append(["ridge_test", run_func(ridge_test, x, y, model, names, score_type)])
    sets.append(["random_forest_test", run_func(random_forest_test, x, y, model, names, score_type)])
    sets.append(["mean_decrease_accuracy", run_func(mean_decrease_accuracy, x, y, model, names, score_type)])
    metric_scores = {}
    for k in names:
        metric_scores[k] = 0
        for metric, score_set in sets:
            score_names = [el[1] for el in score_set]
            metric_scores[k] += score_set[score_names.index(k)][0]/len(sets)
    return {'details': sets, 'metric_scores': metric_scores}

def univariate_test(x, y, model, names, score_type):
    scores = []
    X = np.matrix(x)
    for i in range(X.shape[1]):
        score = cross_val_score(model, X[:, i:i+1], y, scoring=score_type, cv=ShuffleSplit(len(X), 3, .3))
        scores.append(round(np.mean(score), 3))
    maxval = max(scores)
    minval = min(scores)
    dist = maxval-minval
    return list(zip((np.array(scores)-minval)/dist, names))

def stability_test(x, y, model, names, score_type):
    if score_type != "r2":
        rlasso = RandomizedLogisticRegression()
        rlasso.fit(x, y)
    else:
        rlasso = RandomizedLasso(alpha=0.025)
        rlasso.fit(x, y)
    if sum(rlasso.scores_) == 0:
        return [[0, el] for el in names]
    maxval = max(rlasso.scores_)
    minval = min(rlasso.scores_)
    dist = maxval-minval
    return list(zip(map(lambda x: round(x, 4), (rlasso.scores_-minval)/dist), names))

def recursive_feature_elimination_test(x, y, model, names, score_type):
    rfe = RFE(model, n_features_to_select=1)
    rfe.fit(x,y)
    maxval = max(rfe.ranking_)
    minval = min(rfe.ranking_)
    dist = maxval-minval
    rfe_scores = list(zip(map(lambda x: round(x, 4), 1-(rfe.ranking_-minval)/dist), names))
    return rfe_scores

def recursive_feature_elimination_cross_validation_test(x, y, model, names, score_type):
    rfecv = RFECV(model, step=1, cv=2)
    rfecv.fit(x,y)
    maxval = max(rfe.ranking_)
    minval = min(rfe.ranking_)
    dist = maxval-minval
    rfe_scores = list(zip(map(lambda x: round(x, 4), (rfe.ranking_-minval)/dist), names))
    return rfe_scores

def regularization_test(x, y, model, names, score_type):
    if score_type != "r2":
        regularized = LogisticRegression()
        regularized.fit(StandardScaler().fit_transform(x), y)
        scores = [sum(np.abs(el)) for el in regularized.coef_]
    else:
        regularized = Lasso(alpha=0.025)
        regularized.fit(StandardScaler().fit_transform(x), y)
        scores = regularized.coef_
    maxval = max(scores)
    minval = min(scores)
    dist = maxval-minval
    return list(zip((np.array(scores)-minval)/dist, names))

def lin_regress_test(x, y, model, names, score_type):
    lr = None
    lr_coefs = []
    if score_type != "r2":
        lr = LogisticRegression()
    else:
        lr = LinearRegression()
    for i in range(10):
        np.random.seed(seed=i)
        lr.fit(x,y)
        lr_coefs.append(lr.coef_)
    if score_type != "r2":
        lr_coefs = [np.mean(el) for el in np.transpose([[np.mean(np.abs(el)) for el in np.transpose(coef)] for coef in lr_coefs])]
    else:
        lr_coefs = [np.mean(np.abs(el)) for el in np.transpose(lr_coefs)]
    maxval = max(lr_coefs)
    minval = min(lr_coefs)
    dist = maxval-minval
    return list(zip((np.array(lr_coefs)-minval)/dist, names))

def ridge_test(x, y, model, names, score_type):
    ridge_coefs = []
    for i in range(10):
        np.random.seed(seed=i)
        ridge = None
        if score_type == "r2":
            ridge = Ridge(alpha=10)
            ridge.fit(x,y)
            ridge_coefs.append(ridge.coef_)
        else:
            ridge = RidgeClassifier(alpha=10)
            ridge.fit(x,y)
            ridge_coefs.append([sum(np.abs(el)) for el in np.transpose(ridge.coef_)])
    scores = [sum(el) for el in np.transpose(ridge_coefs)]  
    maxval = max(scores)
    minval = min(scores)
    dist = maxval-minval
    return list(zip((np.array(scores)-minval)/dist, names))

def random_forest_test(x, y, model, names, score_type):
    rf = None
    if score_type == "r2":
        rf = RandomForestRegressor()
    else:
        rf = RandomForestClassifier()
    rf.fit(x,y)
    scores = list(map(lambda x: round(x, 4), rf.feature_importances_))
    maxval = max(scores)
    minval = min(scores)
    dist = maxval-minval
    return list(zip((np.array(scores)-minval)/dist, names))

    
def mean_decrease_accuracy(x, y, model, names, score_type):
    scores = defaultdict(list)
    X = np.matrix(x)
    Y = np.array(y)
    for train_idx, test_idx in ShuffleSplit(len(x), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = model.fit(X_train, Y_train)
        acc = r2_score(Y_test, model.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, model.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    scored = [round(np.mean(score), 4) for feat, score in scores.items()]
    maxval = max(scored)
    minval = min(scored)
    dist = maxval-minval
    return list(zip((np.array(scored)-minval)/dist, [el[0] for el in scores.items()]))
