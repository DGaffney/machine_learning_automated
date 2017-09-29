from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC

def fast_models():
    return [AdaBoostClassifier(learning_rate= 0.5, n_estimators=10),
    AdaBoostClassifier(n_estimators=10),
    GaussianNB(),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 10, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 10, 4, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 15, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 15, 6, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 20, 8, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 25, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 25, 20, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 30, 12, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 5, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 20, 5, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 2, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 2, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 8, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25, 10, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30, 2, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2, 2), random_state=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2, 2), random_state=1),
    NearestCentroid(),
    linear_model.Perceptron(fit_intercept=False, n_iter=10, shuffle=False),
    linear_model.Perceptron(fit_intercept=False, n_iter=3, shuffle=False),
    linear_model.Perceptron(fit_intercept=False, n_iter=5, shuffle=False),
    linear_model.Perceptron(fit_intercept=True, n_iter=10, shuffle=False),
    linear_model.Perceptron(fit_intercept=True, n_iter=3, shuffle=False),
    linear_model.Perceptron(fit_intercept=True, n_iter=5, shuffle=False),
    RandomForestClassifier(criterion='entropy', n_estimators=10),
    RandomForestClassifier(criterion='entropy', n_estimators=18),
    RandomForestClassifier(criterion='entropy', n_estimators=2),
    RandomForestClassifier(n_estimators=10),
    RandomForestClassifier(n_estimators=18),
    RandomForestClassifier(n_estimators=2),
    RandomForestClassifier(random_state=1),
    SGDClassifier(loss='hinge', penalty='l2'),
    SGDClassifier(loss='log'),
    linear_model.ElasticNet(alpha=0.5, l1_ratio=0.7),
    linear_model.Lasso(alpha = 0.5),
    linear_model.LassoLars(alpha=0.1),
    linear_model.LassoLars(alpha=0.5),
    linear_model.Ridge(alpha = 0.5),
    tree.DecisionTreeClassifier(),
    tree.DecisionTreeRegressor()]

def fast_ordinal_models():
    return [linear_model.ElasticNet(alpha=0.5, l1_ratio=0.7),
    linear_model.Lasso(alpha = 0.5),
    linear_model.LassoLars(alpha=0.1),
    linear_model.LassoLars(alpha=0.5),
    linear_model.Ridge(alpha = 0.5),
    linear_model.ElasticNet(alpha=0.05, l1_ratio=0.7),
    linear_model.Lasso(alpha = 0.05),
    linear_model.LassoLars(alpha=0.01),
    linear_model.LassoLars(alpha=0.05),
    linear_model.Ridge(alpha = 0.05),
    linear_model.ElasticNet(alpha=0.005, l1_ratio=0.7),
    linear_model.Lasso(alpha = 0.005),
    linear_model.LassoLars(alpha=0.001),
    linear_model.LassoLars(alpha=0.005),
    linear_model.Ridge(alpha = 0.005),
    tree.DecisionTreeRegressor()]
    
def model_list():
    return ["AdaBoostClassifier",
    "GaussianNB",
    "GaussianProcessRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "KNeighborsClassifier",
    "MLPClassifier",
    "NearestCentroid",
    "NuSVC",
    "linear_model.Perceptron",
    "RandomForestClassifier",
    "SGDClassifier",
    "linear_model.BayesianRidge",
    "linear_model.ElasticNet",
    "linear_model.Lasso",
    "linear_model.LassoLars",
    "linear_model.LinearRegression",
    "linear_model.LogisticRegression",
    "linear_model.Ridge",
    "linear_model.RidgeCV",
    "svm.LinearSVC",
    "svm.SVC",
    "svm.SVR",
    "tree.DecisionTreeClassifier",
    "tree.DecisionTreeRegressor"]

def model_ordinal_list():
    return ["GaussianProcessRegressor",
    "GradientBoostingRegressor",
    # "linear_model.BayesianRidge",
    "linear_model.ElasticNet",
    "linear_model.Lasso",
    "linear_model.LassoLars",
    "linear_model.LinearRegression",
    "linear_model.Ridge",
    "linear_model.RidgeCV",
    "svm.SVR",
    "tree.DecisionTreeRegressor"]
    

def models():
    return {'linear_model.LinearRegression': linear_model.LinearRegression,
    'linear_model.Ridge': linear_model.Ridge,
    'linear_model.RidgeCV': linear_model.RidgeCV,
    'linear_model.Lasso': linear_model.Lasso,
    'linear_model.LassoLars': linear_model.LassoLars,
    'linear_model.BayesianRidge': linear_model.BayesianRidge,
    'linear_model.ElasticNet': linear_model.ElasticNet,
    'linear_model.Perceptron': linear_model.Perceptron,
    'linear_model.LogisticRegression': linear_model.LogisticRegression,
    'svm.LinearSVC': svm.LinearSVC,
    'svm.SVC': svm.SVC,
    'svm.SVR': svm.SVR,
    'SGDClassifier': SGDClassifier,
    'NearestNeighbors': NearestNeighbors,
    'NearestCentroid': NearestCentroid,
    'GaussianProcessRegressor': GaussianProcessRegressor,
    'GaussianNB': GaussianNB,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'tree.DecisionTreeClassifier': tree.DecisionTreeClassifier,
    'tree.DecisionTreeRegressor': tree.DecisionTreeRegressor,
    'KNeighborsClassifier': KNeighborsClassifier,
    'AdaBoostClassifier': AdaBoostClassifier,
    'Perceptron': Perceptron,
    'NuSVC': NuSVC,
    'MLPClassifier': MLPClassifier}

def hyperparameters():
    return {'KNeighborsClassifier': {'n_neighbors': [1, 2, 5, 11], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [10, 20, 30, 40, 50]},
    'linear_model.LinearRegression': {},
    'linear_model.Perceptron': {'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]},
    'linear_model.Ridge': {'fit_intercept': [True, False], 'normalize': [True, False], 'alpha': [1.5**el for el in range(10)], 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']},
    'linear_model.RidgeCV': {'fit_intercept': [True, False], 'normalize': [True, False], 'alphas': [1.5**el for el in range(10)], 'gcv_mode':['auto', 'svd', 'eigen']},
    'linear_model.Lasso': {'fit_intercept': [True, False], 'normalize': [True, False], 'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'alpha': [1.5**el for el in range(10)]},
    'linear_model.LassoLars': {'fit_intercept': [True, False], 'normalize': [True, False], 'eps': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'alpha': [1.5**el for el in range(10)]},
    'linear_model.BayesianRidge': {'normalize': [True, False], 'n_iter': [10, 100, 1000, 2000], 'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'alpha_1': [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'alpha_2': [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'lambda_1': [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'lambda_2': [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]},
    'linear_model.ElasticNet': {'l1_ratio': [0.2, 0.5, 0.7], 'fit_intercept': [True, False], 'normalize': [True, False], 'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'alpha': [1.5**el for el in range(10)]},
    'linear_model.LogisticRegression': {'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'fit_intercept': [True, False], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
    'svm.LinearSVC': {'loss': ['hinge', 'squared_hinge'], 'tol': [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'fit_intercept': [True, False], 'multi_class': ['crammer_singer', 'ovr']},
    'svm.SVC': {'probability': [True], 'tol': [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'decision_function_shape': ['ovo', 'ovr'], "kernel": ['linear','rbf']},
    'svm.SVR': {'tol': [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'epsilon': [0.01, 0.05, 0.1, 0.5]},
    'SGDClassifier': {'loss': ['hinge', 'log'], 'tol': [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'epsilon': [0.01, 0.05, 0.1, 0.5]},
    'NearestNeighbors': {'n_neighbors': [1, 2, 5, 11], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [10, 20, 30, 40, 50]},
    'KNearestNeighbors': {'n_neighbors': [1, 2, 5, 11], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [10, 20, 30, 40, 50]},
    'NearestCentroid': {},
    'GaussianProcessRegressor': {'alpha': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8], 'normalize_y': [True, False]},
    'GaussianNB': {},
    'RandomForestClassifier': {'criterion': ['gini', 'entropy'], 'n_estimators': [1, 5, 10, 15, 20], 'max_features': ['auto', 'sqrt', 'log2', None]},
    'GradientBoostingClassifier': {'loss': ['deviance', 'exponential'], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.7], 'n_estimators': [5, 20, 50, 100, 120, 150]},
    'GradientBoostingRegressor': {'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.7], 'n_estimators': [5, 20, 50, 100, 120, 150]},
    'tree.DecisionTreeClassifier': {'max_features': ['auto', 'sqrt', 'log2', None]},
    'tree.DecisionTreeRegressor': {'max_features': ['auto', 'sqrt', 'log2', None]},
    'AdaBoostClassifier': {'n_estimators': [10, 20, 50, 60, 70, 100]},
    'NuSVC': {'nu': [0.1, 0.3, 0.5, 0.7, 0.9], 'kernel': ['sigmoid', 'linear', 'poly', 'rbf']},
    'Perceptron': {'alpha': [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'fit_intercept': [True, False], 'shuffle': [True, False]},
    'MLPClassifier': {'solver': ['lbfgs', 'sgd', 'adam'], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'learning_rate': ['constant', 'invscaling', 'adaptive']}}
