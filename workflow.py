from datetime import datetime
import time
from typing import List
from data_processor import DataProcessor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np


def grid_search(param_grid, params, n_estimators, dftrain, y_train, kfold):
    """Grid Search function"""

    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
        **{**params, **{'n_estimators': n_estimators}}),
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=kfold,
        verbose=2
    )

    gsearch.fit(dftrain, y_train)

    print("\nBest: %f using %s\n" % (gsearch.best_score_, gsearch.best_params_))

    means = gsearch.cv_results_['mean_test_score']
    stds = gsearch.cv_results_['std_test_score']
    params = gsearch.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return gsearch.best_params_


def update_params(param_grid, params, n_estimators, dftrain, y_train, xgtrain, kfold, kfold_list):
    """Update params and number of trees"""
    best_params = grid_search(param_grid, params, n_estimators, dftrain, y_train, kfold)
    for key in param_grid.keys():
        params[key] = best_params[key]

    number_of_trees = xgb.cv(
        params,
        xgtrain,
        num_boost_round=1000,
        early_stopping_rounds=50,
        folds=kfold_list,
        verbose_eval=50
    ).shape[0]

    print("Updated params : ", params)
    print("Updated number of trees : ", number_of_trees)

    # update number of trees param
    return number_of_trees


def features_importance(importance_type, bst):
    """
    Calculate pourcentage features importance
    """
    print("\nImportance type : ", importance_type)
    dict_importance = bst.get_score(importance_type=importance_type)
    sum_importance = sum(dict_importance.values())
    dict_importance.update((x, y / sum_importance) for x, y in dict_importance.items())
    dict_importance_sorted = {k: v for k, v in sorted(dict_importance.items(), key=lambda item: item[1], reverse=True)}
    print("Total importance sum : ", sum_importance)
    print(dict_importance_sorted)



def multi_class_accuracy(preds, gt):
    hits = 0
    for i in range(gt.shape[0]):
        gt_label = np.argmax(gt[i])
        pred_label = np.argmax(preds[i])
        if gt_label==pred_label: hits+=1
    return hits/gt.shape[0]




def do_workflow(dftrain, y_train, dftest, y_test, kfolds):
    """
        Running the workflow
    """

    # xgboost matrix creation
    #xgtrain = xgb.DMatrix(dftrain.values, label=y_train.values, feature_names=dftrain.keys().values)
    xgtrain = xgb.DMatrix(dftrain.values, label=y_train, feature_names=dftrain.keys().values)
    #xgtest = xgb.DMatrix(dftest.values, feature_names=dftrain.keys().values)
    xgtest = xgb.DMatrix(dftest.values, feature_names=dftrain.keys().values)

    # create folds
    kfold = KFold(n_splits=kfolds, shuffle=False)
    kfold_list = []
    for train_index, test_index in kfold.split(dftrain, y_train):
        kfold_list.append((train_index, test_index))

    # we fix the initials params
    params = {
        'max_depth': 5,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'min_child_weight': 1,
        'gamma': 0,
        'nthread': 4,
        'scale_pos_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
        'seed': 27
    }

    print("\nBase model without tuning : ")
    base_model = xgb.train(params=params, dtrain=xgtrain, num_boost_round=100)

    # model predictions on test datas
    preds = base_model.predict(xgtest)
    #print(preds)
    #predictions = [np.argmax(value) for value in preds]

    base_model_accuracy = multi_class_accuracy(preds, y_test)
    print("Val accuracy: %.2f%%" % (base_model_accuracy * 100.0))

    # ------------------------------ Workflow ------------------------------

    # step 1 : Estimate numbers of trees
    print("\nStep 1 : Estimate numbers of trees...")
    
    num_boost_round = xgb.cv(params,
                             xgtrain,
                             num_boost_round=1000,
                             early_stopping_rounds=50,
                             folds=kfold_list,
                             verbose_eval=50
                             ).shape[0]

    print("Initial num_boost_round : ", num_boost_round)
    
    # step 2 : Tune max_depth and min_child_weight
    print("\nStep 2 : Tune max_depth and min_child_weight...")
    num_boost_round = update_params({
        'max_depth': [i for i in range(3, 6)],
        'min_child_weight': [i for i in range(1, 6)],
    }, params, num_boost_round, dftrain, y_train, xgtrain, kfold, kfold_list)
    #bst.save_model('0001.model')
    # step 3: Tune gamma
    
    print("\nStep 3 : Tune gamma...")
    num_boost_round = update_params({
        'gamma': [i / 10.0 for i in range(0, 3)]
    }, params, num_boost_round, dftrain, y_train, xgtrain, kfold, kfold_list)
    #bst.save_model('0001.model')
    # step 4: Tune regularization parameters
    print("\nStep 4 : Tune regularization parameters...")
    num_boost_round = update_params({
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 1e-2, 0.1, 1]
    }, params, num_boost_round, dftrain, y_train, xgtrain, kfold, kfold_list)
    #bst.save_model('0001.model')
    # step 5: Tune subsample and colsample_bytree
    print("\nStep 5 : Tune subsample and colsample_bytree...")
    update_params({
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1],
    }, params, num_boost_round, dftrain, y_train, xgtrain, kfold, kfold_list)
    #bst.save_model('0001.model')
    # step 6: Reducing learning rate and tune number of trees parameter
    print("\nStep 6 : Reducing learning rate and tune number of trees parameter...")
    params['eta'] = 0.01
    #bst.save_model('0001.model')
    cv = xgb.cv(params, xgtrain, num_boost_round=10000, early_stopping_rounds=500, folds=kfold_list, verbose_eval=500)
    num_boost_round = cv.shape[0]

    # training model
    bst = xgb.train(params=params, dtrain=xgtrain, num_boost_round=num_boost_round)
    bst.save_model('0001.model')
    # model predictions on test datas
    preds = bst.predict(xgtest)
    predictions = [round(value) for value in preds]

    accuracy = multi_class_accuracy(preds, y_test)
    print("\nFinal number of trees : ", num_boost_round)
    print("Final params : ", params)

    print("CV Training accuracy : %.2f%%" % ((1 - cv['test-error-mean'].min()) * 100.0))
    print("Val accuracy: %.2f%%" % (accuracy * 100.0))

    # confusion matrix
    print("Confusion matrix : \n", confusion_matrix(y_test, predictions))

    # weight importance
    features_importance("weight", bst)

    # weight importance
    features_importance("gain", bst)

    # weight importance
    features_importance("cover", bst)

    return bst




def run_workflow_with_train_test_file(
        train_file: str,
        test_file: str,
        kfolds: int,
        label_column_name: str,
        features_file: str = None,
        features_group_ids: List[int] = None,
        features_ids: List[int] = None
):
    """
    Runs the workflow for a given excel file.
    It runs tests with various parameters with GridSearchSCV and returns the resulting accuracy for a set of parameters.

    Warning: this function assumes there is a `display` function that exists in the scope of where this function is
    called. It is so because this will be used directly in a jupyter notebook, but on the off-chance it is used in
    a different context, you must create a display function before calling this function.


    :param train_file: The path to the excel train file.
    :param test_file: The path to the excel test file.
    :param kfolds: The number of folds for Cross Validation.
    :param label_column_name: The name of the label column in the excel file.
    :param group_column_name: The name of the group column in the excel file.
    :param features_file: Needed when using features in particular.
    :param features_group_ids: Needed when using some group of features in particular.
    :param features_ids: Needed when using some features in particular.
    :return: The list of results of parameters tests in the workflow; a result is a (parameters, accuracy) tuple.
    """


    start_time = time.time()

    data_proc = DataProcessor()
    print("Loading data...")
    dftrain = data_proc.load(train_file, **{
        "features_file": features_file,
        "features_ids": features_ids,
        "features_group_ids": features_group_ids,
        "features_to_keep": [label_column_name, group_column_name]
    })

    dftest = data_proc.load(test_file, **{
        "features_file": features_file,
        "features_ids": features_ids,
        "features_group_ids": features_group_ids,
        "features_to_keep": [label_column_name, group_column_name]
    })

    print("Data loaded.")

    display(dftrain)

    y_train, y_test = label_extracting(dftrain=dftrain,
                                       dftest=dftest,
                                       label_column_name=label_column_name,
                                       group_column_name=group_column_name)

    # executing the workflow
    bst = do_workflow(
        dftrain=dftrain,
        y_train=y_train,
        dftest=dftest,
        y_test=y_test,
        kfolds=kfolds,
    )

    print("Time to run workflow:", time.time() - start_time, "s")

    return bst
