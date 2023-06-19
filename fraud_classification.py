import os
import sys
import argparse
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK, Trials, tpe, hp, fmin
from sklearn.metrics import classification_report,confusion_matrix

#get current directory
directory = os.path.realpath(os.path.dirname(__file__))

def parse_args():
    # Defines the aruguments required to run the script (specify classifier)
    parser = argparse.ArgumentParser(description='Train a classifier for fraud prediction.')
    parser.add_argument('--classifier', dest='clf_name', choices=['lgbmc', 'lgbmc-tuned', 'logistic'])
    args = parser.parse_args()
    return args

def load_file(file_name: str) -> pd.DataFrame:
    # Loads csv files as pandas dataframe

    #load file
    df = pd.read_csv(directory + '/data/' + file_name + '.csv')
    
    return df

def prepare_data(independent_variables: pd.DataFrame, dependent_variable: pd.Series, test_size: int) -> Union[pd.DataFrame, pd.Series]:
    # Split data into train and test set based on the provided test size
    
    X_train, X_test, y_train, y_test =\
        train_test_split(independent_variables, 
                         dependent_variable, 
                         test_size=test_size, 
                         random_state=42)

    return X_train, X_test, y_train, y_test

def calulate_zip_code(zip_item: str) -> str:
    # Validates zip code
    # No information was provided about the countries or how a valid zip code should be formatted
    # Therefore, zip code is validated for null invalid and valid zip codes
    # We only label zip codes with characters or 0's as Invalid zip codes 
    # Further study: Identify source of the data and identify bettwe ways to process zip codes

    # if zip code not provided for transaction
    if pd.isnull(zip_item):
        return 'not_available'
    # if zip code is invalid (e.g. '**')
    elif zip_item in ['**', '***','..','...','....','.....','0']:
        return 'invalid_zip_code'
    # if zip code valid
    else:
        return 'valid_zip_code'

def countplot_output(df: pd.DataFrame, x_variable: str, title: str, fig_name: str):  
    # Outputs countplot figure
        
    plt.figure(figsize=(9, 5))
    sns.countplot(data = df, x = x_variable)
    plt.title(title)
    plt.xticks(rotation=90)
    
    plt.savefig(directory + '/output/figures/' + fig_name, bbox_inches='tight')

def apply_scaler(df: pd.DataFrame, items: list) -> pd.DataFrame:
    # Applies RobustScaler to selected variables
    
    rob_scaler = RobustScaler()

    #loop through each item and apply robust scaler    
    for item in items:
        df['scaled' + item] = rob_scaler.fit_transform(df[item].values.reshape(-1,1))
        df.drop(item, axis=1, inplace=True)

    return df

def apply_encoders(df, logreg = None):
    # Applies Ordinal encoder and Label encoder based on condition

    #Only applies ordinal encodeR for accountNumber if the model is lightbgm
    if logreg is None:
        #Create ordinal encoder
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df['accountNumber_encoded'] = ordinal_encoder.fit_transform(np.array(df['accountNumber']).reshape(-1, 1)).astype(int)
    #Drop variable after encoding or without for logistic regression
    df.drop('accountNumber', axis=1, inplace=True)

    #Create transformer for one hot encoding for stated variables
    transformer = make_column_transformer((OneHotEncoder(), ['zip_validation', 'day_of_week', 'month']),remainder='passthrough')
    #Fit transformer and create dataframe
    transformed = transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

    #Create new dependent and independet features
    X = transformed_df.drop('remainder__target_label', axis=1)
    y = transformed_df['remainder__target_label'].astype('int')
    
    #Return based on model 
    if logreg is None:
        return X, y, ordinal_encoder, transformer
    else:
        return X, y, transformer

def calculate_class_weights(df: pd.DataFrame) -> dict:
    # Weights of minory class adjusted over come the issue of class imbalance
    # Formula: wj = n_samples / (n_classes * n_samplesj)
        #wj is the weight for each class(j signifies the class)
        #n_samplesis the total number of samples or rows in the dataset
        #n_classesis the total number of unique classes in the target
        #n_samplesjis the total number of rows of the respective class

    #Calculate class distribution
    neg_class_count = df['target_label'].value_counts()[0]
    pos_class_count = df['target_label'].value_counts()[1]
    #Calculate sum of traning data
    total_class_count = neg_class_count + pos_class_count
    
    #Calculate new weights
    weight_for_0 =  total_class_count / (2 * neg_class_count)
    weight_for_1 =  total_class_count / (2 * pos_class_count)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    #print('Weight for class 0: {:.2f}'.format(weight_for_0))
    #print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return class_weight

def display_model_performance(X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.Series, model: str, type: str):
    # Exports classification report as csv for model evaluvation
    # Ouputs confusion matrix of model performance as figure
    
    #Produce and export classification report
    clsf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    clsf_report.to_csv(directory + '/output/figures/' + model + '_' + type + '.csv', index= True)
    
    #Create and output confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    fig, ax = plt.subplots(figsize = (5,5))
    sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(model + ' ' + type);
    ax.xaxis.set_ticklabels(['Non-fraud', 'Fraud'])
    ax.yaxis.set_ticklabels(['Non-fraud', 'Fraud']);
    plt.savefig(directory + '/output/figures/' + model + '_' + type + '.png', bbox_inches='tight')

if __name__ == "__main__":

    args = parse_args()

    #load data files as dataframe
    trans_df = load_file('transactions_obf')
    label_df = load_file('labels_obf')
    
    #join the dataframes on eventId
    df = trans_df.merge(label_df, on='eventId', how='left')

    #calculate target label
    #fraud transactions are entires with reportedTime and labelled as 1 
    # non fraudulent transactions are entries with null values for reportedTime and labelled as 0
    df['target_label'] = np.where(df['reportedTime'].isnull(), 0, 1)
    #reportedTime variable dropped as will not be required for further use
    df.drop('reportedTime', axis=1, inplace=True) 

    #convert transactionTime to date time
    df['transactionTime'] = pd.to_datetime(df['transactionTime'])

    #extract features from transactionTime
    df['day_of_week'] = df['transactionTime'].apply(lambda x: x.day_name())
    df['month'] = df['transactionTime'].apply(lambda x: x.month_name())
    df['day'] = df['transactionTime'].apply(lambda x: x.day)
    df['hour'] = df['transactionTime'].apply(lambda x: x.hour)
    df['year'] = df['transactionTime'].apply(lambda x: x.year)
    
    #calculate proportion of transactionAmount
    df['transaction_proportion'] = df['transactionAmount'] / df['availableCash'] * 100
    
    #validate zip code using calulate_zip_code function and then drop varibale
    df['zip_validation'] = df['merchantZip'].apply(lambda x: calulate_zip_code(x))
    df.drop('merchantZip', axis=1, inplace=True)
    
    #remove variables that will not be used
    #transactionTime removed as we have extracted that information required from the variable for our modeling
    #eventId is removed as it's a unique identifier which holds no significant value
    #merchantId is removed as it's a unique indentifier
    df.drop(['transactionTime', 'eventId', 'merchantId'], axis=1, inplace=True)
    
    #Model will be trained and validated on 2017 data
    #2018 January data will be used for model evaluvation
    df_2018 = df[df['year']==2018].copy()
    df = df[df['year']==2017].copy()
    df.drop('year', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_2018.drop('year', axis=1, inplace=True)
    df_2018.reset_index(drop=True, inplace=True)

    #EDA of transactions
    countplot_output(df, 'month', 'Total Transactions (2017)', 'month_trans.png')
    countplot_output(df[df['target_label']==1],'month', 'Fraud Transactions (2017)', 'month_trans_fraud.png')

    # RobustScaler is used to scale the variables as it is less prone to outliers.
    df = apply_scaler(df,['transactionAmount','availableCash', 'transaction_proportion'])

    # apply encoding
    if args.clf_name == "logistic":
        df.drop('merchantCountry', axis=1, inplace=True)
        df.drop('mcc', axis=1, inplace=True)
        X, y, transformer = apply_encoders(df, "Yes")
    else:
        X, y, ordinal_encoder, transformer = apply_encoders(df)
    
    #split data for modelling
    X_train, X_test, y_train, y_test = prepare_data(X, y, 0.2)
    
    #calculate class weights
    class_weight = calculate_class_weights(df)

    if args.clf_name == 'lgbmc':
        
        #create and train calssifier 
        clf = lgb.LGBMClassifier(class_weight = class_weight, random_state=42)
        clf.fit(X_train, y_train)

        # predict the results
        y_pred = clf.predict(X_test)

        #visual model evaluvation
        display_model_performance(X_test, y_test, y_pred, 'LGBMClassifier', 'Baseline')
    
    elif args.clf_name == 'lgbmc-tuned':
        #find the optimal parameters for lgbmc
        
        algo = 'lightgbm'

        N_FOLDS = 5
        MAX_EVALS = 10

        def objective(params, n_folds = N_FOLDS):
            clf = lgb.LGBMClassifier(
                                    class_weight = class_weight,
                                    application = 'binary', 
                                    objetive = 'binary',
                                    metric ='auc',                          
                                    **params,random_state=42)
            scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=n_folds,  scoring='roc_auc')
            best_score = max(scores)
            loss = 1 - best_score
            return {'loss': loss, 'params': params, 'status': STATUS_OK}

        #parameters to be optimized
        space = {
            'num_leaves':  hp.choice('num_leaves', range(5,100)),
            'max_bin':  hp.choice('max_bin', range(5,100)),
            'min_data_in_leaf':  hp.choice('min_data_in_leaf', range(300,1000)),
            'num_iterations':  hp.choice('num_iterations', range(100,1000)),
            'min_sum_hessian_in_leaf':  hp.choice('min_sum_hessian_in_leaf', range(20,60)),
            'max_depth':  hp.choice('max_depth', range(3,8)),
            'feature_fraction':  hp.uniform('feature_fraction', 0.2, 0.5),
            'subsample':  hp.uniform('subsample', 0.5, 0.9),
            'bagging_fraction':  hp.uniform('bagging_fraction', 0.5, 0.9),    
            'learning_rate':  hp.uniform('learning_rate', 0.001, 0.1),
            'lambda_l1':  hp.uniform('lambda_l1', 0.0001, 1),
            'lambda_l2':  hp.uniform('lambda_l2', 0.0001, 1)
        }   
            
        tpe_algorithm = tpe.suggest

        # Trials object to track progress
        bayes_trials = Trials()

        # Optimize
        best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

        # Fit 
        clf = lgb.LGBMClassifier(class_weight = class_weight, 
                             application = 'binary', 
                             objetive = 'binary',
                             metric ='auc',
                             num_leaves = best['num_leaves'],
                             max_bin = best['max_bin'],
                             min_data_in_leaf = best['min_data_in_leaf'],
                             min_sum_hessian_in_leaf = best['min_sum_hessian_in_leaf'],
                             max_depth = best['max_depth'],
                             feature_fraction = best['feature_fraction'],
                             subsample = best['subsample'],
                             bagging_fraction = best['bagging_fraction'],
                             learning_rate = best['learning_rate'],
                             lambda_l1 = best['lambda_l1'],
                             lambda_l2 = best['lambda_l2'],
                             random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        display_model_performance(X_test, y_test, y_pred, 'LGBMClassifier', 'Tuned')

    elif args.clf_name == 'logistic':
        clf = LogisticRegression(random_state=42, class_weight=class_weight)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        display_model_performance(X_test, y_test, y_pred, 'LRClassifier', 'Baseline')
    
    
    #Evaluvate model on unseen data 
    df_2018_new = apply_scaler(df_2018.copy(),['transactionAmount','availableCash', 'transaction_proportion'])
    
    # apply encoding
    if args.clf_name == "logistic":
        df_2018_new.drop('merchantCountry', axis=1, inplace=True)
        df_2018_new.drop('mcc', axis=1, inplace=True)
    else:
        df_2018_new['accountNumber_encoded'] = ordinal_encoder.transform(np.array(df_2018_new['accountNumber']).reshape(-1, 1)).astype(int)
    
    df_2018_new.drop('accountNumber', axis=1, inplace=True)
    transformed_unseen = transformer.transform(df_2018_new)
    transformed_df_unseen = pd.DataFrame(transformed_unseen, columns=transformer.get_feature_names_out())
    
    X_unseen = transformed_df_unseen.drop('remainder__target_label', axis=1, ).copy()
    y_unseen = transformed_df_unseen['remainder__target_label'].astype('int')
    
    y_pred_unseen = clf.predict(X_unseen)
    display_model_performance(X_unseen, y_unseen, y_pred_unseen, 'Unseen', args.clf_name)

    #probability
    y_prob = clf.predict_proba(X_unseen)
    label_prob_df = pd.merge(pd.DataFrame(y_prob,columns=['label_0', 'label_1']), y_unseen, left_index=True, right_index=True)
    
    final_df = pd.merge(df_2018,label_prob_df, left_index=True, right_index=True)
    review_df = final_df.sort_values('label_1', ascending=False).iloc[:400]
    print('Prevented transactions: ', review_df['target_label'].value_counts()[1])
    print('Prevented amount: ', review_df[review_df['target_label']==1]['transactionAmount'].sum())