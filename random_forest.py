# Importing Libraries
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC



# Load Data
data = pd.read_csv("Data/transactions_obf.csv")
target = pd.read_csv("Data/labels_obf.csv")
print(data.head())

# Prepare Target Labels
y = np.zeros(data.shape[0])
ind_dict = {k: i for i, k in enumerate(data.eventId)}
indices = [ind_dict[x] for x in set(ind_dict).intersection(target.eventId)]
y[indices] = 1

# Data Exploration
data.info()

#Check for the distribution of the transaction amount column
data['transactionAmount'].describe()

#check for the null values in the data
data.isnull().sum()

#check for duplicates in the data
any(data.duplicated())

#Check for the distribution of the labels
plt.hist(y, bins=3)
plt.xticks([0,1])
print(f'overall there are ore only {int(sum(y))} frauds!! ({round(sum(y)*100/len(y),2)}%)')

#Check for the distribution of the transaction amount for fraud and non-fraud transactions
plt.hist(data[data['eventId'].isin(target['eventId'])]['transactionAmount'], bins=100, alpha=0.5, label='Fraud')
plt.hist(data[~data['eventId'].isin(target['eventId'])]['transactionAmount'], bins=100, alpha=0.5, label='Non-Fraud')


# Splitting Data
X_train_full = data.loc[:len(data)-20000-1]
y_train_full = y[:len(data)-20000]

# training set: first 8 months
X_train = X_train_full.loc[:len(X_train_full)-20000-1]
y_train = y_train_full[:len(X_train_full)-20000]

# validation set 1 : 9th month
X_val1 = X_train_full.loc[len(X_train_full)-20000:len(X_train_full)-10000-1]
y_val1 = y_train_full[len(X_train_full)-20000:len(X_train_full)-10000]

# validation set 1 : 10th month
X_val2 = X_train_full.loc[len(X_train_full)-10000:]
y_val2 = y_train_full[len(X_train_full)-10000:]

# test set: last two months 
X_test = data.loc[len(data)-20000:]
y_test = y[len(data)-20000:]

#Reset the Index of the new dataframes/datasets 
# The new datasets we have created still above have the index of the "data" datafame.  We want each dataset to have an independent indexing however, we want also to preserve the relative timing of each transaction. For this reason the old indexes are saved in a new column "temporal_order"
def reset_indexes(dataset):
    save_order_info = False
    if save_order_info:
        return dataset.reset_index().rename(columns={"index": "temporal_order"})#.drop(columns='index')
    else:
        return dataset.reset_index().drop(columns='index')

datasets = [X_train, X_val1, X_val2]
X_train, X_val1, X_val2 = [reset_indexes(datasets[i]) for i in range(3)]

#Feature Engineering

# Identify categorical columns
categorical_cols = ['accountNumber', 'merchantId', 'merchantZip', 'merchantCountry', 'posEntryMode', 'mcc']

# Check the number of unique values in each categorical column
for i in categorical_cols:
    print(f'{i}: {len(X_train[i].unique())}')

#We are going to use the LabelEncoderExt class to encode the categorical variables.
class LabelEncoderExt(object):
    """
    It differs from LabelEncoder by handling new classes (unseen during .fit() ) and providing a value for it [Unknown]
    Unknown will be added in fit and transform will take care of new item. It gives unknown class id
    """
    def __init__(self):

        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        data_list = data_list.astype(str).to_list()
        new_data_list = data_list
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

col_list = ['accountNumber','merchantId', 'merchantZip']

#We are going to encode the categorical variables in the training, validation and test sets.    
X_train_1 = X_train.copy()
X_val1_1 = X_val1.copy()
X_val2_1 = X_val2.copy()

# Encode categorical features
le = LabelEncoderExt()
for col in col_list:
    le.fit(X_train[col])
    X_train_1[col] = le.transform(X_train[col])
    X_val1_1[col] = le.transform(X_val1[col])
    X_val2_1[col] = le.transform(X_val2[col])

#Encoding of Time Information
#This function extracts useful time-based features from the transactionTime column and removes the original timestamp.
def add_time(dataset):
    tm = dataset.copy()
    tm['transactionTime'] = pd.to_datetime(tm['transactionTime'])
    tm['month']  = tm.transactionTime.dt.month
    tm['day']  = tm.transactionTime.dt.day
    tm['hour']  = tm.transactionTime.dt.hour
    tm.drop(columns=['transactionTime'], inplace=True)
    return tm 

#We are going to encode the time information in the training, validation and test sets.
datasets = [X_train_1, X_val1_1, X_val2_1]
X_train_1, X_val1_1, X_val2_1 = [add_time(datasets[i]) for i in range(3)]

print(X_train_1.head())


# Use an evaluation class to evaluate if the addition of new features improves the model performance
#This class is designed to evaluate the performance of a fraud detection model. It train and test the classifier, measures F1 score, confusion matrix, and fraud detection rate and identifies the most important features in fraud detection. 
class Evaluation(object):
    """
    Features evaluation
    We want to test the effect of adding or removing features on a fix classifier

    Example: 
    ev = Evaluation()
    ev.fit(X_traint, y_train) # fit a Random forest classifier with the training set 
    ev.evaluate(X_val2_1, y_val2) # test on validation set

    ev.feature_importance # display features importance
    """
    def __init__(self):
        """
        Create a vanilla Random foreset classifier 
        """
        self.clf = RandomForestClassifier(random_state=2000, n_estimators=500, n_jobs=-1, bootstrap=False)

    def fit(self,X_traint, y_traint):
        """
        obj.fit(self,X_traint, y_traint) # Fit the classifier 
        """
        print('Running fit...')
        self.clf.fit(X_traint, y_traint) 
        print('Running fit... DONE!')
    
    def evaluate(self, X, y, title):
        """
        Estimate confusion matrix, f1 score and percentage of detected anomalies* in a new dataset. 
        """
        self.X = X
        self.y = y
        # Confusion Matrix - Helps understand false positives and false negatives.
        self.conf_matrix =confusion_matrix(y, self.clf.predict(X))
        
        # f1 score - Good for imbalanced fraud detection.
        self.f1_score = f1_score(y,self.clf.predict(X))
        
        # Percent detected in sub-pool
        prob = self.clf.predict_proba(X)
        self.percent_detected = self.Res(prob[:,1], y)
        
        self.instance_num = sum(y)
        self.title = title
        self.display(title)
        return self.f1_score

    def display(self,title):
        '''
        Display Model performance
        '''
        
        print('--- ' + title + ' ---')
        print('confusion matrix')
        print(self.conf_matrix)
        print(f'f1 score = {self.f1_score}')
        print(f'by checking 400 trans. per month we can find {round(self.percent_detected)}% of the frauds (tot {self.instance_num}) ')
        print('')

    def Res(self, fraud_prob, y_test):
        '''
        Calculate fraud detection rate in the 400 most likely fraudulent transactions.         Find the Best 400 Fraud Cases
        Ranks transactions by fraud probability.
        Selects the top 400 highest-risk transactions.
        Calculates the percentage of total fraud cases found.
        '''
        # Get indices of the 400 highest-probability transactions
        ps = fraud_prob.argsort() 
        index_selected_trans = ps[-400:]
        y_sel = y_test[index_selected_trans]
        percent_frauds_detected = sum(y_sel)/sum(y_test)*100

        self.proba = fraud_prob
        self.index_selected_trans = index_selected_trans
        return percent_frauds_detected

    def Saved_Money(self):
        '''
        Calculate Financial Impact of the model and estimates the total fraudulent transaction amount detected.
        '''
        ii = self.index_selected_trans 
        yy = self.y[ii]
        xx = self.X['transactionAmount'].loc[ii]
        self.transAmount_selection = sum(xx.loc[yy==1])
        self.transAmount_total = sum(self.X['transactionAmount'].loc[self.y==1])
        return self.transAmount_selection


    def feature_importance(self):
        '''
        Find the Most Important Features and extracts feature importance scores from the classifier
        Display features importance
        Prints top 5 most important features.
        '''
        feat_imp = pd.DataFrame()
        feat_imp['feature'] = self.clf.feature_names_in_
        feat_imp['importance'] = self.clf.feature_importances_
        feat_imp = feat_imp.sort_values(by='importance', ascending=False).reset_index()
        feat_imp.drop(columns='index')
        self.feat_imp = feat_imp
        print('')
        print('-- Features Importance for classifier --')
        print(feat_imp.loc[:5])
    
    
    def Mutual_Info(self, X, y):
        '''
        Find Features with High Information Gain
        Uses mutual_info_classif() to identify features with the most predictive power.
        Displays the top 5 most informative features.
        '''
        high_score_features = []
        feature_scores = mutual_info_classif(X, y, random_state=0)
        threshold = 5  # the number of most relevant features
        for score, f_name in sorted(zip(feature_scores, X.columns), reverse=True)[:threshold]:
                print(f_name, score)
                high_score_features.append(f_name)

#Test Fraud Detection Performance on the current set of Features
#Creates copies of training and validation datasets preventing modifications to the original data.
X_traint = X_train_1.copy()
X_valit = X_val1_1.copy()
X_val2t = X_val2_1.copy()

#Creates an instance of the Evaluation class to train, evaluate, and analyze the model
ev_1 = Evaluation()
ev_1.fit(X_traint, y_train)
datasets = [X_traint, X_valit, X_val2t]
y_datasets = [y_train, y_val1, y_val2]
title = ['Training Set', 'Validation Set #1', 'Validation Set #2']
det_train, det_val1, det_val2 = [ev_1.evaluate(datasets[i], y_datasets[i], title[i]) for i in range(3)]
# Comments - 
#Using the current dataset with the vanilla classifier, we would detect 38% of frauds in the first validation set and 66% in the second (i.e. using the model to select 400 transactions out of 10.000).
#This is already not bad given the difficulty of the task. 
#Now, Let's check now the contribution of each feature to the performance of the model 
  
#Feature Importance    
ev_1.feature_importance()
# Comment: 
# - The account number, day and transaction amount are the most informative features for fraud detection according to this model.

#Plot the 5 features with stronger Mutual Informationwith respect to the target
ev_1.Mutual_Info(X_traint, y_train)
#Mutual Information is not very informative since it is pretty low for all the features.
 

#Adding fraud probability for each category
#For every categorical feature, I am going to estimate the probability that each value is associated with a certain account (e.g. account = 1 buys from merchant = 1). Then we create a new column for each categorical feature, reporting this probability for each instance(every time we have a account = 1 and merchant = 1 we add the estimated probability). 
#This class creates new features by calculating the probability of a categorical value appearing for each account.
#It adds new probability-based features that help the model understand patterns in how accounts interact with different merchants, locations, and transaction types.
class  Add_cat_prob_per_account(object):
    '''
    Loops through every categorical column 
    For each accountNumber, calculates:
    a)How often each category appears for that account.
    b)Probability of seeing each category for that account.
    Then stores these probabilities in a dictionary and saves them to a .pkl file.
    '''
    def __init__(self):
        if not os.path.isdir('./temp'):
            os.makedirs('./temp')

    def fit(self, X_train, col_list):
        self.col_list = col_list
        for col in col_list:
            # for every account find the prob of each categorical value
            for iAcc in X_train['accountNumber'].unique():
                
                if sum(X_train['accountNumber']==iAcc) == 0:
                    # if there are not transactions for this account stop here
                    continue
                # prob calculation made only on the training set
                tot_transactions_account = len(X_train[col].loc[X_train['accountNumber']==iAcc])
                caunt_per_cat = X_train[col].loc[X_train['accountNumber']==iAcc].value_counts()
                cat_prob = caunt_per_cat / tot_transactions_account

                cat_prob_per_account = {iAcc: cat_prob.to_dict()}

                fname = str(f'./temp/{col}_{iAcc}.pkl')
                with open(fname, 'wb') as f:
                    pickle.dump(cat_prob_per_account, f)
               
            
      
    def transform(self, ds1):
        '''
        transform() Method – Assigning Probabilities to Transactions
        It Creates a new column (col + '_prob') to store probability values.
        Loops through each account and:
        a) Loads the saved probability dictionary from .pkl files.
        b) Assigns probabilities to each transaction based on the stored values.        
        '''
        ds = ds1.copy()
        for col in self.col_list:
            ds[col + '_prob'] = 0 # init the new column/feature 
            # for every account find the prob of each categorical value
            for iAcc in ds['accountNumber'].unique():
                fname = str(f'./temp/{col}_{iAcc}.pkl')

                if not os.path.isfile(fname):
                    continue
                
                with open(fname, 'rb') as f:
                    cat_prob_per_account = pickle.load(f)
       
               
                # estimate the probability for each category and add as a new feature
                cat_prob = cat_prob_per_account[iAcc]#cat_prob_[col]
                keys= list(cat_prob.keys())
                prob= list(cat_prob.values())
                for i in range(len(keys)): # values categorical var
                    cat = keys[i]
                    ds[col + '_prob'].loc[ (ds[col]==cat) & (ds['accountNumber']==iAcc)] = prob[i]
        return ds

#Applying the Transformation to Train & Validation Data
col_list = ['merchantId','merchantZip','mcc','posEntryMode'] 
ap = Add_cat_prob_per_account()
ap.fit(X_train_2, col_list)

#This is Useful in Fraud Detection since: 
#a) It Improves categorical encoding by using probabilities instead of raw values.
#b)  Adds account-specific fraud detection insights:
#- If an account usually shops at merchant A, but suddenly transacts at merchant X, that’s suspicious. ✅ Helps the model understand account behavior trends:
#- A frequent customer at merchant B is more likely to have legitimate transactions there.

datasets = [X_train_1, X_val1_1, X_val2_1]
# ,'posEntryMode',
X_train_3, X_val1_3, X_val2_3 = [ap.transform(datasets[i]) for i in range(3)]

X_val1_3.head()

X_val1_3.columns


##############
# EVALUATION
##############

X_traint = X_train_3.copy()
X_val1t = X_val1_3.copy()
X_val2t = X_val2_3.copy()

ev_4 = Evaluation()
ev_4.fit(X_traint, y_train)
datasets = [X_traint, X_val1t, X_val2t]
y_datasets = [y_train, y_val1, y_val2]
title=['Training set', 'Validation set #1', 'Validation set #2']
det_train, det_val1, det_val2 = [ev_4.evaluate(datasets[i], y_datasets[i], title[i]) for i in range(3)]
ev_4.feature_importance()



## Test different classifiers
#After selecting the best features we are going to select the best classifier. 

#We are going to compare the performance of 5 different classifiers. 

#Performances are estimated using f1 score and area under the ROC curve. 

### Standardize all features

#The random forest classifier does not require normalization. However, this is important for the other classifiers we are going to test.
X_traint = X_train_1.copy()
X_val1t = X_val1_1.copy()
X_val2t = X_val2_1.copy()

ss = StandardScaler()

X_train_ss = pd.DataFrame(ss.fit_transform(X_traint), columns=X_traint.columns)
X_val1_ss  = pd.DataFrame(ss.transform(X_val1t), columns=X_val1t.columns)
X_val2_ss  = pd.DataFrame(ss.transform(X_val2t), columns=X_val2t.columns)


#We are going to test the performance of the following classifiers:

# test 8 different models 
classifiers_labels = [
    'KNeighborsClassifier',
    'LogisticRegression',
    #OneClassSVM(gamma='scale', nu=0.01),
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier',
    'LGBMClassifier',
    'CatBoostClassifier',
    'SVC'
    ]


#The classifiers are tested using the F1 score and the area under the ROC curve.
# evaluation is based on f1_score even if also area under the ROC curve is shown
classifiers = [
    KNeighborsClassifier(algorithm="kd_tree"),
    LogisticRegression( class_weight='balanced'),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier(),
    LGBMClassifier(),
    CatBoostClassifier(verbose=0),
    SVC(probability=True)
    ]

#Initializing Performance Metrics Lists i.e. f1, auc1, auc2 and tt
#f1 → Stores F1 scores (important for imbalanced fraud data).
#auc1 → Stores Area Under ROC Curve (AUC) scores.
#auc2, tt → These variables are not used in the code.
f1, auc1, auc2, tt = [], [],[], []
for classifier in classifiers:
    '''
    
    '''
    steps = [('clf', classifier)]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train_ss, y_train) 

    # test on the two validation sets   
    X = [X_val1_ss,X_val2_ss]
    Y = [y_val1,y_val2]
    
    # estimate f1 score
    #------------------
    ff =0
    for i in range(2):
        ff+=f1_score(Y[i], pipeline.predict(X[i]))

    f1.append(ff/2)
    # print(f'f1 = {ff/2}')
    
    # estimate area under the ROC curve
    #------------------
    #y_val_decisions = pipeline.decision_function(X_val) # you can also use the dicision values 
    y_val_pred_prob = pipeline.predict_proba(X_val1_ss)
    y_val_prob = y_val_pred_prob[:,1]
    
    auc1.append(roc_auc_score(y_val1, y_val_prob))
    # print(roc_auc_score(y_val1, y_val_prob))
    
    classifiers_labels = [
    'K Neighbors',
    'Logistic Regression',
    #OneClassSVM(gamma='scale', nu=0.01),
    'Random Forest',
    'AdaBoost',
    'Gradient-Boosting'
    ]
    
Summ = pd.DataFrame({'Classifier':classifiers_labels,'f1_score':f1,'AUC':auc1})
Summ.sort_values(by='f1_score', inplace=True, ascending=False)
Summ 


    














# Handle Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_full, y_train_full)

# Standardization
numeric_cols = X_train_resampled.select_dtypes(include=[np.number]).columns  # Select only numerical columns
ss = StandardScaler()
X_train_ss = pd.DataFrame(ss.fit_transform(X_train_resampled[numeric_cols]), columns=numeric_cols)
X_test_ss = pd.DataFrame(ss.transform(X_test[numeric_cols]), columns=numeric_cols)

# Random Forest Model with Optimized Hyperparameters
clf = RandomForestClassifier(
    random_state=2000, n_estimators=500, n_jobs=-1, bootstrap=False, class_weight='balanced'
)

clf.fit(X_train_ss, y_train_resampled)

# Model Evaluation
y_pred = clf.predict(X_test_ss)
y_pred_proba = clf.predict_proba(X_test_ss)[:, 1]
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(10,4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.show()

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'F1 Score: {f1}')
print(f'ROC AUC Score: {roc_auc}')
