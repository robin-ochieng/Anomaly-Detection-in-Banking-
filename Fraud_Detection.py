#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=12)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif

# models 
from sklearn.svm import SVC, NuSVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras

# Data Importation and Formatting
data = pd.read_csv(r"C:/Users/Robin Ochieng/OneDrive - Kenbright/Gig/Python Projects/Fraud Detection/Data/transactions_obf.csv")
target = pd.read_csv(r"C:/Users/Robin Ochieng/OneDrive - Kenbright/Gig/Python Projects/Fraud Detection/Data/labels_obf.csv")
print('data loaded')

#----------------------
# Labels Preparation
#----------------------
# Format target labels: 
# fraud labels are provided as events. 
# To train models we need to convert them to 
# a list of 0 or 1 for each transaction
ind_dict = dict((k,i) for i,k in enumerate(data.eventId))
inter = set(ind_dict).intersection(target.eventId)
indices = [ ind_dict[x] for x in inter ]
indices.sort()
y = np.zeros(data.shape[0])
y[indices] = 1


# Data Exploration
data.head()
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
 #The labels are highly imbalanced with only 0.74% of the data being frauds.
 #There are many ways to deal with imbalanced data:
    #1. Oversampling the minority class
    #2. Undersampling the majority class
    #3. Using a different metric to evaluate the model
    
#Since Accuracy is not a good metric for imbalanced data, we will use the F1 score and Area under the ROC curve to evaluate the models.

#EDA
#I would like to check for:
    #1. Transactions/ frauds across time
    #2. Number of Frauds per user/account
    #3. Amount of monetary frauds per user/account

#prepare data for display
df = data.copy()
df['transactionTime'] = pd.to_datetime(data['transactionTime'])
df['month'] = df['transactionTime'].dt.month

# 1) time-course transactions per account with fraud events 
#-------------------------
mpl.rc('axes', labelsize=12)
col = ['k', 'r']


fig, axs = plt.subplot_mosaic([['a)', 'c)'], 
                               ['b)', 'c)'], 
                               ['b)', 'd)'],
                               ['b)', 'd)']],
                               figsize=(10,8))


plt.axes(axs['b)'])
acc_with_frauds = sorted(df['accountNumber'].loc[y==1].unique().tolist())
for i in range(50):
    acc = acc_with_frauds[i]
    fraud = y[df['accountNumber']==acc]
    xx = df['transactionTime'].loc[df['accountNumber']==acc]
    
    plt.scatter(xx, np.ones(len(xx))*i, marker='.', c=fraud, cmap='cool', alpha=0.3)
   
    plt.scatter(xx[fraud==1], np.ones(len(xx[fraud==1]))*i, c='r',cmap='cool')

plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Accounts')
plt.title('transactions per account')


#2) Frauds per month
#-------------------------
plt.axes(axs['a)'])
plt.hist(df['transactionTime'].loc[y==1], bins=12, color='r')
plt.xticks([])
plt.ylabel('Frauds number')
plt.title('Frauds per month')


#3) Number of frauds per account
#-------------------------
label_encoder = LabelEncoder()
df = data.copy()
df.accountNumber = label_encoder.fit_transform(df.accountNumber)
df['transactionTime'] = pd.to_datetime(df['transactionTime'])

num_fraud_per_account = []
for acc in range(max(df['accountNumber'])):
    num_fraud_per_account.append(sum(y[df['accountNumber']==acc]))
num_fraud_per_account = np.array([num_fraud_per_account])


tm = num_fraud_per_account[np.where(num_fraud_per_account>0)]

plt.axes(axs['c)'])
plt.hist(tm, bins=np.arange(0,60,3))
plt.title('Num. of frauds per account')
plt.ylabel('Accounts with fraud')

# Fraud amount per account
#-------------------------
num_accounts_with_fraud = len(num_fraud_per_account[np.where(num_fraud_per_account>0)])
df_fraud = df.loc[y==1] # take only fraud trans.
fraud_amount_per_account = df_fraud.groupby('accountNumber')['transactionAmount'].sum()

plt.axes(axs['d)'])
plt.hist(fraud_amount_per_account)
plt.title('Fraud amount per account (GBP)')
plt.ylabel('Accounts with fraud')


plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)

for label, ax in axs.items():
    ax.set_title(label, fontfamily='serif', loc='left', fontsize=16)

# plt.figtext(0.4, -0.1, txt, wrap=True, horizontalalignment='center', fontsize=12)

#Modelling
# 1). Data Split
#We are going to use the last two months of transactions as a test set and the two months before as validation set
#Our model must take one month of transactions (~10.000 events) and return the 400 ones more likely to be frauds. Consider that for 10.000 transactions we have less than 100 frauds. 

#We are going to use the last two months of transactions as test set and the two months before as validation set. 
#As we saw in Fig1a, the number of frauds per month is reasonably similar. 

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

# 2) Feature Engineering
#We are going to create some new features to test if they improve the model performance. 

#Categorical feature encoding
#Check the number of labels in each category
# categorical var (we removed 'transactionTime' and 'eventId')
col_list = ['accountNumber','merchantId', 'merchantZip', 'merchantCountry', 'posEntryMode','mcc']
for i in col_list:
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
    
#We are going to encode the categorical variables in the training, validation and test sets.    
X_train_1 = X_train.copy()
X_val1_1 = X_val1.copy()
X_val2_1 = X_val2.copy()
# categorical var (we removed 'transactionTime' and 'eventId')
col_list = ['accountNumber','merchantId', 'merchantZip']
# encoding
le = LabelEncoderExt()
for col in col_list:
    le.fit(X_train[col])
    X_train_1[col] = le.transform(X_train[col])
    X_val1_1[col] = le.transform(X_val1[col])
    X_val2_1[col] = le.transform(X_val2[col])
    
#Encoding of Time Information
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

#Logarithmic transformation of the transaction amount
def Conv_to_log(dataset):
    dataset['transactionAmount'].loc[dataset['transactionAmount']<5] = 5 # this trick simplifies working with the log
    dataset['transactionAmount'] = dataset['transactionAmount'].apply(np.log1p)
    return dataset

datasets = [X_train_1, X_val1_1, X_val2_1]
col_list = ['merchantId', 'merchantZip','mcc','posEntryMode'] 
X_train_1, X_val1_1, X_val2_1= [Conv_to_log(datasets[i]) for i in range(3)]

#Drop eventid column
X_train_1.drop(columns=['eventId'],inplace=True)#'transactionTime',
X_val1_1.drop(columns=['eventId'],inplace=True)
X_val2_1.drop(columns=['eventId'],inplace=True)

# Use an evaluation class to evaluate if the addition of new features improves the model performance
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
        # Confusion Matrix
        self.conf_matrix =confusion_matrix(y, self.clf.predict(X))
        
        # f1 score
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
        calculate percentage of frauds in the 400 more likely transactions
        '''
        # get the index of the 400 instances with higer probability  
        ps = fraud_prob.argsort() 
        index_selected_trans = ps[-400:]
        y_sel = y_test[index_selected_trans]
        percent_frauds_detected = sum(y_sel)/sum(y_test)*100

        self.proba = fraud_prob
        self.index_selected_trans = index_selected_trans
        return percent_frauds_detected

    def Saved_Money(self):
        ii = self.index_selected_trans 
        yy = self.y[ii]
        xx = self.X['transactionAmount'].loc[ii]
        self.transAmount_selection = sum(xx.loc[yy==1])
        self.transAmount_total = sum(self.X['transactionAmount'].loc[self.y==1])
        return self.transAmount_selection


    def feature_importance(self):
        '''
        Display features importance
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
        high_score_features = []
        feature_scores = mutual_info_classif(X, y, random_state=0)
        threshold = 5  # the number of most relevant features
        for score, f_name in sorted(zip(feature_scores, X.columns), reverse=True)[:threshold]:
                print(f_name, score)
                high_score_features.append(f_name)

#Test Fraud Detection Performance on the current set of Features
X_traint = X_train_1.copy()
X_valit = X_val1_1.copy()
X_val2t = X_val2_1.copy()

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
 
#Encode posEntryMode with a OneHotEncoder(dummies)
class one_hot(object):
    """
    Ignore new labels in validation set
    """
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X_train, col):
        self.target_column = col
        cat = X_train[[col]]
        self.ohe.fit(cat)
    
    def transform(self, X_test):
        cat = X_test[[self.target_column]]
        cat2 = self.ohe.transform(cat).toarray()

        new_col_names = []
        [new_col_names.append(self.target_column + '_' + str(c)) for c in list(self.ohe.categories_[0])]
        
        
        X_test = pd.concat([X_test, pd.DataFrame(cat2, columns=new_col_names)], axis='columns') # , columns= new_col_names
        return X_test


oo = one_hot()
oo.fit(X_train_1,'posEntryMode') # 'posEntryMode'
X_train_2 = oo.transform( X_train_1.copy())
X_val1_2 = oo.transform(X_val1_1.copy())
X_val2_2 = oo.transform(X_val2_1.copy())
 
 # EVALUATION
X_traint = X_train_2.copy()
X_val1t = X_val1_2.copy()
X_val2t = X_val2_2.copy()

ev_2 = Evaluation()
ev_2.fit(X_traint, y_train)
datasets = [X_traint, X_val1t, X_val2t]
y_datasets = [y_train, y_val1, y_val2]
title=['Training set', 'Validation set #1', 'Validation set #2']
det_train, det_val1, det_val2 = [ev_2.evaluate(datasets[i], y_datasets[i], title[i]) for i in range(3)]
ev_2.feature_importance()

#This operation adds several new colums without adding much information, therefore we will not use it in the final model.


#Adding fraud probability for each category
#For every categorical feature, I am going to estimate the probability that each value is associated with a certain account (e.g. account = 1 buys from merchant = 1). Then we create a new column for each categorical feature, reporting this probability for each instance(every time we have a account = 1 and merchant = 1 we add the estimated probability). 
import pickle
import os

class  Add_cat_prob_per_account(object):
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

col_list = ['merchantId','merchantZip','mcc','posEntryMode'] 
ap = Add_cat_prob_per_account()
ap.fit(X_train_2, col_list)

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

X_val1_3.columns

### Create new features with unsuprevised learning: Isolation forest
# Isolation forest is one of the most effective unsupervised algorithms for anomaly detection
# We use it to generate new features to use with our supervised learning approach.

from sklearn.ensemble import IsolationForest
X_train_4= X_train_2.copy()
X_val1_4= X_val1_2.copy()
X_val2_4= X_val2_2.copy()
X_test_4= X_test_2.copy()
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(X_train_2)
X_train_4['scores']=model.decision_function(X_train_2)
X_train_4['anomaly']=model.predict(X_train_2)

X_val1_4['scores']=model.decision_function(X_val1_2)
X_val1_4['anomaly']=model.predict(X_val1_2)

X_val2_4['scores']=model.decision_function(X_val2_2)
X_val2_4['anomaly']=model.predict(X_val2_2)

X_test_4['scores']=model.decision_function(X_test_2)
X_test_4['anomaly']=model.predict(X_test_2)

##############
# EVALUATION
##############

X_traint = X_train_4.copy()
X_val1t = X_val1_4.copy()
X_val2t = X_val2_4.copy()
X_testt = X_test_4.copy()

ev_4 = Evaluation()
ev_4.fit(X_traint, y_train)
datasets = [X_traint, X_val1t, X_val2t]
y_datasets = [y_train, y_val1, y_val2]
title=['Training set', 'Validation set #1', 'Validation set #2']
det_train, det_val1, det_val2 = [ev_4.evaluate(datasets[i], y_datasets[i], title[i]) for i in range(3)]
ev_4.feature_importance()


#Comment: Also these new features only make the performance worse for the second validation set and therefore are dropped.

#Surprisingly, it turns out that the dataset with less new features (X_train_2) produces the best performance.  

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


# test 5 different models 
classifiers_labels = [
    'KNeighborsClassifier',
    'LogisticRegression',
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'GradientBoostingClassifier'
    ]
# evaluation is based on f1_score enven if also area under the ROC curve is shown
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
classifiers = [
    KNeighborsClassifier(algorithm="kd0_tree"),
    LogisticRegression( class_weight='balanced'),
    # OneClassSVM()
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]

f1, auc1, auc2, tt = [], [],[], []
for classifier in classifiers:
    #print(classifier)
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
    'Random Forest',
    'AdaBoost',
    'Gradient-Boosting'
    ]
Summ = pd.DataFrame({'Classifier':classifiers_labels,'f1_score':f1,'AUC':auc1})
Summ.sort_values(by='f1_score', inplace=True, ascending=False)
Summ 

#### Comment: 
#the Random Forest classifier is the one producing the best performance

## Random forest Optimization 
X_train_val1 = pd.concat([X_train_1, X_val1_1, X_val2_1],axis=0,ignore_index=True)
y_train_val1 = np.concatenate([y_train, y_val1 , y_val2])
# Create a list where train data indices are -1 and validation data indices are 0
split_index = np.concatenate([np.ones(X_train_1.shape[0])*-1, np.zeros(X_val1_1.shape[0]), np.ones(X_val2_1.shape[0])])
# Use the list to create PredefinedSplit
pds = PredefinedSplit(test_fold = split_index)

rf=RandomForestClassifier(random_state=2000, n_jobs=-1)
rf2 = Pipeline([
	# ("kmeans", KMeans()),
	("rndf_clf", RandomForestClassifier(random_state=2000, n_jobs=-1))
])

# Create the random grid
random_grid = {'n_estimators': [400, 500, 600],
               'max_features': [2, 4, 8, 'auto'],
               'max_depth': [2, 5, None],
               'bootstrap': [True, False],
               'class_weight':['balanced', None]
               }
print(random_grid)

# Random search of parameters, using 3 fold cross validation,
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = pds, verbose=2, random_state=2000, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_val1,y_train_val1)

# Best parameters
rf_random.best_params_

# This were the best parameters found
'''
{'n_estimators': 500,
 'max_features': 'auto',
 'max_depth': None,
 'class_weight': 'balanced',
 'bootstrap': False}
'''
### Evaluation of optimized classifier

#######################
# Evaluation
######################

X_traint = X_train_ss.copy()
X_val1t = X_val1_ss.copy()
X_val2t = X_val2_ss.copy()

# Optimized hyperparameters
clf = RandomForestClassifier(random_state=2000, n_estimators=500, n_jobs=-1,  bootstrap = False)#max_features = 4, class_weight='balanced', , n_estimators=300, max_features='auto'
clf.fit(X_traint, y_train) 

# Evaluate the model
ev_5 = Evaluation()
ev_5.clf= clf

# Evaluate the model on the training, validation and test sets
datasets = [X_traint, X_val1t, X_val2t]
y_datasets = [y_train, y_val1, y_val2]
title=['Training set', 'Validation set #1', 'Validation set #2']
det_train, det_val1, det_val2 = [ev_5.evaluate(datasets[i], y_datasets[i], title[i]) for i in range(3)]
ev_5.feature_importance()

# Display precision/recall vs threshold
def plot_precision_recall_vs_threshold(clf, X, y): 
    '''plot_precision_recall_vs_threshold(clf, X)'''
    
    y_scores = clf.predict_proba(X)[:,1]
    #y_scores = rnf_clf.decision_function(X_val)
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('threshold')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(recalls,precisions)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_precision_recall_vs_threshold(clf, X_val1t, y_val1)


## Evaluation of the best model on the test set
#- The model was trained using 10 consecutive months of transactions, exclueded 20.000 randomly selected transactions that we used as validation set.
#- Here we evaluate the performance of the model on two new months of transactions (20.000 transactions). 
#- We generate 20 random subsamples, each one corresponding to one month of transactions (i.e. 10.000 trans.), from the test set (i.e. bootstrap). 
#- For each subsample, we use the model to select the 400 transactions more likely to be a fraud. Then, we estimate the percentage of the total fraud transactions of the month we have found. 
#- As a control, we also show the average percentage of frauds we can detect by randomly selecting 400 transactions from the same subsamples.  
#- Furthermore, we show the average percentage of frauds we can detect using the model from the validation set. 


# Pre-processing
def PreProcess(X_train, X_test):
    '''
    X_train_pp, X_test_pp = PreProcess(X_train, X_test) 

    Transform the row data using the pre-processing steps selected using the validation sets
    '''
    datasets = [X_train, X_test]
    # 1. reset index
    X_train, X_test = [reset_indexes(datasets[i]) for i in range(2)]
    # 2. transform categorical features 
    col_list = ['accountNumber','merchantId', 'merchantZip']
    le = LabelEncoderExt()
    for col in col_list:
        le.fit(X_train[col])
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # 3. Add time features 
    datasets = [X_train, X_test]
    X_train, X_test = [add_time(datasets[i]) for i in range(2)]

    # 4. Drop eventId
    X_train.drop(columns=['eventId'],inplace=True)
    X_test.drop(columns=['eventId'],inplace=True)

    # 5. Standardize
    ss = StandardScaler()
    X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)
    return X_train, X_test



X_train_full_ss, X_test_ss = PreProcess(X_train_full, X_test) 

#Training
# Optimized hyperparameters
clf = RandomForestClassifier(random_state=2000, n_estimators=500, n_jobs=-1,  bootstrap = False)#max_features = 4, class_weight='balanced', , n_estimators=300, max_features='auto'
clf.fit(X_train_full_ss, y_train_full) 

# Evaluate the model
X_testt= X_test_ss.copy()
ev_6 = Evaluation()
ev_6.clf = clf

N = 10000 # sample size corresponding to 1 month
percent_frauds, percent_frauds_control = [], []
money_saved, total_money  = [], []
for n_tests in range(20):

    # Test set
    ii = np.random.RandomState(seed=n_tests ).permutation(np.arange(1,X_testt.shape[0],1))
    bs_index = ii[:N]
    xtt = X_testt.iloc[bs_index,:].copy()
    ytt = y_test[bs_index]
    y_val_prob = clf.predict_proba(xtt)
    s = ev_6.Res(y_val_prob[:,1], ytt)
    percent_frauds.append(s)

    #money_saved.append( ev_6.Saved_Money() )
    #ev_5.X = xtt
    #ev_5.y = ytt
    #total_money.append(ev_5.transAmount_total)
    #print(f'test set: detected {s} % ')
    
    # Control: random selection 
    ii = np.random.RandomState(seed=n_tests ).permutation(np.arange(1,len(y_test),1))
    sens = sum(y_test[ii[:400]])/sum(y_test)*100
    percent_frauds_control.append(sens)
    #print(f'random selection: detected {sens}%')
    
    
print('Average across 20 bootstrapped datasets')
print(f'1) model selection on test set {np.mean(percent_frauds)} % ')
print(f'2) random selection on test set {np.mean(percent_frauds_control)} %')
#print(f'3) model selection on validation set {np.mean(percent_frauds_cv)}')  
# print(f'4) Money saved per month {np.mean(money_saved)} over {np.mean(total_money)} GBP')  

#### Comment 
#If our client was randomly selecting 400 transactions per month to check for frauds, it would detect less than the 2% of the them. On the other hand, **using my model it would detect 15 times more frauds (the 30%)!**

#This classification task is challenging for many reasons: 
#- the unbalance in the categories (only 1% of frauds)
#- the very high detection threshold (we need to select 400 trials out of 10.000)
#- fraudsters constantly optimize their strategy to escape anomaly detection strategies
#- Finally, since we train the model over a certain period of time and then predict frauds from a different period (two new months), this is an **extrapolation problem**. 
#Since the behaviors of the owners of the accounts and the fraudsters change over time, generalizations to future transactions are difficult.   

#The model performance would be way better if we were randomly sampling the test set transactions over the same period of time as the training set. 

#I am going to show that in the next session. 


## Model peformance: test set from the same period of time as the training set 
# Split the data by random sampling 
# --------------------------------
# Create a training set and two validation sets 
splitter_test = StratifiedShuffleSplit(n_splits=2, 
    test_size=20000, random_state=1000)

for train_index, test_index in splitter_test.split(data, y):
    X_train, X_test1 = data.loc[train_index], data.loc[test_index]
    y_train, y_test1 = y[train_index], y[test_index]


# Pre-processing 
# --------------------------------
X_train_control, X_test_control = PreProcess(X_train, X_test1) 

clf_2 = RandomForestClassifier(random_state=2000, n_estimators=500, n_jobs=-1,  bootstrap = False)#max_features = 4, class_weight='balanced', , n_estimators=300, max_features='auto'
clf_2.fit(X_train_control, y_train) 

# Evaluation
# --------------------------------
X_testt= X_test_control.copy()
ev_7 = Evaluation()
ev_7.clf = clf_2

N = 10000 # sample size corresponding to 1 month

percent_frauds_contr2 = [] 
money_saved, total_money  = [], []
for n_tests in range(20):

    # Test set
    ii = np.random.RandomState(seed=n_tests ).permutation(np.arange(1,X_testt.shape[0],1))
    bs_index = ii[:N]
    xtt = X_testt.iloc[bs_index,:].copy()
    ytt = y_test1[bs_index]
    y_val_prob = clf.predict_proba(xtt)
    s = ev_7.Res(y_val_prob[:,1], ytt)
    percent_frauds_contr2.append(s)

print(f'2) random selection on test set {np.mean(percent_frauds_contr2)} %')

#### Comment 
#As expected, If the test data are sampled from the same period of time as the training data (cross-validation) the performance of the model raises to 91% on average. 

#This confirms that the model does a pretty good job in predicting new frauds. The limitations in its performance depend on the variability of the data across time. 

## Summary Figure
a = pd.DataFrame()
a['w/o model'] = percent_frauds_control # random selection
a['with model \n (extrapol.)'] = percent_frauds
a['with model \n (interpol.)'] = percent_frauds_contr2

plt.figure(figsize=(6,5))
a.boxplot()
plt.ylabel('frauds detected (%)')
plt.title('Model Performance on Test set')
plt.ylim([-5,100])

# task: select the 400 transactions over 10000 with the higer probability to be a fraud
#txt="task: select the 400 transactions over 10000 with the higer probability to be a fraud"
#plt.figtext(0.4, -0.1, txt, wrap=True, horizontalalignment='center', fontsize=12)

## Summary 
#The goal of this study was the selection of a machine learning pipeline to predict the probability of frauds based on trasactions features. 
#I created many diffent new features and tested several classification algorithms (supervised and unsupervised). 
#I found that a Random Forest Classifier produces the best performance. 
#I optimized the hyperparameters of this classifier and estimated the performance on the test set. 
#Overall, using this model our client will detect 15 times more frauds than by randomly selecting the transactions to check. 

## Neural network
#I am going to test if a NN can produce better performance on this task

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_train_ss = pd.DataFrame(ss.fit_transform(X_train_1), columns=X_train_1.columns)
X_val1_ss  = pd.DataFrame(ss.transform(X_val1_1), columns=X_val1_1.columns)
X_val2_ss  = pd.DataFrame(ss.transform(X_val2_1), columns=X_val2_1.columns)

X_traint = X_train_ss.copy()
X_val1t  = X_val1_ss.copy()
X_val2t  = X_val2_ss.copy()

## Optimize NN
n_feat = X_train_ss.shape[-1]
def build_model(n_hidden=0, n_neurons=10, learning_rate=1e-3, dropout=0,  n_feat=n_feat):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_feat, input_dim=n_feat, activation='relu',kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(rate=dropout))
    for layer in range(n_hidden):
        keras.layers.Dropout(rate=dropout)
        model.add(keras.layers.Dense(n_neurons, activation='relu', kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])
    return model


keras_reg = keras.wrappers.scikit_learn.KerasClassifier(build_model)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True) # stop training earlier if there are no progresses

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    'n_hidden':[0,1,2,3],
    'n_neurons':[5, 50, 100],
    'dropout': [0, 0.2, 0.5],
    'learning_rate':[0.001, 0.0001, 0.01]
}


X_train_val1 = pd.concat([X_train_ss, X_val1_ss, X_val2_ss],axis=0,ignore_index=True)
y_train_val1 = np.concatenate([y_train, y_val1 , y_val2])
# Create a list where train data indices are -1 and validation data indices are 0

combine_val_sets = True
if combine_val_sets:
    # combine validation sets (to increase speed)
    split_index = np.concatenate([np.ones(X_train_1.shape[0])*-1, np.zeros(X_val1_1.shape[0]+ X_val2_1.shape[0])])
else:
    # keep the two validation sets separated 
    split_index = np.concatenate([np.ones(X_train_1.shape[0])*-1, np.zeros(X_val1_1.shape[0]), np.ones(X_val2_1.shape[0])])



# Use the list to create PredefinedSplit
pds = PredefinedSplit(test_fold = split_index)


rnf_search = RandomizedSearchCV(keras_reg, param_distribs, n_iter=50, cv=pds, verbose=2, random_state=2000, n_jobs = -1)

# Fit the random search model
BATCH_SIZE = 32
history = rnf_search.fit(X_train_val1, y_train_val1, epochs=20, 
            validation_data=(X_val1_ss, y_val1), 
            callbacks=[early_stopping_cb],
            batch_size=BATCH_SIZE)

rnf_search.best_params_
# {'n_neurons': 5, 'n_hidden': 1, 'learning_rate': 0.001, 'dropout': 0}

### Optimized NN model on validation set 
n_feat = X_traint.shape[-1]
model = keras.models.Sequential([
    keras.layers.Dense(n_feat, input_dim=n_feat, activation='relu',kernel_initializer="he_normal" ), # ,kernel_regularizer=keras.regularizers.l2() ,kernel_regularizer=keras.regularizers.l2(0.001)
    keras.layers.Dense(1, activation='sigmoid') # one output neuron for each class
])
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['AUC'])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True) # stop training earlier if there are no progresses

BATCH_SIZE = 32
history = model.fit(X_traint, y_train, epochs=30, 
                    validation_data=(X_val1t, y_val1), 
                    callbacks=[early_stopping_cb],
                    batch_size=BATCH_SIZE)

### Learning curves of the traning and validation sets  
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid()
plt.gca().set_ylim(0,1)

#### Comment: 
#The AUC of training and validation sets are very similar, therefore the model is not overfitting. 

### Evaluation on validation set
def Res(y_val_prob, y_test):
    # find index of pool with higher prob
    prob = np.sort(y_val_prob)
    pp_thr = prob[-399]
    ii = np.where(y_val_prob>pp_thr)
    yy_val = y_test[ii]
    sens = sum(yy_val)/sum(y_test)*100
    print(f'by checking 400 trans. per month we can find {round(sens)}% of the frauds (tot {sum(y_test)}) ')
    #return sens


y_val_prob =  model.predict(X_val1t)


p = y_val_prob[:,-1].copy()
p[p<0.5] = 0
p[p>=0.5] = 1

conf =confusion_matrix(y_val1, p )
print(conf)

f1 = f1_score(y_val1, p)
print(f1)

Res(y_val_prob[:,-1], y_val1)

#---
y_val_prob =  model.predict(X_val2t)
p = y_val_prob[:,-1].copy()
p[p<0.5] = 0
p[p>=0.5] = 1
conf =confusion_matrix(y_val2, p )
print(conf)

f1 = f1_score(y_val2, p)
print(f1)

Res(y_val_prob[:,-1], y_val2)

### Training on training set + validation set to evaluate on test set
n_feat = X_train_full_ss.shape[-1]
model = keras.models.Sequential([
    keras.layers.Dense(n_feat, input_dim=n_feat, activation='relu',kernel_initializer="he_normal" ), # ,kernel_regularizer=keras.regularizers.l2() ,kernel_regularizer=keras.regularizers.l2(0.001)
    keras.layers.Dense(1, activation='sigmoid') # one output neuron for each class
])
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['AUC'])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True) # stop training earlier if there are no progresses

BATCH_SIZE = 32
history = model.fit(X_train_full_ss, y_train_full, epochs=30, 
                    validation_data=(X_test_ss, y_test), 
                    callbacks=[early_stopping_cb],
                    batch_size=BATCH_SIZE)

history.params

### Display learning curve test set
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid()
plt.gca().set_ylim(0,1)

#### Comment
#Here the overfitting is very severe

### Evaluation on the test set
X_testt= X_test_ss.copy()
ev_7 = Evaluation()
ev_7.clf = model

N = 10000 # sample size corresponding to 1 month
percent_frauds, percent_frauds_control, percent_frauds_cv = [], [], []
money_saved, total_money  = [], []
for n_tests in range(20):

    # Test set
    ii = np.random.RandomState(seed=n_tests ).permutation(np.arange(1,X_testt.shape[0],1))
    bs_index = ii[:N]
    xtt = X_testt.iloc[bs_index,:].copy()
    ytt = y_test[bs_index]

    # adapted 
    y_val_prob = model.predict(xtt)
    
    s = ev_7.Res(y_val_prob[:,-1], ytt)
    
    percent_frauds.append(s)

    # Control: random selection 
    ii = np.random.RandomState(seed=n_tests ).permutation(np.arange(1,len(y_test),1))
    sens = sum(y_test[ii[:400]])/sum(y_test)*100
    percent_frauds_control.append(sens)
    #print(f'random selection: detected {sens}%')
    
    
print('Average across 20 bootstrapped datasets')
print(f'1) model selection on test set {np.mean(percent_frauds)} % ')
print(f'2) random selection on test set {np.mean(percent_frauds_control)} %')
#print(f'3) model selection on validation set {np.mean(percent_frauds_cv)}')

#### Comment 
#The performance of a shallow NN is half the one of the Random forest. 