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


from sklearn.preprocessing import LabelEncoder, StandardScaler
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

