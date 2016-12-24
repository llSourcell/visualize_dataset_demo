
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# step 1: download the data
dataframe_all = pd.read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
num_rows = dataframe_all.shape[0]

# step 2: remove useless data
# count the number of missing elements (NaN) in each column
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]
# remove the columns with missing elements
dataframe_all = dataframe_all[counter_without_nan.keys()]
# remove the first 7 columns which contain no discriminative information
dataframe_all = dataframe_all.ix[:,7:]
# the list of columns (the last column is the class label)
columns = dataframe_all.columns
#print columns

# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = dataframe_all.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)

# step 4: get class labels y and then encode it into number 
# get class label data
y = dataframe_all.ix[:,-1].values
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# original dataset is large, therefore, we sample a sub-set for hyperparameter tunning
np.random.seed(1234)
index = range(len(y))
np.random.shuffle(index)
x_std_random = x_std[index]
x_std_subset = x_std_random[:1000,:]
y_random = y[index]
y_subset = y_random[:1000]

# step 5: split the data into training set and test set
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std_subset, y_subset, test_size = test_percentage, random_state = 0)

# ****  SVM ********************
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Linear SVM accuracy without any transformation: %.2f%%" % (accuracy_svm * 100.0))

# tunning a single parameter
from sklearn.learning_curve import validation_curve
param_range = [0.1, 1.0, 10.0, 100.0, 1000.0]
classifier = SVC(kernel='linear', random_state=0)
# tunning parameter with 10-fold cross validation
train_scores, test_scores=validation_curve(estimator=classifier, X=x_train, y=y_train, param_name='C', param_range=param_range, cv=4)
avg_train_scores = np.mean(train_scores, 1)
std_train_scores = np.std(train_scores, 1)
avg_test_scores = np.mean(test_scores, 1)
std_test_scores = np.std(test_scores, 1)
# print the results
for i in range(len(param_range)):
    print 'when c=%.2f, the training accuracy is %.2f, the test accuracy is %.2f.' % (param_range[i], avg_train_scores[i], avg_test_scores[i])
    
# hyperparameter tunning via grid search
from sklearn.grid_search import GridSearchCV
svm_gs = SVC(random_state = 0)
C_range  =[1.0, 10.0, 100.0, 1000.0]
gamma_range = [0.01, 0.1, 1.0, 10]
# parameter grid [{para1:[value11, value12], para2:[value21, value22]}, {para1:[value], para2:[value2], para3:[value3]}]
param_grid=[{'C': C_range, 'kernel': ['linear']},
            {'C': C_range, 'gamma': gamma_range, 'kernel':['rbf']} ]      
gs = GridSearchCV(estimator = svm_gs, param_grid = param_grid, scoring='accuracy', cv=4, n_jobs=-1)
gs.fit(x_train, y_train)
print 'The best accuracy: ', gs.best_score_
print 'The best parameters: ', gs.best_params_

# use the best paramters obtained from grid search to do cross-validation
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y=y_train, n_folds=4, random_state=0)
scores=[]
svm_best = SVC(kernel='rbf', C=10.0, gamma=0.01, random_state=0)
for k, (train, test) in enumerate(kfold):
    print k
    print train
    print test
    svm_best.fit(x_train[train], y_train[train])
    score=svm_best.score(x_train[test], y_train[test]) # score on test fold
    scores.append(score)
    print 'Round: %d, class distribution: %s, accuracy: %.3f' % (k+1, np.bincount(y_train[train]), score )
# print the average accuracy
print 'Average accuracy: %.3f with standard deviation of %.3f.' % (np.mean(scores), np.std(scores))

# ***** alternative way for k-fold cross validation: more concise
from sklearn.cross_validation import cross_val_score
scores1 = cross_val_score(estimator=svm_best, X=x_train, y=y_train, cv=4, n_jobs=-1)
print 'CV accuracy scores: %s' % scores1
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores1), np.std(scores1)))






