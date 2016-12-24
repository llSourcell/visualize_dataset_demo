'''
Comaprison results among various classifiers:
    
Logistic Regression Accuracy: 74.68%
Linear SVM Accuracy: 79.98%
KNN Accuracy: 97.76%
Decision Tree Accuracy: 43.91%
Naive Bayes Accuracy: 51.81%
AdaBoost Accuracy: 97.66%
Bagging Accuracy: 99.29%
Random Forest Accuracy: 99.85%
XGBoost Accuracy: 95.31%
'''
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

# step 5: split the data into training set and test set
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)

# step 6: compare different classifiers

# ************** Single Classfier ******************************
# *** 1) Logistic Regression *******
from sklearn.linear_model import LogisticRegression
#initialize the classifier instance
model1 = LogisticRegression(C=1000.0, random_state = 0)
# train the model
model1.fit(x_train, y_train)
# evaluation on test set
y_pred1 = model1.predict(x_test)
accuracy1 = accuracy_score(y_test, y_pred1)
print("Logistic Regression Accuracy: %.2f%%" % (accuracy1 * 100.0))

# **** 2) SVM ********************
from sklearn.svm import SVC
model2 = SVC(kernel='linear', C=1.0, random_state=0)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Linear SVM Accuracy: %.2f%%" % (accuracy2 * 100.0))

# ****** 3) kNN ********************
from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
accuracy3 = accuracy_score(y_test, y_pred3)
print("KNN Accuracy: %.2f%%" % (accuracy3 * 100.0))

# ***** 4) decision tree **********
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state = 0)
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
accuracy4 = accuracy_score(y_test, y_pred4)
print("Decision Tree Accuracy: %.2f%%" % (accuracy4 * 100.0))

# ****** 5) Naive Bayes ********************
from sklearn.naive_bayes import GaussianNB
model5 = GaussianNB()
model5.fit(x_train, y_train)
y_pred5 = model5.predict(x_test)
accuracy5 = accuracy_score(y_test, y_pred5)
print("Naive Bayes Accuracy: %.2f%%" % (accuracy5 * 100.0))

# ************** Ensemble Learning ***************************
# ****** 6) AdaBoost ********************
from sklearn.ensemble import AdaBoostClassifier
tree6 = DecisionTreeClassifier(criterion='entropy')
model6 = AdaBoostClassifier(base_estimator=tree6, n_estimators=60, learning_rate=0.1, random_state=0)
model6.fit(x_train, y_train)
y_pred6 = model6.predict(x_test)
accuracy6 = accuracy_score(y_test, y_pred6)
print("AdaBoost Accuracy: %.2f%%" % (accuracy6 * 100.0))

# ****** 7) Bagging ********************
from sklearn.ensemble import BaggingClassifier
tree7 = DecisionTreeClassifier(criterion='entropy')
model7 = BaggingClassifier(base_estimator=tree7, n_estimators=60, max_samples=1.0, max_features=1.0, 
bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
model7.fit(x_train, y_train)
y_pred7 = model7.predict(x_test)
accuracy7 = accuracy_score(y_test, y_pred7)
print("Bagging Accuracy: %.2f%%" % (accuracy7 * 100.0))

# ****** 8) Random Forest ********************
from sklearn.ensemble import RandomForestClassifier
model8 = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=-1)
model8.fit(x_train, y_train)
y_pred8 = model8.predict(x_test)
accuracy8 = accuracy_score(y_test, y_pred8)
print("Random Forest Accuracy: %.2f%%" % (accuracy8 * 100.0))

# ****** 9) XGBoost ********************
from xgboost import XGBClassifier
model9 = XGBClassifier()
model9.fit(x_train, y_train)
y_pred9 = model9.predict(x_test)
accuracy9 = accuracy_score(y_test, y_pred9)
print("XGBoost Accuracy: %.2f%%" % (accuracy9 * 100.0))



