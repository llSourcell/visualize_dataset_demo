'''
Comaprison results with PCA or LDA:
Linear SVM accuracy without any transformation: 79.98%
Linear SVM accuracy with PCA transformation: 80.03%
Linear SVM accuracy with LDA transformation: 71.52%
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

# ****  SVM ********************
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Linear SVM accuracy without any transformation: %.2f%%" % (accuracy_svm * 100.0))

# ****  PCA+SVM ********************
from sklearn.decomposition import PCA
pca = PCA() # keep all components
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
svm2 = SVC(kernel='linear', C=1.0, random_state=0)
svm2.fit(x_train_pca, y_train)
y_pred_pca_svm = svm2.predict(x_test_pca)
accuracy_pca_svm = accuracy_score(y_test, y_pred_pca_svm)
print("Linear SVM accuracy with PCA transformation: %.2f%%" % (accuracy_pca_svm * 100.0))

# ****  LDA+SVM ********************
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=4)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)
svm3 = SVC(kernel='linear', C=1.0, random_state=0)
svm3.fit(x_train_lda, y_train)
y_pred_lda_svm = svm3.predict(x_test_lda)
accuracy_lda_svm = accuracy_score(y_test, y_pred_lda_svm)
print("Linear SVM accuracy with LDA transformation: %.2f%%" % (accuracy_lda_svm * 100.0))
