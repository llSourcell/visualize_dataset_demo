
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# visulaize the important characteristics of the dataset
import matplotlib.pyplot as plt
# seaborn is a library for drawing statistics plots based on matplotlib
import seaborn as sns

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
print columns

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

# ****** 8) Random Forest ********************
from sklearn.ensemble import RandomForestClassifier
model8 = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=-1)
model8.fit(x_train, y_train)
y_pred8 = model8.predict(x_test)
accuracy8 = accuracy_score(y_test, y_pred8)
print("Random Forest Accuracy: %.2f%%" % (accuracy8 * 100.0))

# ************** Data Visualization ******************

# 1) plot the pair-wise relationship between two features
sns.set(style='whitegrid', context='notebook')
#the first 4 columns 
cols=columns[:4]
# scatterplot the pair-wise relationship
sns.pairplot(dataframe_all[cols],size=2.5)
plt.show()

# 2) plot the heatmap to show the correlation coefficiencies in pair-wise-features
#the last column is the class label 
cols2 = columns[:-1]
# calculate the correlation coefficiencies
correlation_matrix = np.corrcoef(dataframe_all[cols2].values.T)
#sns.set(font_scale = 1.5)
# heatmap
plt.figure()
heatmap_correlations = sns.heatmap(correlation_matrix, cbar=True, annot=True, fmt='.2f', annot_kws={'size':10}, 
 yticklabels=[x for x in range(len(cols2))],xticklabels=[x for x in range(len(cols2))])
plt.show()

# 3) fetaure importance obtained from random forest
importances = model8.feature_importances_
plt.figure()
plt.title('Feature Importances')
plt.bar(range(importances.shape[0]),importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xlim([0,importances.shape[0]])
plt.show()

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_test)

# scatter plot the sample points among 5 classes
markers=('s', 'd', 'o', '^', 'v')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()





