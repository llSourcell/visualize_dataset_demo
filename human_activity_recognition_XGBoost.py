import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle as pk


from matplotlib import pyplot as plt

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

# step 6: define the classifier model
model = XGBClassifier()
model.fit(x_train, y_train)

# step 7: evaluation on test set
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# convert the confusion matrix into a data frame
matrix = confusion_matrix(y_test, y_pred)
# plot the confusion matrix
df_confusion_matrix = pd.DataFrame(matrix, index = [i for i in "ABCDE"], columns = [i for i in "ABCDE"])
plt.figure(figsize = (10,7))
heatmap = sn.heatmap(df_confusion_matrix, annot=True)
fig1 = heatmap.get_figure()
fig1.savefig("output_confusion_matrix.jpg")
# plot the tree
fig2 = plt.figure()
# plot the 5-th tree
plot_tree(model, num_trees=4)
plt.show()
# plot feature importance
fig3 = plt.figure()
plot_importance(model)
plt.show()

# step 8: save the model
pk.dump(model, open("output_model.dat", "wb"))
# load the model
# loaded_model = pk.load(open("C:\\Users\\yifeng\\output_model.dat", "rb"))



