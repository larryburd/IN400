########################################################################
# Author: Laurence T. Burden
# Date: 20231007
# For: Purdue University Global
#      IN400 - AI: Deep Learning and Machine Learning
# Title: Unit 7 Assignment Part 1/ Module 5 Competency Assessment Part 1
#        Breast Cancer Random Forest Classification Using Python
########################################################################

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn import datasets
from matplotlib import rcParams
from datetime import datetime

# Set plot parameters
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1

# Output header
print('\nUnit 7 Assignment Part 1 / Module 5 Competency Assessment Part 1 Output\n')
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), '\n')

# Load dataset from sklearn
bcDataset = datasets.load_breast_cancer()

# Convert dataset to pandas dataframe
bcDataframe = pd.DataFrame(bcDataset.data, columns=bcDataset.feature_names)

# Append dataframe containing tumor features with diagnostic outcomes
bcDataframe['Diagnosis'] = bcDataset.target

# Exploratory data analysis
print("DATA SNIPPET")
print(bcDataframe.head())
print()

# Look at tumor features in relation to diagnosis
print("MEAN VALUES BY DIAGNOSIS")
print(bcDataframe.groupby('Diagnosis').mean())
print()

# Dataframe for differential diagnosis visual comparisons
# Malignant = 0 | Benign = 1
bcDataframe_m = bcDataframe[bcDataframe['Diagnosis'] == 0]
bcDataframe_b = bcDataframe[bcDataframe['Diagnosis'] == 1]

print("MALIGNANT DIAGNOSIS DATA")
print(bcDataframe_m)
print()
print("BENIGN DIAGNOSIS DATA")
print(bcDataframe_b)
print()

# Create list of features related to mean tumor characteristics
features_means = list(bcDataframe.columns[0:10])

# Dataframe for presenting diagnosis
outcome_count = bcDataframe.Diagnosis.value_counts()
outcome_count = pd.Series(outcome_count)
outcome_count = pd.DataFrame(outcome_count)
outcome_count.index = ['Benign', 'Malignant']

# Change column name to 'Diagnosis', because the index function renamed the column to 'count'
outcome_count = outcome_count.rename(columns={'count': 'Diagnosis'})

# Add percent column
outcome_count['Percent'] = 100 * outcome_count['Diagnosis']/sum(outcome_count['Diagnosis'])

print('Percentage of tumors classified as \'malignant\' in this data set is: {}'
      .format(100*float(bcDataframe.Diagnosis.value_counts()[0]/float(len(bcDataframe)))))
print()
print('A good classifier should therefore outperform blind guessing knowing the proportions i.e. > 62% accuracy')
print(outcome_count)
print()

# Visualize frequency of diagnoses in dataset
sns.set(rc={"figure.figsize": (8, 5)})
sns.barplot(x = ['Benign', 'Malignant'], y='Diagnosis', data=outcome_count, alpha=.8)
plt.title('Frequency of Diagnostic Outcomes in Dataset')
plt.ylabel('Frequency')
plt.savefig('FrequencyPlot.png')

# Visualize relationships between features and diagnosis
# Dataset already has the mean values of all the columns
heatMapParamList = features_means + list(bcDataframe.columns[-1:])
sns.set(rc={"figure.figsize": (12, 6)})
sns.set(font_scale=0.6)
sns.heatmap(bcDataframe[heatMapParamList].corr())
sns.set_style("whitegrid")
plt.title('Feature Means/Diagnosis Heat Map')
plt.savefig('FeatureAndMeansHeatMap.png')

# Split data into training and test sets and normalize data
x_train, x_test, y_train, y_test = train_test_split(bcDataframe.iloc[:, :-1], bcDataframe['Diagnosis'], train_size=.8)
norm = Normalizer()

# Fit training set with the normalization
norm.fit(x_train)

# Transform both training and testing sets
x_train_norm = norm.transform(x_train)
x_test_norm = norm.transform(x_test)

# Perform model testing
# Define parameters to optimize random forest using dictionaries
RF_params = {'n_estimators': [10, 50, 100]}

# Set the scoring parameter to measure model accuracy
scoring = 'accuracy'

# Using the k-fold cross-validation technique, break the training data into 5 folds
kfold = KFold(n_splits=5, random_state=2, shuffle=True)

# Instantiate gridsearch
model_grid = GridSearchCV(RandomForestClassifier(), RF_params)

# Use cross validation method with gridsearch model
cv_results = cross_val_score(model_grid, x_train_norm, y_train, cv=kfold, scoring=scoring)

# Define a string object to list model name, cv accuracy, and cv standard deviation
msg = "Cross Validation Accuracy %s: Accuracy: %f SD: %f" % ('RFC', cv_results.mean(), cv_results.std())

#Print message object
print(msg)
print()

# Apply random forest classifier on test data
# Set parameters
RF_params = {'n_estimators': [10, 50, 100, 200]}

# Instantiate random forest
RFC_2 = RandomForestClassifier(random_state=42)

# Instantiate gridsearch using random forest model and dictated parameters
RFC_2_grid = GridSearchCV(RFC_2, RF_params)

# Fit model to training data
RFC_2_grid.fit(x_train_norm, y_train)

# Print best parameters
print('Optimized number of estimators: {}'.format(RFC_2_grid.best_params_.values()))
print()

# Train RFC on whole training set
RFC_3 = RandomForestClassifier(n_estimators=50, random_state=42)

# Fit random forest to training data
RFC_3.fit(x_train_norm, y_train)

# Predict on training data using fitted random forest
RFC_3_predicted = RFC_3.predict(x_test_norm)

# Evaluate RFC with test data
print('Model accuracy on test data: {}'.format(accuracy_score(y_test, RFC_3_predicted)))
print()

# Check random forest result metrics
# Confusion matrix
confusion_matrix_RF = pd.DataFrame(confusion_matrix(y_test, RFC_3_predicted),
                                   index=['Actual Malignant', 'Actual Benign'],
                                   columns=['Predicted Malignant', 'Predicted Benign'])

print('Random Forest Model Confusion Matrix\n')
print(confusion_matrix_RF)
print()
print('Random Forest Model Classification Report\n')
print(classification_report(y_test, RFC_3_predicted))
