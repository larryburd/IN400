########################################################################
# Author: Laurence T. Burden
# Date: 20231018
# For: Purdue University Global
#      IN400 - AI: Deep Learning and Machine Learning
# Title: Unit 9 Assignment Module 6 Competency Assessment Part 1
#        Cereal Rating Predictor Using Python
########################################################################

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import statsmodels.api as sm
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from datetime import datetime

# Setting some parameters and options
plt.rcParams['figure.figsize'] = [10, 6]
pd.options.mode.chained_assignment = None # default='warn'

# Data file location
dataFile = "/home/larryburd/Documents/Purdue/IN400/Unit 9/data.csv"

# Output Header
print('\nUnit 9 Assignment / Module 6 Competency Assessment Part 1 Output\n')
print(datetime.now().strftime("%m/$d/%Y $H:$M:$S"), '\n')

# Load dataset and convert csv to dataframe
cereal_df = pd.read_csv(dataFile)

# Data exploration
# Show header plus few rows of data
print()
print("DATA SNIPPET")
print(cereal_df.head())
print()

# Check number of rows before removing missing data
print("Shape of Dataframe before removing missing data (Rows, Columns): " + str(cereal_df.shape))
print()

# Remove rows with missing data
cereal_df = cereal_df.dropna()

# Check number of rows after removing missing data
print("Shape of Dataframe after removing missing data (Rows, Columns): " + str(cereal_df.shape))
print()

# Calculate Correlation between Nutritional Rating and Sugars
corr = cereal_df[['Rating', 'Sugars']].corr()
print("Correlation between Nutritional Rating and Sugars")
print(corr)
print()

# Scatterplot of Rating vs Sugars to check correlation (negative strong = -0.76)
plt.scatter(cereal_df['Rating'], cereal_df['Sugars'])

# Add title to the plot
plt.title("Correlation between Nutritional Ratings and Sugars")

# Label X and Y Axes
plt.xlabel("Nutritional Rating")
plt.ylabel("Sugars")

# Visualize the plot
plt.show()
plt.savefig('RatingVsSugars.png')

# Calculate Correlation between nutritional rating and fiber
corr = cereal_df[['Rating', 'Fiber']].corr()
print("Correlation between Nutritional Rating and Fiber")
print(corr)
print()

# Scatterplot of Ratting vs Fiber to check correlation (positive weak = 0.6)
plt.scatter(cereal_df['Rating'], cereal_df['Fiber'])

# Plot title
plt.title("Correlation between Nutritional Rating and Fiber")

# Label Axes
plt.xlabel("Nutritional Rating")
plt.ylabel("Fiber")

# Visualize the plot
plt.show()
plt.savefig('RatingVsFiber.png')

# Create a list with the names of the features we will use
feature_names = ['Sugars', 'Fiber']
print("Features: " + str(feature_names))
print()

# Create features and targets
features = cereal_df[feature_names]
targets = cereal_df['Rating']

# Create dataframe from target column and feature columns
feature_and_target_cols = ['Rating'] + feature_names
feature_target_df = cereal_df[feature_and_target_cols]

# Check dimensions of targets
print("Targets Shape: " + str(targets.shape))
print()

# Check dimensions of features
print("Features Shape: " + str(features.shape))
print()

# Calculate correlation matrix
corr_mat = feature_target_df.corr()
print("Correlation Matrix:")
print(corr_mat)
print()

# Plot heatmap of correlation matrix
sns.heatmap(corr_mat, annot=True)
plt.yticks(rotation=0); plt.xticks(rotation=90)
plt.tight_layout()

# Add title to the plot
plt.title("Correlation Matrix Heatmap")

# Visualize the plot
plt.show()
plt.savefig('CorrelationMatrixHeatMap.png')

# Build the Neural Network Model
# Create a training set that is 80% of the total samples
train_size = int(0.8 * features.shape[0])

# Train set
train_features = features[:train_size]
train_targets = targets[:train_size]

# Test set
test_features = features[train_size:]
test_targets = targets[train_size:]

# Print features shapes
print("FEATURE SHAPES (TOTAL/TRAIN/TEST)")
print(features.shape, train_features.shape, test_features.shape)
print()

# Scale data for better performance - sets mean to 0 and standard deviation to 1
# Transforms train data
scaled_train_features = scale(train_features)

# Transforms test data
scaled_test_features = scale(test_features)

# Build and fit a simple Neural Network Model
model_1 = Sequential()
model_1.add(Dense(300, input_dim=scaled_test_features.shape[1], activation='relu'))
model_1.add(Dense(150, activation='relu'))
model_1.add(Dense(1, activation='linear'))

# Show NN Summary
print()
print("MODEL SUMMARY")
print(model_1.summary())
print()

# Fit the model
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(scaled_train_features, train_targets, epochs=80)

# Plot the losses from the fit
plt.plot(history.history['loss'])

# Plot Model Loss Curve
plt.title("Model Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.savefig('ModelLossCurve.png')

# Measure performance
# Calculate R^2 score
train_preds = model_1.predict(scaled_train_features)
test_preds = model_1.predict(scaled_test_features)
print()
print("R^2 value for Training Set: ", r2_score(train_targets, train_preds))
print()
print("R^2 value for Test Set: ", r2_score(test_targets, test_preds))
print()

# Plot Predictions vs Actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.title("Predicted vs. Actual Nutritional Rating Plot")
plt.xlabel("Predicted Nutritional Rating")
plt.ylabel("Actual Nutritional Rating")
plt.legend()
plt.show()
plt.savefig('PredictedVsActualRating.png')
