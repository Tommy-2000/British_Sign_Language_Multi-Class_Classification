# -*- coding: utf-8 -*-
"""
ICA Assignment - Element 1 - Machine Learning Application
British Sign Language Phrase Recognition - Multi-Class Classification
"""

# %% 1. Import Libraries and Data

print("Import Libraries and Data")

# Import any in-built Python libraries
from collections import Counter

# Import data processing and visualisation libraries
import os
import sys
import time
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

# Import ML libraries from SciKit-Learn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Import ML models from SciKit-Learn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Import data from CSV file to pandas dataframe
bsl_data = pd.read_csv(
    './Dataset/BSL-leap-motion.csv', header=0, engine='python')

# Save a copy of the unmodified dataset
bsl_data_RAW_COPY = bsl_data.copy()

# Open a new TXT file to export print output
sys.stdout = open("ICA_Console_Output_FINAL.txt", "w")
# Comment this line out to display print output to console window

# %% 2. Exploratory Data Analysis

print("Exploratory Data Analysis")

print("\n")

# Check the length, shape and total number of elements in the imported dataset
print("Dataset Length:", "\n", len(bsl_data))
print("Dataset Shape:", "\n", bsl_data.shape)
print("Dataset Total Number of Elements:", "\n", bsl_data.size)

print("\n")

# Printing the first 10 rows of the dataset
print("First 10 Rows:", "\n", bsl_data.head(10))

print("\n")

# Printing the last 10 rows of the dataset
print("Last 10 Rows:", "\n", bsl_data.tail(10))

print("\n")

# Printing the description of the dataset
print("Description of Dataset:", "\n", bsl_data.info)

print("\n")

# Printing the data types of the dataset
print("Data Types of Dataset:", "\n", bsl_data.dtypes)

print("\n")

# Printing the statistics of the dataset
print("Statistics of Dataset:", "\n", bsl_data.describe)

print("\n")

# Printing a summary of unique values in the dataset
print("Unique Values of Dataset:", "\n", bsl_data.nunique)

print("\n")

# Number of unique values in CLASS column
print("Number of unique values in CLASS column")
print(len(bsl_data.CLASS.unique()))

print("\n")

# Count of each target in CLASS column
print("Count of each target in CLASS column")
print(bsl_data.CLASS.value_counts())

# Count of each target in CLASS column - Countplot
class_count_plt = plt.subplots(figsize=(6,6))
class_count_plt = sns.countplot(y='CLASS', data=bsl_data)
plt.show()

print("\n")

# Calculate the number of duplicate data in the dataset
dup_data = bsl_data.duplicated()
# Printing the number of duplicate data in the dataset
print("Duplicate data in Dataset:" "\n", dup_data.any())

print("\n")


# %% 3. Cleaning and Pre-Processing Data

print("Cleaning and Pre-Processing Data")

print("\n")

# Clean the dataset by checking for null values and checking the number of columns
# Check column names of the dataset
print("Check columns names of the dataset")
bsl_data_cols = list(bsl_data.columns)
print(bsl_data.columns)

print("\n")

# Check for missing values in dataset
print("Check for missing values in dataset")
print(bsl_data.isna().sum()/len(bsl_data)*100)

print("\n")

# Check for null values in dataset
print("Check for null values in dataset")
print(bsl_data.isnull().sum()/len(bsl_data)*100)

print("\n")

# Create a copy of the cleaned dataset
print("Create a copy of the cleaned dataset")
cleaned_bsl_data = bsl_data.copy()

print("\n")

# Print the distribution of the target
print("Print the distribution of the target")
print(np.array(np.unique(bsl_data.CLASS, return_counts=False)))

print("\n")

# Call the MinMaxScaler class
mms = preprocessing.MinMaxScaler()

# Find the feature names by its index position using iloc
bsl_data_features = bsl_data.iloc[:, 0:428]
bsl_data_targets = bsl_data.iloc[:, 428:429]

# Fit the MinMax estimator on the features of the dataset using the MinMaxScaler
processed_features = mms.fit_transform(bsl_data_features)

# Check the contents of the MinMax scaled features in the array
print("Check the contents of scaled features in the array")
print(processed_features)

# Merge the encoded labels with the scaled features into a DataFrame
scaled_bsl_data = pd.DataFrame(processed_features, bsl_data_targets)

# Create a copy of scaled_bsl_data before performing PCA
scaled_bsl_data_RAW_COPY = scaled_bsl_data.copy()

print("\n")

# Check the first 10 rows of the scaled DataFrame
print("The first 10 rows of the transformed DataFrame")
print(scaled_bsl_data.head(10))

print("\n")

# Check the last 10 rows of the transformed DataFrame
print("The last 10 rows of the transformed DataFrame")
print(scaled_bsl_data.tail(10))

print("\n")


# %% 4.1 Feature Engineering

print("Feature Engineering")

print("\n")

# Split the PCA transformed features along with the processed targets into training and testing data
# 80% training data and 20% testing data
features_train, features_test, target_train, target_test = train_test_split(
    processed_features, np.ravel(bsl_data_targets), test_size=0.2, random_state=100, stratify=np.ravel(bsl_data_targets))


# Check the count of target variables in the target training and testing sets
print("Count of target variables in the target training subset" + "\n")
print(Counter(target_train))
print("\n")
print("Count of target variables in the target testing subset" + "\n")
print(Counter(target_test))

print("\n")

# %% 4.2 PCA Data Analysis

# Initialise the PCA
# The number of components could change in order to achieve the best accuracy possible

pca = PCA(n_components=0.95)

# Fit the PCA on the training features for analysis
pca_training_features = pca.fit_transform(features_train)

# Transform the PCA on the testing features for analysis
pca_testing_features = pca.transform(features_test)

# Check the number of princpal components generated, this is called on the pipeline object
print("Number of principal components generated",
      "\n", pca.n_components_)

print("\n")

# Check the shape of the PCA training features
print("Shape of PCA Training Features: {}".format(pca_training_features.shape))

print("\n")

# Check the shape of the PCA testing features
print("Shape of PCA Testing Features: {}".format(pca_testing_features.shape))

print("\n")

# Create a DataFrame that stores the training features
pca_train_features_DF = pd.DataFrame(data=pca_training_features, columns=[
    'principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10', 'principal component 11', 'principal component 12', 'principal component 13', 'principal component 14', 'principal component 15', 'principal component 16', 'principal component 17', 'principal component 18'])


# Create a DataFrame that stores the testing features
pca_test_features_DF = pd.DataFrame(data=pca_testing_features, columns=[
    'principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10', 'principal component 11', 'principal component 12', 'principal component 13', 'principal component 14', 'principal component 15', 'principal component 16', 'principal component 17', 'principal component 18'])

# Show the first 5 rows of the prinicipal component training DataFrame
print("The first 5 rows of the Principal Component Training DataFrame")
print(pca_train_features_DF.head())

print("\n")

# Show the first 5 rows of the prinicipal component testing DataFrame
print("The first 5 rows of the Principal Component Testing DataFrame")
print(pca_test_features_DF.head())

print("\n")

# Show the last 5 rows of the principal component training DataFrame
print("The last 5 rows of the Principal Component Training DataFrame")
print(pca_train_features_DF.tail())

print("\n")

# Show the last 5 rows of the principal component training DataFrame
print("The last 5 rows of the Principal Component Testing DataFrame")
print(pca_test_features_DF.tail())

print("\n")

# Print the cumulative sum of the PCA Training DataFrame
print("Cumulative sum of the PCA Training DataFrame")
print(pca_train_features_DF.cumsum())

print("\n")

# Print the cumulative sum of the PCA Testing DataFrame
print("Cumulative sum of the PCA Testing DataFrame")
print(pca_test_features_DF.cumsum())

print("\n")

# Print the covariance matrix of the PCA Training DataFrame
print("Covariance Matrix of the PCA Training DataFrame: \n%s" %
      np.cov(pca_train_features_DF.T))

print("\n")

# Print the covariance matrix of the PCA Testing DataFrame
print("Covariance Matrix of the PCA Testing DataFrame: \n%s" %
      np.cov(pca_test_features_DF.T))

print("\n")

# Find the Eigenvalues for each Principal Component
print("Eigenvalues for each Principal Component: {}".format(
    pca.explained_variance_))

print("\n")

# Find the ratio of the explained variation for each principal component
print("Ratio of the explained variation for each Principal Component: {}".format(
    pca.explained_variance_ratio_))

print("\n")

# Find the cumulative proportion of explained variance
print("Cumulative proportion of explained variation: {}".format(
    np.cumsum(pca.explained_variance_ratio_)))

print("\n")


# Visualise the number of principal components required to retain maximum variance - Curve plot
print("Visualise Number of Principal Components Required - Curve plot")
pca_plot, pca_ax = plt.subplots()
x_components = np.arange(0,18, step=1)
y_pca_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(x_components, y_pca_variance, marker = 'o', linestyle='-', color='b')
plt.text(0.5, 0.85, 'Variance Cut-Off Threshhold', color='red', fontsize=10)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.xticks(np.arange(0,18, step=1))
plt.ylabel('Cumulative Explained Variance (%)')
plt.ylim(0.0, 1.2)
plt.title("The Number of Principal Components Needed to Explain Variance - Curve plot")

pca_ax.grid(axis='x')
plt.show()

print("\n")

# Visualise the Eigenvalues for each PCA Component - Barplot
print("Eigenvalues for each PCA Component - Barplot")
pca_eigenvalues_DF = pd.DataFrame(
    {'Eigenvalues': pca.explained_variance_, 'Principal Components': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']})
sns.barplot(x='Principal Components', y="Eigenvalues",
            data=pca_eigenvalues_DF, color="c")
plt.title('Eigenvalues for each Principal Component')
plt.show()

print("\n")

# %% 5. Machine Learning Modeling
# %% 5.1 Support Vector Classifier Machine Model

# Initialise the model with the necessary hyperparameters
svc = SVC(kernel='rbf', gamma='auto', C=25, random_state=42)
          
# Create the pipeline that fits both the PCA and model to the training subset of data
svc_pipe = list()
svc_pipe.append(('pca', pca))
svc_pipe.append(('svc_model', svc))
svc_pipeline = Pipeline(svc_pipe)

# Declare a variable that stores the time taken to train the model
svc_training_time = time.time()

# Fit the model and PCA within the pipeline on the training features and targets
svc_pipeline.fit(features_train, target_train)

# Check the amount of time taken to train
print("SVC Training Time: ", time.time()-svc_training_time)

# Check the accuracy of the model in the pipeline on the training data
print("SVC Coefficient of Determination - Training Data: ", svc_pipeline.score(features_train, target_train))

print("\n")

# Check the accuracy of the model in the pipeline on the testing data
print("SVC Coefficient of Determination - Testing Data: ", svc_pipeline.score(features_test, target_test))

print("\n")

# Check cross-validation score over 5 K-folds of the model on training data in the pipeline
print("SVC - Cross-Validation Score - Training Data: " + "\n", cross_val_score(svc_pipeline, features_train, target_train, cv=5))

print("\n")

# Check cross-validation score over 5 K-folds of the model on testing data in the pipeline
print("SVC - Cross-Validation Score - Testing Data: " + "\n", cross_val_score(svc_pipeline, features_test, target_test, cv=5))

print("\n")

# Predict the accuracy of the target classification based on the PCA transformed testing features
svc_target_test_pred = svc_pipeline.predict(features_test)
print("Support Vector Classifier Prediction - Testing Data: ", metrics.accuracy_score(
    target_test, svc_target_test_pred))

print("\n")

# Print the classification report for this model to check the recall and f1 score on the testing data
print("Classification Report - SVC: " + "\n", metrics.classification_report(target_test, svc_target_test_pred))

# Show the test predictions made on a confusion matrix
svc_test_cm = metrics.confusion_matrix(target_test, svc_target_test_pred)

# Set the axis labels for the Confusion Matrices
x_cm_labels = bsl_data.CLASS.unique()
y_cm_labels = bsl_data.CLASS.unique()

print("\n")

# Visualise the confusion matrix for the testing subset using matplotlib and Seaborn
print("Confusion Matrix - Support Vector Classifier Prediction - Testing Data")
svc_ax = plt.subplots(figsize=(9, 9))
svc_ax = plt.title('SVC Testing Subset - Confusion Matrix', fontsize=18)
svc_ax = sns.heatmap(svc_test_cm, xticklabels=x_cm_labels, yticklabels=y_cm_labels, square=True, annot=True,
                    fmt="d", cbar=True, linewidths=.5)
plt.show()

print("\n")

# %% 5.2 K-Nearest Neighbour Model

# Initialise the model
# The default value for n_neighbors is 5
knn = KNeighborsClassifier()

# Create the pipeline that fits both the PCA and model to the training subset of data
knn_pipe = list()
knn_pipe.append(('pca', pca))
knn_pipe.append(('knn_model', knn))
knn_pipeline = Pipeline(knn_pipe)

# Declare a variable that stores the time taken to train the model
knn_training_time = time.time()

# Fit the pipeline on the training features and targets
knn_pipeline.fit(features_train, target_train)

# Check the amount of time taken to train
print("K-NN Training Time: ", time.time()-knn_training_time)

print("\n")

# Check the accuracy of the model in the pipeline on the training data
print("K-NN Coefficient of Determination - Training Data: ", knn_pipeline.score(features_train, target_train))

print("\n")

# Check the accuracy of the model in the pipeline on the testing data
print("K-NN Coefficient of Determination - Testing Data: ", knn_pipeline.score(features_test, target_test))

print("\n")

# Check cross-validation score over 5 K-folds of the model on training data in the pipeline
print("K-NN - Cross-Validation Score - Training Data: " + "\n", cross_val_score(knn_pipeline, features_train, target_train, cv=5))

print("\n")

# Check cross-validation score over 5 K-folds of the model on testing data in the pipeline
print("K-NN - Cross-Validation Score - Testing Data: " + "\n", cross_val_score(knn_pipeline, features_test, target_test, cv=5))

print("\n")

# Predict the accuracy of the target classification based on the PCA transformed testing features
knn_target_test_pred = knn_pipeline.predict(features_test)
print("K-Nearest Neighbours Prediction - Testing Data: ",
      metrics.accuracy_score(target_test, knn_target_test_pred))

print("\n")

# Print the classification report for this model to check the recall and f1 score on the testing data
print("Classification Report - K-Nearest Neighbours: " + "\n", metrics.classification_report(target_test, knn_target_test_pred))

# Show the testing predictions made on a confusion matrix
knn_test_cm = metrics.confusion_matrix(target_test, knn_target_test_pred)

print("\n")

# Visualise the confusion matrix using matplotlib and Seaborn
print("Confusion Matrix - K-Nearest Neighbours Prediction - Testing Data")
knn_ax = plt.subplots(figsize=(9, 9))
knn_ax = plt.title('KNN Testing Subset - Confusion Matrix' , fontsize=18)
knn_ax = sns.heatmap(knn_test_cm, xticklabels=x_cm_labels, yticklabels=y_cm_labels, square=True, annot=True,
                      fmt="d", cbar=True, linewidths=.5)
plt.show()

print("\n")

# %% 5.3 Random Forests Model

# Initialise the model
# The default parameters will be used
rfc = RandomForestClassifier()

# Create the pipeline that fits both the PCA and model to the training subset of data
rfc_pipe = list()
rfc_pipe.append(('pca', pca))
rfc_pipe.append(('rfc_model', rfc))
rfc_pipeline = Pipeline(rfc_pipe)

# Declare a variable that stores the time taken to train the model
rfc_training_time = time.time()

# Fit the model on the training features and targets
rfc_pipeline.fit(features_train, target_train)

# Check the amount of time taken to train
print("Random Forest Training Time: ", time.time()-rfc_training_time)

print("\n")


# Check the accuracy of the model in the pipeline on the training data
print("Random Forest Coefficient of Determination - Training Data: ", rfc_pipeline.score(features_train, target_train))

print("\n")

# Check the accuracy of the model in the pipeline on the testing data
print("Random Forest Coefficient of Determination - Testing Data: ", rfc_pipeline.score(features_test, target_test))

print("\n")

# Check cross-validation score over 5 K-folds of the model on training data in the pipeline
print("Random Forest Classifier - Cross-Validation Score - Training Data: " + "\n", cross_val_score(rfc_pipeline, features_train, target_train, cv=5))

print("\n")

# Check cross-validation score over 5 K-folds of the model on testing data in the pipeline
print("Random Forest Classifier - Cross-Validation Score - Testing Data: " + "\n", cross_val_score(rfc_pipeline, features_test, target_test, cv=5))

print("\n")

# Predict the accuracy of the target classification based on the PCA transformed testing features
rfc_target_test_pred = rfc_pipeline.predict(features_test)
print("Random Forest Classifier Prediction - Testing Data: ",
      metrics.accuracy_score(target_test, rfc_target_test_pred))

print("\n")

# Print the classification report for this model to check the recall and f1 score on the testing data
print("Classification Report - Random Forest Classifier: " + "\n", metrics.classification_report(target_test, rfc_target_test_pred))

# Show the testing predictions made on a confusion matrix
rfc_test_cm = metrics.confusion_matrix(target_test, rfc_target_test_pred)

print("\n")

# Visualise the confusion matrix using matplotlib and Seaborn
print("Confusion Matrix - Random Forest Prediction - Testing Data")
rfc_ax = plt.subplots(figsize=(9, 9))
rfc_ax = plt.title('Random Forest Testing Subset - Confusion Matrix', fontsize=18)
rfc_ax = sns.heatmap(rfc_test_cm, xticklabels=x_cm_labels, yticklabels=y_cm_labels, square=True, annot=True,
                      fmt="d", cbar=True, linewidths=.5)
plt.show()

print("\n")


# %% 6. Model Accuracy Results

# Visualise the accuracy for each learning
print("Overall accuracy for each learning")

# Create the DataFrame containing the results from each model
accuracy_scores_DF = pd.DataFrame([['Support Vector Classifier', metrics.accuracy_score(target_test, svc_target_test_pred)],
                                    ['K-Nearest Neighbours',
                                        metrics.accuracy_score(target_test, knn_target_test_pred)],
                                    ['Random Forest Classifier', metrics.accuracy_score(target_test, rfc_target_test_pred)]],
                                  columns=['Models', 'Accuracy'])

# Reshape the DataFrame using pandas.melt
accuracy_scores_DF = pd.melt(
    frame=accuracy_scores_DF, id_vars='Models', var_name='Accuracy', value_name='Level of Accuracy')

# Print the resulting DataFrame
print("Overall Accuracy Scores: ", "\n", accuracy_scores_DF)

print("\n")

# Plot the DataFrame as a Barplot
print("Model Classification Accuracy Scores")
accuracy_scores_plot = plt.subplots(figsize=(8, 8))
accuracy_scores_plot = plt.title("Model Classification Accuracy Scores", fontsize=18)
accuracy_scores_plot = sns.barplot(
    data=accuracy_scores_DF, x='Models', y='Level of Accuracy')
plt.show()


# %% 7. Export DataFrames

# Raw input data
bsl_data.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\RAW_data.csv')

# Dataset features
bsl_data_features.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\FEATURES_data.csv')

# Dataset targets
bsl_data_targets.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\TARGETS_data.csv')

# Scaled dataset features
scaled_bsl_data.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\SCALED_FEATURES_data.csv')

# PCA Components training features
pca_train_features_DF.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\PCA_TRAIN_FEATURES_data.csv')

# PCA Components testing features
pca_test_features_DF.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\PCA_TEST_FEATURES_data.csv')

# PCA Eigenvalues dataset
pca_eigenvalues_DF.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\PCA_EIGENVALUES_data.csv')

# Accuracy Scores dataset
accuracy_scores_DF.to_csv(r'C:\Users\laptop\Documents\TU - Modules\Machine Learning\ICA Assignment\B1662096_Thomas_Popham\ICA Element 1 - Application\DataFrames\ACCURACY_SCORES_data.csv')

# Print message once exporting is complete
print("Exporting Complete")


