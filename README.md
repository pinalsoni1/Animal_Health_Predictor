# Animal Health Predictive Modeling Project

## Overview

This project is developed for NSDC Data Science Projects, focuses on wildlife conservation by building a predictive model to determine if an animal's health condition is dangerous and if it is at risk of dying. The model utilizes five distinct symptoms from a diverse dataset of animal species. The primary goal is to classify animal risk levels, thereby aiding in decision-making for animal welfare and contributing to bio-heritage conservation.

## Dataset

The project uses the "Animal Disease" dataset sourced from Kaggle:
[https://www.kaggle.com/datasets/gracehephzibahm/animal-disease](https://www.kaggle.com/datasets/gracehephzibahm/animal-disease)

The dataset includes information on various animal species and reported symptoms.

## Project Workflow

The project is structured into several key milestones:

1.  **Milestone 1: Importing Libraries and Dataset**
    * Importing necessary Python libraries (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, etc.).
    * Setting up the Kaggle API to download and unzip the dataset.

2.  **Milestone 2: Exploratory Data Analysis (EDA)**
    * Loading and inspecting the dataset (shape, column names, data types).
    * Viewing top and bottom rows to understand data structure.
    * Getting a summary of the data using `.info()`.

3.  **Milestone 3: Data Cleaning and Preprocessing**
    * Handling missing values (dropping rows with nulls in the target 'Dangerous' column).
    * Standardizing 'AnimalName':
        * Converting to lowercase.
        * Consolidating similar animal names (e.g., "goats" to "goat", "moos" to "cow").
    * Standardizing Symptom Columns ('symptoms1' through 'symptoms5'):
        * Converting text to lowercase.
        * Removing unwanted spaces and special characters using a custom cleaning function.
        * Correcting spelling errors and standardizing similar symptom descriptions using `difflib.SequenceMatcher` to identify and replace terms (e.g., 'difficultty in breathing' to 'breathing difficulty').
    * Renaming columns for clarity (e.g., 'AnimalName' to 'Animal', 'symptoms1' to 'Symptom 1').

4.  **Milestone 4: Addressing Imbalanced Dataset**
    * Visualizing the class distribution of the 'Dangerous' target variable.
    * Applying `RandomOverSampler` from the `imbalanced-learn` library to balance the dataset.
    * Using `LabelEncoder` to convert categorical symptom data and the target variable into numerical format for model training.

5.  **Milestone 5: Model Training - Random Forest Classifier**
    * Splitting the preprocessed data into training and testing sets.
    * Training a Random Forest Classifier on the training data.
    * Performing predictions on single samples and the test dataset.
    * Evaluating the model's accuracy.

6.  **Milestone 6: Comparing Other Machine Learning Models**
    * Training and evaluating several other classification models:
        * Logistic Regression
        * Gradient Boosting Classifier
        * XGBoost Classifier
        * Support Vector Machine (SVC)
        * K-Nearest Neighbors (KNN) Classifier
        * Decision Tree Classifier
        * LightGBM Classifier
        * AdaBoost Classifier
    * For each model, performance is assessed using confusion matrices, classification reports (precision, recall, F1-score), and overall accuracy.

7.  **Milestone 7: Model Evaluation**
    * A comprehensive evaluation of all trained models.
    * Calculating and comparing accuracy, precision, recall, F1-score, and ROC AUC scores.
    * Plotting ROC curves for visual comparison of model performance.

## Libraries Used

* `pandas`
* `numpy`
* `seaborn`
* `matplotlib`
* `string`
* `difflib`
* `textblob` (imported but usage not prominent in the final steps)
* `scikit-learn`:
    * `model_selection` (train_test_split)
    * `ensemble` (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
    * `linear_model` (LogisticRegression)
    * `svm` (SVC)
    * `neighbors` (KNeighborsClassifier)
    * `tree` (DecisionTreeClassifier)
    * `preprocessing` (LabelEncoder)
    * `metrics` (confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
* `imbalanced-learn` (RandomOverSampler)
* `xgboost`
* `lightgbm`
* `catboost`

## How to Run

This project is designed as a Google Colab notebook (`.ipynb` file).
1.  Open the notebook in Google Colab.
2.  Ensure you have a `kaggle.json` API token to download the dataset from Kaggle (instructions for uploading and configuring are in the notebook).
3.  Run the cells sequentially to execute the data loading, preprocessing, model training, and evaluation steps.

## Objective

The primary objective is to build a robust classifier that can accurately identify animals in critical condition based on reported symptoms, thereby supporting animal welfare and bio-heritage conservation initiatives. The project also demonstrates a complete machine learning pipeline from data acquisition and cleaning to model comparison and evaluation.
