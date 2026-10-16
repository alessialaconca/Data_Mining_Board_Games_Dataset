# Board Games Data Mining Project

Project developed for the Data Mining course at the University of Pisa (A.Y. 2025/2026).

## Project Overview
This repository contains the code and documentation for the exploratory and predictive analysis of the Board Games Dataset. The dataset gathers information on over 20,000 board games rated by an online community, including features such as publication year, game complexity, number of players, category rankings, and game themes/mechanics.

The goal of this project is to apply core Data Mining methodologies to extract meaningful insights from the data, discover hidden patterns, group similar games, and build predictive models.

## Analysis Structure (Tasks)

The project is divided into four main modules:

### 1. Data Understanding & Preparation

* Data Semantics & Exploration: Analysis of variable meanings, study of statistical distributions (univariate and bivariate), and correlation assessment via correlation matrices.
* Data Quality & Cleaning: Identification and handling of missing values, outliers, errors, and semantic inconsistencies.
* Feature Engineering & Transformation: Application of variable transformations and feature selection for subsequent modules.

### 2. Clustering
Dataset segmentation using different families of clustering algorithms to identify homogeneous groups of board games:

* Centroid-based methods
* Density-based methods
* Hierarchical Clustering: Analysis using different distance functions and dendrogram inspection.
* Final Evaluation: Qualitative and quantitative comparison to determine the best-performing algorithm for the dataset.

### 3. Classification & Regression

* Classification: Prediction of the mandatory target variable Rating (and potential additional targets) using Decision Trees, K-Nearest Neighbors (KNN), and Naive Bayes. Includes hyperparameter tuning and performance evaluation via Confusion Matrix, Accuracy, Precision, Recall, F1-score, and ROC curves.
* Regression:
* Simple Linear Regression: Analyzing the relationship between a target variable and a single predictor.
* Multiple Regression: Predicting the target variable using 2 or more independent variables through linear and at least two non-linear approaches, evaluated via MSE and R^2.

### 4. Pattern Mining

* Frequent Pattern Extraction: Mining frequent itemsets across varying minimum support thresholds (min_sup).
* Association Rules: Generating and analyzing association rules qualitatively and quantitatively across different confidence thresholds (min_conf), examining confidence and lift distributions.
* Rule Exploitation: Leveraging the most meaningful extracted rules for missing value imputation or target variable prediction.

## Technologies Used

* Language: Python
* Main Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
* Report: LaTeX (Overleaf)
