ELE489 HW1 – K-Nearest Neighbors on Wine Dataset - Cenk Bural 
This repository contains the solution to Homework 1 of ELE489: Fundamentals of Machine Learning at Hacettepe University. The goal of this assignment is to implement the K-Nearest Neighbors (KNN) algorithm from scratch and compare its performance to the built-in implementation provided by Scikit-learn, using the Wine dataset from the UCI Machine Learning Repository.

Project Structure
knn.py: Custom implementation of the KNN algorithm (supporting Euclidean and Manhattan distances).
analysis.ipynb: Jupyter Notebook that includes data visualization, preprocessing, KNN experiments, performance comparison, and evaluation.
wine.data: UCI Wine dataset (must be in the project folder).
README.md: Description of the project and instructions to run the code.

Dataset Information
Dataset: UCI Wine Dataset

Source: https://archive.ics.uci.edu/dataset/109/wine

Number of Instances: 178

Features: 13 numerical features

Classes: 3 wine cultivars (labeled as 1, 2, 3)

What’s Implemented
Data loading and exploratory data analysis (EDA)
Visualization using Seaborn and Matplotlib (KDE plots, boxplots, pairplots)
Data preprocessing and normalization using StandardScaler
Splitting the dataset into training (80%) and testing (20%) sets
Custom implementation of the KNN algorithm (supports both Euclidean and Manhattan distances)
Accuracy comparison for different K values (1 to 30)
Confusion matrices and classification reports
Comparison between custom KNN and Scikit-learn’s KNeighborsClassifier
Evaluation of prediction match rates between both implementations

How to Run
Install required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn
Make sure the wine.data file is in the same directory.

Run the Jupyter notebook:

Open analysis.ipynb in Jupyter Notebook or Google Colab and run all the cells to reproduce the results and visualizations.


Accuracy is plotted for each K value using both distance metrics.

Confusion matrices and classification reports are generated for multiple settings.

Match rate between the custom and sklearn predictions is calculated and visualized.

Instructor Information
Course: ELE489 – Fundamentals of Machine Learning
Instructor: Prof. Seniha Esen Yüksel
Department of Electrical and Electronics Engineering
Hacettepe University

References
UCI Wine Dataset: https://archive.ics.uci.edu/dataset/109/wine
Scikit-learn documentation: https://scikit-learn.org/stable/
KNN tutorial on Kaggle: https://www.kaggle.com/code/prashant111/knn-classifier-tutorial
Confusion matrix explanation: https://www.w3schools.com/python/python_ml_confusion_matrix.asp

