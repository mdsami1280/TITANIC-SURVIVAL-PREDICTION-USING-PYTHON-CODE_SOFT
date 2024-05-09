# TITANIC SURVIVAL PREDICTION USING PYTHON/CODE_SOFT

🚢 Titanic Survival Prediction Project 🛳️

The Titanic Survival Prediction project sets sail to construct a predictive model determining passengers' survival based on various features. It harnesses the power of two machine learning juggernauts: Random Forest and Gradient Boosting, joined in harmony through a Voting Classifier ensemble for enhanced accuracy. ⚡

Dependencies 📦

Make sure you have these Python libraries aboard:

Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

You can install them using pip:

Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Dataset 📊

The dataset, residing in 'Titanic-Dataset.csv', encapsulates Titanic passengers' information, featuring age, gender, class, and more.

Data Preprocessing 🛠️

Ahoy! Let's navigate through the data:

Missing Values: Filled with sagacious decisions:
'Age' filled with the median age.
'Embarked' with the mode (most frequent value).
'Fare' with the median fare.
Feature Engineering 🛠️🔬

We're crafting new insights:

FamilySize: Merging 'SibSp' (siblings/spouses) and 'Parch' (parents/children).
Title: Extracting titles from 'Name' and simplifying them.
Model Training 🤖

We're training the models with gusto:

Random Forest and Gradient Boosting: Hypertuning their parameters with GridSearchCV.
Model Evaluation 📊

Evaluating the ensemble model's prowess:

Cross-validation scores 🔄
Accuracy, Classification Report, Confusion Matrix 📈
Data Visualization 📊

Sailing through the visual seas:

Correlation Matrix Heatmap 🔥
Age Distribution by Survival 📉
Survival Rate by Family Size 🏠
Fare Distribution by Class and Survival 💰
Feel free to explore the code and set sail on your own data adventure! 🌊✨
