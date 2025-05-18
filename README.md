PROJECT DESCRIPTION:

FreakFIT-  AI-Based Fitness Package Recomendation System
This project is about making a smart system that helps people choose the best fitness plan on FreakFiit, a website that offers Zumba and workout classes.

When a new user visits the site, they fill out a short survey about their fitness interests, habits, and goals. The system then looks at their answers and uses machine learning (a type of AI) to suggest the best plan for them—Monthly, Quarterly, or Yearly.

PROJECT GOAL:

1-Develop a machine learning model that accurately predicts the best-fit fitness package for each user.
2-Create a user-friendly interface that allows users to input their information and receive personalized recommendations.
3-Automate the recommendation process using user input collected via a survey.
4-Personalize the user journey to improve satisfaction and package uptake rates.

IMPORT THE NECESSARY LIBRARIES LIKE:
pandas
numpy
sklearn
SHAP ## SHAP values help explain the contribution of each feature to a specific prediction
joblib## joblib is commonly used to export trained models for later use


DATASET DESCRIPTION:

A synthetic dataset of 20,000 entries was generated using NumPy and pandas. Each data point represents a user who filled out a fitness interest survey. The dataset includes the following features:

Age: Randomly selected between 18 and 60

Gender: Male, Female, or Other

Fitness Goal: Includes options like Weight Loss, Muscle Gain, Flexibility, etc.

Workout Experience: Beginner, Intermediate, or Advanced

Hours/Week: Weekly workout time (1 to 14 hours)

Workout Type: Zumba, Yoga, HIIT, or Mix

Timing: Preferred workout time (Morning, Evening, or Flexible)

Budget: User's spending capacity (Low, Medium, High)

Recommended Plan: Target label assigned based on simple business rules related to user experience, time availability, and budget. Users with less experience and time are more likely to get a Monthly plan, while those with more commitment and higher budgets get Quarterly or Yearly recommendations.

##This dataset is used to train the machine learning model for personalized fitness package recommendations.

DATA PREPROCESSING:

To prepare the dataset for machine learning, a preprocessing pipeline was created using scikit-learn. Different transformations were applied based on the type of feature:
1. ColumnTransformer for feature engineering
2. StandardScaler for numeric features
3. OrdinalEncoder for ordinal categorical features
4. OneHotEncoder for nominal categorical features


1-NUMERIC FEATURES:
* Features: Age, Hours/Week

Processing:

 * Missing values filled with median

 * Scaled using StandardScaler for normalization


2-ORDINAL CATEGORICAL FEATURES:
 * Features: Workout Experience, Budget

Processing:

  * Missing values filled with a constant ("Missing")

  * Ordinal encoded based on logical order (e.g., Beginner < Intermediate < Advanced)


3-NOMINAL CATEGORICAL FEATURES:

* Features: Gender, Fitness Goal, Workout Type, Timing

Processing:

  * Missing values filled with a constant ("Missing")

  * One-hot encoded to convert categories into binary columns

COLUMN TRANSFORMER:

All pipelines were combined using ColumnTransformer to apply the appropriate transformations to each feature type efficiently.

MODEL PIPELINE & TUNING:
A machine learning pipeline was built using Random Forest as the classifier. The pipeline includes:

1- Preprocessing: Cleans and transforms the data using the previously defined preprocessor.

2- Feature Selection: Uses SelectKBest to choose the top features based on statistical tests.

3- Classification: Applies a RandomForestClassifier for predicting the recommended fitness package.

HYPERPARAMETER TUNING:
To improve model performance, a randomized search was performed over key hyperparameters:

* Number of features selected (k)

* Number of trees (n_estimators)

* Minimum samples for split and leaf (min_samples_split, min_samples_leaf)


MODEL TRAINING AND EVALUATION:
The model was trained on the training data and evaluated on the test data using accuracy, precision, recall, f1.

We used RandomizedSearchCV with cross-validation to find the best model parameters. After training the model on 80% of the data, we tested it on the remaining 20%. The best model was selected based on accuracy and evaluated using a classification report.



SHAP:

To understand the feature importance and interactions, we used SHAP (SHapley Additive exPlanin ations) values. SHAP assigns a value to each feature for a specific prediction, indicating its contribution to the outcome. This helps in understanding which features are most influential in the model's decision-making process .


MODEL SAVING:

We used joblib.dump() to save the trained machine learning pipeline as a file named best_model.pkl. This allows us to reuse the model later without retraining. By loading this file, we can make predictions on new data quickly and easily.



PREDICTION FUNCTION:

The predict_plan() function takes user details as input and uses the trained model to recommend the best fitness plan—Monthly, Quarterly, or Yearly—based on those details.


FASTAPI:

This code builds a simple web app using FastAPI to let users input their fitness details through a form. It loads the trained model (best_model.pkl) to predict the best fitness plan based on the user’s input.

Users fill out the form on the homepage, and when submitted, the app processes the data, matches it to the model’s expected format, and returns the recommended plan (Monthly, Quarterly, or Yearly) as a response.



MODEL LOADING:

This code is used to load the saved machine learning model (best_model.pkl) before using it for predictions. It prints messages to confirm whether the model was loaded successfully or if there was an error. This helps in debugging and ensures the model is ready to use.


WEB FORM(HTML):

This HTML form collects user fitness details and sends them to the backend to get a recommended fitness package. It provides an easy way for users to receive personalized plan suggestions through the website.
