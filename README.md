This is the streamlit application demo: https://fraud-detection-app-dx2suzndewdgfzgtpfqrwc.streamlit.app/

🕵️‍♂️ Fraud Detection using Machine Learning
This project aims to detect fraudulent transactions using various machine learning classification algorithms. The goal is to compare multiple models and select the best-performing one based on accuracy, using a dataset from Kaggle.
🕵️‍♂️ Fraud Detection using Machine Learning
This project aims to detect fraudulent transactions using various machine learning classification algorithms. The goal is to compare multiple models and select the best-performing one based on accuracy, using a dataset from Kaggle.

📁 Dataset
File Used: Fraud.csv

Source: Kaggle Dataset under /kaggle/input/fraud-data/

Target Column: isFraud (Binary classification - 1 indicates fraud, 0 indicates non-fraud)

🧪 Project Workflow
1. Data Loading & Initial Analysis
Loaded the CSV using pandas

Performed .info(), .describe(), and .isnull().sum() checks

Verified the data shape and data types

2. Data Preprocessing
Categorical variable type was encoded using pd.get_dummies with drop_first=True to avoid multicollinearity.

Separated features and labels:

python
Copy
Edit
X = df.drop("isFraud", axis=1)
y = df["isFraud"]
Standardized features using StandardScaler from sklearn.preprocessing

3. Train-Test Split
First, a 5% sample of the data was used to quickly evaluate model performances.

Final training used an 80-20 split:

python
Copy
Edit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
4. Model Selection (5% Sample)
10 classification models were evaluated:

Logistic Regression

Random Forest

Gradient Boosting

Decision Tree

Support Vector Machine

K-Nearest Neighbors

Naive Bayes

AdaBoost

XGBoost

LightGBM

Top 3 Models (based on accuracy):

XGBoost - 99.96%

Random Forest - 99.96%

AdaBoost - 99.94%

5. Final Model Training
Chose XGBoost for final training on 80% of the data.

Used XGBClassifier with the following parameters:

python
Copy
Edit
xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
6. Model Saving
The final trained XGBoost model was saved using:

python
Copy
Edit
import joblib
joblib.dump(best_model_xgb, "fraud_detection_xgboost_model.pkl")
📈 Performance
Accuracy achieved with final XGBoost model was extremely high (~99.96%)

Dataset was highly imbalanced (very few frauds compared to total transactions), so accuracy is not the only metric to be considered in production.

📦 Libraries Used
pandas

numpy

seaborn, matplotlib (for visualizations)

sklearn

xgboost

lightgbm

joblib

✅ Future Work
Handle class imbalance using techniques like SMOTE, oversampling, or cost-sensitive learning

Incorporate AUC-ROC, F1-score, precision, and recall for better evaluation

Explore deep learning models


📁 Dataset
File Used: Fraud.csv

Source: Kaggle Dataset under /kaggle/input/fraud-data/

Target Column: isFraud (Binary classification - 1 indicates fraud, 0 indicates non-fraud)

🧪 Project Workflow
1. Data Loading & Initial Analysis
Loaded the CSV using pandas

Performed .info(), .describe(), and .isnull().sum() checks

Verified the data shape and data types

2. Data Preprocessing
Categorical variable type was encoded using pd.get_dummies with drop_first=True to avoid multicollinearity.

Separated features and labels:

python
Copy
Edit
X = df.drop("isFraud", axis=1)
y = df["isFraud"]
Standardized features using StandardScaler from sklearn.preprocessing

3. Train-Test Split
First, a 5% sample of the data was used to quickly evaluate model performances.

Final training used an 80-20 split:

python
Copy
Edit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
4. Model Selection (5% Sample)
10 classification models were evaluated:

Logistic Regression

Random Forest

Gradient Boosting

Decision Tree

Support Vector Machine

K-Nearest Neighbors

Naive Bayes

AdaBoost

XGBoost

LightGBM

Top 3 Models (based on accuracy):

XGBoost - 99.96%

Random Forest - 99.96%

AdaBoost - 99.94%

5. Final Model Training
Chose XGBoost for final training on 80% of the data.

Used XGBClassifier with the following parameters:

python
Copy
Edit
xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
6. Model Saving
The final trained XGBoost model was saved using:

python
Copy
Edit
import joblib
joblib.dump(best_model_xgb, "fraud_detection_xgboost_model.pkl")
📈 Performance
Accuracy achieved with final XGBoost model was extremely high (~99.96%)

Dataset was highly imbalanced (very few frauds compared to total transactions), so accuracy is not the only metric to be considered in production.

📦 Libraries Used
pandas

numpy

seaborn, matplotlib (for visualizations)

sklearn

xgboost

lightgbm

joblib

✅ Future Work
Handle class imbalance using techniques like SMOTE, oversampling, or cost-sensitive learning

Incorporate AUC-ROC, F1-score, precision, and recall for better evaluation

Explore deep learning models




