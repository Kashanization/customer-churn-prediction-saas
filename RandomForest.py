# Random Forest model for customer churn prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("C:/Users/pegahg/Desktop/ChurnProject/churn_dataset_raw_realistic.csv")

# Drop unnecessary columns
df_model = df.copy()
df_model.drop(['CustomerID', 'SignupDate', 'LastLoginDate'], axis=1, inplace=True)

# Encode categorical variables
categorical_cols = ['SubscriptionType', 'Industry']
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

# Split features and target
X = df_model.drop('Churned', axis=1)
y = df_model['Churned']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))