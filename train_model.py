import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

print("Loading datasets...")
app = pd.read_csv('application_record.csv')
credit = pd.read_csv('credit_record.csv')

print("Merging datasets...")
df = app.merge(credit, on='ID', how='inner')

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum())

print("Cleaning data...")
df = df.dropna(subset=['AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'STATUS'])

# Credit history proxy (percentage of good months - 0 = good, C = paid off, X = no debt)
df['good_months_ratio'] = df['STATUS'].apply(lambda x: 1 if x in ['0', 'C', 'X'] else 0)
df['credit_history_proxy'] = df.groupby('ID')['good_months_ratio'].transform('mean')

# Top 4 features
top_features = [
    'AMT_INCOME_TOTAL',       # Income
    'DAYS_EMPLOYED',          # Employment stability
    'CNT_FAM_MEMBERS',        # Family size
    'credit_history_proxy'    # Credit history proxy (replaces weaker feature)
]
X = df[top_features]
y = df['STATUS'].apply(lambda x: 1 if x == '0' else 0)  # 1 = good/approved, 0 = bad

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Saving model...")
joblib.dump(model, 'credit_card_model.pkl')
print("Model saved as credit_card_model.pkl")