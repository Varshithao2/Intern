import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- Setup and Data Loading ---
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
os.makedirs(model_dir, exist_ok=True)

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "transactions.csv")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: transactions.csv not found at {data_path}. Please check the file path.")
    exit()

df.columns = df.columns.str.strip()

# --- Feature Engineering ---
def time_to_minutes(t):
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(':'))
            return h * 60 + m
        except ValueError:
            return 0
    return 0

df['Time'] = df['Time'].fillna('00:00').apply(time_to_minutes)

X = df.drop('Class', axis=1)
y = df['Class']

categorical_cols = [col for col in ['location', 'merchant'] if col in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Visualize class distribution before SMOTE ---
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class (0=Legit, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# --- Apply SMOTE (for visualization only) ---
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

plt.figure(figsize=(6,4))
sns.countplot(x=y_res)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class (0=Legit, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# --- Model Pipeline ---
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),   # actual training with SMOTE
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training the model with SMOTE-balanced data...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- Evaluation ---
print("\n--- Model Evaluation on Original Test Data ---")
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("True Positives:", cm[1, 1])
print("False Positives:", cm[0, 1])
print("True Negatives:", cm[0, 0])
print("False Negatives:", cm[1, 0])
# --- Visualize Confusion Matrix ---
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#log loss
ll = log_loss(y_test, y_pred_proba)
print(f"\nLog Loss: {ll:.4f}")

# --- Save Model ---
model_path = os.path.join(model_dir, "model.joblib")
try:
    joblib.dump(model_pipeline, model_path)
    print(f"\nModel saved successfully at {model_path}.")
except Exception as e:
    print(f"\nError saving model: {e}")
