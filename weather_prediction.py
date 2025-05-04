import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("Weather_Data.csv")

# Preprocess
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill missing values
for col in numeric_columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Label encode categorical features
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Features and label
label = 'RainToday'
features = data.drop([label, 'Date'], axis=1)
target = data[label]

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
features_resampled, target_resampled = smote.fit_resample(features, target)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_resampled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target_resampled, test_size=0.25, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# --- Prediction Section ---
# Feature names expected by the model
feature_names = features.columns.tolist()

def get_user_input():
    print("\nEnter values for the following features:")
    user_values = []
    for feat in feature_names:
        while True:
            try:
                val = float(input(f"{feat}: "))
                user_values.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    return np.array(user_values).reshape(1, -1)

def predict(user_input):
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]
    return prediction, probability

if __name__ == "__main__":
    print("\n--- Rain Prediction System ---")
    user_input = get_user_input()
    pred, prob = predict(user_input)
    print("\n🌦️ Prediction:")
    print(f"Will it rain today? {'Yes' if pred == 1 else 'No'}")
    print(f"Probability of rain: {prob:.2f}")
