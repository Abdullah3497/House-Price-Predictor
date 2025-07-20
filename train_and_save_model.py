# train_and_save_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import joblib

# 📥 Load and clean data
data = pd.read_csv("Cleaned_data_for_model.csv")
data = data.drop(columns='Unnamed: 0')

# 🔁 Label Encoding
label_encoders = {}
for col in ['property_type', 'city', 'purpose', 'location']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 🎯 Features & Target
X = data.drop(columns='price')
y = data['price']

# 🔀 Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Train
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
model.fit(x_train, y_train)

# 📈 Evaluate
pred = model.predict(x_test)
print(f"✅ R² Score: {r2_score(y_test, pred):.4f}")

# 💾 Save model and encoders
joblib.dump(model, "xgb_model.pkl")
for name, encoder in label_encoders.items():
    joblib.dump(encoder, f"{name}_encoder.pkl")

print("✅ Model and encoders saved!")
