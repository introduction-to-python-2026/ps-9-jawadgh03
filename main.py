

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1️⃣ قراءة البيانات
df = pd.read_csv('parkinsons.csv')

# 2️⃣ اختيار الميزات والمخرجات
features = ['MDVP:Fo(Hz)', 'HNR']   # Inputs
target = 'status'                    # Output
X = df[features]
y = df[target]

# 3️⃣ تطبيع البيانات
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 4️⃣ تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5️⃣ اختيار الموديل وتدريبه
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6️⃣ تقييم الموديل
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# 7️⃣ حفظ الموديل
joblib.dump(model, 'parkinsons_model.joblib')
print("Model saved as parkinsons_model.joblib")




