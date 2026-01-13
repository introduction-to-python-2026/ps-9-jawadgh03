



# =========================
# main.py - Parkinson's Detection
# =========================

# 1️⃣ استيراد المكتبات
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # لحفظ الموديل

df = pd.read_csv('parkinsons.csv')
print("Columns in dataset:")
print(df.columns)

# Inputs - الميزات المهمة
X = df[['MDVP:Fo(Hz)', 'HNR']]
# Output - الهدف
y = df['status']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("First 5 rows of scaled data:")
print(X_scaled.head())

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("Train/Test sizes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

cols = ['MDVP:Fo(Hz)', 'HNR', 'status']
sns.pairplot(df[cols], hue='status')
plt.show()

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

joblib.dump(model, 'parkinsons_model.joblib')
print("Model saved as 'parkinsons_model.joblib'")



