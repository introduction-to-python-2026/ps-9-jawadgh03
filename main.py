

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
df = pd.read_csv('parkinsons.csv')
features = ['MDVP:Fo(Hz)', 'HNR']   # Inputs
target = 'status'                    # Output
X = df[features]
y = df[target]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
joblib.dump(model, 'parkinsons_model.joblib')
print("Model saved as parkinsons_model.joblib")




