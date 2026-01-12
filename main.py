import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("parkinsons.csv")

features = ["spread1", "PPE"]
target = "status"

X = df[features]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split
(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42)

model.fit(X_train, y_train)

print(accuracy_score(y_val, model.predict(X_val)))

joblib.dump(model, "parkinsons_model.joblib")



