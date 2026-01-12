import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("parkinsons.csv")

features = ["MDVP:Fo(Hz)", "MDVP:RAP"]
target = "status"

X = df[features]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split

(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier
(n_estimators=200,
    max_depth=5,
    random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print(accuracy_score(y_val, y_pred))

joblib.dump(model, "parkinsons_model.joblib")


