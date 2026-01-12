import pandas as pd 
df = pd.read_csv("/content/parkinsons.csv") 
print(list(df.columns)) 
selcted_feature = ['MDVP:Flo(Hz)','NHR'] 
selcted_2 = ['status'] 
from sklearn.preprocessing import MinMaxScaler

X = df[selcted_feature] 
y = df[selcted_2]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train) 

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') 

selected_features: ["A", "B"]  
path: "my_model.joblib"
