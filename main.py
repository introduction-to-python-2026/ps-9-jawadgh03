import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
df = pd.read_csv('parkinsons.csv')
print(df)
df = pd.read_csv('parkinsons.csv')
X = df[['MDVP:Fo(Hz)', 'HNR']]
y = df['status']
print(X.head())
print(y.head())
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('parkinsons.csv')

X = df[['MDVP:Fo(Hz)', 'HNR']]
y = df['status']

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(X_scaled.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('parkinsons.csv')
X = df[['MDVP:Fo(Hz)', 'HNR']]
y = df['status']

# تطبيع
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# التأكد من الأحجام
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
df = pd.read_csv('parkinsons.csv')
X = df[['MDVP:Fo(Hz)', 'HNR']]
y = df['status']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
df = pd.read_csv('parkinsons.csv')
X = df[['MDVP:Fo(Hz)', 'HNR']]
y = df['status']

# تطبيع
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# تقسيم
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# تدريب الموديل
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# توقع
y_pred = model.predict(X_test)

# حساب الدقة
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)



