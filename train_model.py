import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Veri Setini Yükle
print("Veri seti yükleniyor...")
data = pd.read_csv('diabetes.csv')

# 2. Veriyi Hazırla (X ve y)
# 'Outcome' sütunu hedefimiz (y), diğerleri özellik (X)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 3. Eğitim ve Test Setlerine Ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Veriyi Ölçeklendir (Scaling)
# Lojistik Regresyon, özelliklerin ölçeklendirilmesinden fayda sağlar.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modeli Eğit (Lojistik Regresyon)
print("Model eğitiliyor...")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Modelin Başarısını Değerlendir
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {accuracy * 100:.2f}%")

# 7. Modeli ve Scaler'ı Kaydet
# Web uygulamasında yeni verileri ölçeklendirmek için scaler'a da ihtiyacımız var!
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')

print("Model ve Scaler başarıyla 'diabetes_model.pkl' ve 'diabetes_scaler.pkl' olarak kaydedildi.")