from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Flask uygulamasını başlat
app = Flask(__name__)

# Eğitilmiş modeli ve scaler'ı yükle
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    print("Model ve scaler başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: Model veya scaler dosyaları bulunamadı. Lütfen önce train_model.py dosyasını çalıştırın.")
    model = None
    scaler = None
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    model = None
    scaler = None


@app.route('/')
def home():
    """
    Ana sayfayı (index.html) gösterir.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Formdan gelen verileri alır, tahminde bulunur ve sonucu döndürür."""
    if model is None or scaler is None:
        return render_template('index.html', result='HATA: Model yüklenemedi. Lütfen sunucu kayıtlarını kontrol edin.')
    try:
        # Formdan gelen tüm değerleri 'float' tipine çevir
        features = [float(x) for x in request.form.values()]

        # Özellikleri 2D numpy array'ine çevir (modelin beklediği format)
        final_features = np.array(features).reshape(1, -1)

        # Gelen veriyi, modeli eğitirken kullandığımız scaler ile ölçeklendir
        scaled_features = scaler.transform(final_features)

        # Tahminde bulun
        prediction = model.predict(scaled_features)
        # Tahmin sonucunu (0 veya 1) metne çevir
        if prediction[0] == 1:
            result_text = "Diyabet Riskiniz Bulunmaktadır."
        else:
            result_text = "Diyabet Riskiniz Bulunmamaktadır."

        # Sonucu aynı HTML sayfasına gönder
        return render_template('index.html', result=result_text)

    except Exception as e:
        # Hata yönetimi
        return render_template('index.html', result=f'Tahmin sırasında bir hata oluştu: {e}')
if __name__ == '__main__':
    # Uygulamayı çalıştır
    app.run(debug=True)