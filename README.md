# 🚀 Diyabet Hastalığı Tespiti: Lojistik Regresyon ile Erken Teşhis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/m-yusufuzun/Diyabet-Veri-Seti-ve-Lojistik-Regresyon-ile-Diyabet-Hastaligi-Teshisi?style=for-the-badge&color=gold)](https://github.com/m-yusufuzun/Diyabet-Veri-Seti-ve-Lojistik-Regresyon-ile-Diyabet-Hastaligi-Teshisi/stargazers)

Bu proje, hastaların sağlık verilerini (glikoz seviyesi, BMI, yaş vb.) kullanarak **Lojistik Regresyon** modeli ile diyabet hastalığı riskini tahmin etmeyi amaçlayan bir makine öğrenimi uygulamasıdır. 🧪


❗Sonuçlar gerçeği yansıtmayabilir. Diyabet hastalığından şüpheleniyorsanız bir sağlık kuruluşuna başvurun!

---

## ✨ Projenin Amacı

Sağlık alanında erken teşhisin önemi göz önüne alındığında, bu çalışma;
* Basit ama etkili bir sınıflandırma algoritması olan Lojistik Regresyon'u kullanarak diyabet riskini öngörmeyi.
* Hasta verilerine dayalı, yorumlanabilir bir karar destek modeli geliştirmeyi hedeflemektedir.

---

## 📚 Kullanılan Veri Seti

Çalışmamızda, makine öğrenimi topluluğunda sıkça karşılaşılan `diabetes.csv` dosyası (yaygın adıyla **Pima Kızılderilileri Diyabet Veri Seti**) kullanılmıştır.

* **Toplam Gözlem:** [Örn: 768]
* **Öznitelik Sayısı:** 8 (Sayısal)
* **Hedef Değişken:** 1 (İkili sınıflandırma: 0 veya 1)

### Veri Seti Öznitelikleri:

| Öznitelik Adı             | Açıklama                                       |
| :------------------------ | :--------------------------------------------- |
| `Pregnancies`             | Hamilelik sayısı                               |
| `Glucose`                 | Plazma glikoz konsantrasyonu                   |
| `BloodPressure`           | Diyastolik kan basıncı (mm Hg)                 |
| `SkinThickness`           | Triceps deri kıvrım kalınlığı (mm)             |
| `Insulin`                 | 2 saatlik serum insülini (mu U/ml)             |
| `BMI`                     | Vücut Kitle İndeksi (kg/m²)                    |
| `DiabetesPedigreeFunction` | Diyabet soyağacı fonksiyonu (genetik eğilim)   |
| `Age`                     | Yaş (yıl)                                      |
| `Outcome`                 | **Hedef:** Diyabet (1) veya Değil (0)          |

---

## 🛠️ Teknoloji Yığını

Projemiz, güçlü Python ekosistemi üzerinde inşa edilmiştir:

* **Dil:** <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white" />
* **Veri Analizi:** <img alt="Pandas" src="https://img.shields.io/badge/pandas-1.x-red?style=flat&logo=pandas&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/numpy-1.x-blueviolet?style=flat&logo=numpy&logoColor=white" />
* **Makine Öğrenimi:** <img alt="Scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat&logo=scikit-learn&logoColor=white" />
* **Serileştirme:** <img alt="Joblib" src="https://img.shields.io/badge/joblib-1.x-yellowgreen?style=flat&logo=python&logoColor=white" /> (Model ve Scaler kaydetmek için)
* **Ortam:** <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-red?style=flat&logo=jupyter&logoColor=white" /> (Geliştirme ve dokümantasyon için)

---

## ⚙️ Proje Akışı (`train_model.py`)

Projenin temel adımları, `train_model.py` betiğinde aşağıdaki gibi gerçekleştirilmiştir:

1.  **Veri Yükleme:** `diabetes.csv` dosyası `pandas` ile okunur.
2.  **Veri Hazırlama:** 'Outcome' sütunu hedef (`y`), geri kalanlar öznitelikler (`X`) olarak ayrılır.
3.  **Veri Bölme:** Veri seti, **%80 eğitim** ve **%20 test** oranında `train_test_split` ile ayrılır (`random_state=42`).
4.  **Veri Ölçeklendirme:**
    * `StandardScaler` kullanılarak öznitelikler ölçeklenir.
    * Scaler, *yalnızca eğitim verisi* (`X_train`) üzerinde eğitilir ve her iki set (`X_train`, `X_test`) bu scaler ile dönüştürülür.
5.  **Model Eğitimi:**
    * `LogisticRegression(random_state=42)` modeli tanımlanır.
    * Ölçeklendirilmiş eğitim verisi üzerinde model eğitilir.
6.  **Model Değerlendirme:**
    * Modelin doğruluğu (`accuracy_score`) test seti üzerinde hesaplanır.
7.  **Model ve Scaler Kaydı:**
    * Eğitilmiş model (`diabetes_model.pkl`) ve `StandardScaler` objesi (`diabetes_scaler.pkl`) `joblib` ile kaydedilir. Bu, modelin ve ölçekleyicinin gelecekteki tahminlerde kullanılmasını sağlar.

---

## 📊 Sonuçlar ve Değerlendirme

Modelin test verisi üzerindeki performansı şu şekildedir:

### ✅ Doğruluk (Accuracy) Skoru: **`75.32%`**

---

## 📸 Projeden Ekran Görüntüleri

`train_model.py` betiğinin temel kod akışı ve terminal çıktısı aşağıda sunulmuştur.

### ✅ - Diyabet Hastalığı Bulunmayan Durum

![Code Flow Part 1](https://github.com/user-attachments/assets/ce11a170-f907-4859-b11c-f55215a50fe4) 

### ❎ - Diyabet Hastalığı Bulunan Durum

![Code Flow Part 2 & Terminal Output](https://github.com/user-attachments/assets/f6129f51-0a8f-4fd7-bcb1-3a09af606068)

---

## 🚀 Yerel Olarak Çalıştırma

Bu projeyi kendi bilgisayarınızda kurmak ve çalıştırmak için aşağıdaki adımları izleyin:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/m-yusufuzun/Diyabet-Veri-Seti-ve-Lojistik-Regresyon-ile-Diyabet-Hastaligi-Teshisi.git
    cd Diyabet-Veri-Seti-ve-Lojistik-Regresyon-ile-Diyabet-Hastaligi-Teshisi.git
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    *(Eğer `requirements.txt` dosyanız varsa, `pip install -r requirements.txt` kullanabilirsiniz.)*
    ```bash
    pip install pandas scikit-learn joblib
    ```

3.  **Modeli Eğitmek ve Kaydetmek İçin Çalıştırın:**
    ```bash
    python train_model.py
    ```
    Bu komut, modelinizi eğitecek, doğruluğunu ekrana yazdıracak ve `diabetes_model.pkl` ile `diabetes_scaler.pkl` dosyalarını proje dizinine kaydedecektir.

---

## 🤝 Katkıda Bulunma

Projeyi daha da geliştirmek için her türlü katkı ve geri bildirim değerlidir! Eğer bir hata bulursanız veya yeni bir özellik eklemek isterseniz, lütfen bir `issue` açın veya bir `pull request` gönderin.

---

## 📜 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakın.

---
