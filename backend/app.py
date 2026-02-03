import joblib
import os
from pydantic import BaseModel, Field
from datetime import datetime

# membuat model vektorizer
model = joblib.load('./Training Data/knn_model.pkl')
tfidf_vectorizer = joblib.load('./Training Data/vectorizer.pkl')

# prediksi untuk teks baru
new_text = ["contoh teks baru untuk diklasifikasikan"]
new_text_tfidf = tfidf_vectorizer.transform(new_text)
prediction = model.predict(new_text_tfidf)

# output prediksi
if prediction[0] == 1:
    print("Teks ini terdeteksi sebagai penipuan.")
else:
    print("Teks ini tidak terdeteksi sebagai penipuan.")
print("Prediksi:", prediction[0])

class TextInput(BaseModel):
    text: str = Field(..., example="Masukkan teks di sini")
    
class PredictionResponse(BaseModel):
    TextInput:
        hint_text:

def predict_single_text(text: str) -> dict:
    load_models()
