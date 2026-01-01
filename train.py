import json
import pickle
import nltk
from nltk.stem import PorterStemmer # Atau Sastrawi untuk Indo (opsional, pake Porter biar simple di Vercel)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # Menggunakan Support Vector Machine
from sklearn.pipeline import make_pipeline

# Download resource NLTK (perlu untuk tokenisasi)
nltk.download('punkt')

# 1. Load Dataset
with open('intents.json', 'r') as f:
    data = json.load(f)

training_sentences = []
training_labels = []
labels = []
responses = {}

# 2. Preprocessing & Data Preparation
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# 3. Membuat Pipeline Machine Learning (Feature Extraction + Model)
# TfidfVectorizer = Mengubah teks jadi angka
# SVC = Algoritma Supervised Learning untuk klasifikasi
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))

# 4. Training Model
print("Sedang melatih model...")
model.fit(training_sentences, training_labels)
print("Training selesai!")

# 5. Simpan Model dan Data Pendukung
with open('model_chatbot.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('responses_data.pkl', 'wb') as f:
    pickle.dump(responses, f)

print("Model berhasil disimpan.")