import os
import pickle
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv  # <--- INI PENTING! Library untuk baca .env

# 1. PERINTAHKAN PYTHON BACA FILE .ENV
# Load environment variables dari file .env di folder project
load_dotenv() 

app = Flask(__name__)

# 2. Konfigurasi Path Model (Agar aman di Vercel & Lokal)
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'model_chatbot.pkl')
responses_path = os.path.join(base_path, 'responses_data.pkl')

model = None
responses_data = None

# 3. Load Model ML
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(responses_path, 'rb') as f:
        responses_data = pickle.load(f)
    print("✅ Model berhasil dimuat!")
except FileNotFoundError:
    print("❌ ERROR: File model tidak ditemukan.")
    print(f"Dicari di: {model_path}")
    print("Pastikan file .pkl sudah dipindahkan ke folder 'api'!")

# 4. Inisialisasi Groq
# Kode ini akan mengambil kunci dari file .env
api_key = os.environ.get("GROQ_API_KEY")
client = None

if api_key:
    try:
        client = Groq(api_key=api_key)
        print("✅ Groq Client berhasil diaktifkan!")
    except Exception as e:
        print(f"⚠️ Gagal inisialisasi Groq: {e}")
else:
    print("⚠️ PERINGATAN: GROQ_API_KEY tidak ditemukan di file .env")

@app.route('/')
def home():
    return "Tanish Cafe Chatbot API is Running!"

@app.route('/chat', methods=['POST'])
def chat():
    # Cek kesiapan model
    if not model or not responses_data:
        return jsonify({"error": "Model ML belum siap. Cek log terminal."}), 500

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # --- TAHAP 1: PREDIKSI MODEL SENDIRI ---
    try:
        predicted_intent = model.predict([user_input])[0]
        probs = model.predict_proba([user_input])
        confidence = max(probs[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Ambil respons dasar
    import random
    base_response = random.choice(responses_data[predicted_intent])
    final_response = base_response

    # --- TAHAP 2: GROQ ENHANCEMENT ---
    # Hanya pakai Groq jika Client aktif & Confidence cukup tinggi
    if client and confidence > 0.5:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"Kamu CS 'Tanish Cafe'. User: '{user_input}'. Intent: '{predicted_intent}'. Jawaban database: '{base_response}'. Tulis ulang jawaban itu biar lebih ramah & gaul (bhs Indo). Langsung jawab saja."
                    }
                ],
                model="llama3-8b-8192",
            )
            final_response = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq Error: {e}") 
            # Jika error, tetap pakai jawaban base_response
            pass

    return jsonify({
        "intent": predicted_intent,
        "confidence": float(confidence),
        "response": final_response
    })

if __name__ == '__main__':
    app.run(debug=True)