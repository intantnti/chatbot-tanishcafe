import os
import pickle
from flask import Flask, request, jsonify, render_template  # <--- Pastikan render_template ada
from groq import Groq
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Tentukan folder template dan static secara eksplisit agar tidak bingung
# template_folder menunjuk ke folder 'templates' di lokasi file ini berada
app = Flask(__name__, template_folder='templates')

# --- CONFIG PATH ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'model_chatbot.pkl')
responses_path = os.path.join(base_path, 'responses_data.pkl')

model = None
responses_data = None

# --- LOAD MODEL ---
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(responses_path, 'rb') as f:
        responses_data = pickle.load(f)
    print("✅ Model berhasil dimuat!")
except FileNotFoundError:
    print("❌ Model tidak ditemukan. Pastikan file .pkl ada di folder 'api'")

# --- GROQ INIT ---
api_key = os.environ.get("GROQ_API_KEY")
client = None
if api_key:
    client = Groq(api_key=api_key)

# --- ROUTES ---

@app.route('/')
def home():
    # Ini yang membuat tampilan chat muncul
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not model or not responses_data:
        return jsonify({"error": "Model belum siap"}), 500

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "Pesan kosong"}), 400

    # Prediksi
    predicted_intent = model.predict([user_input])[0]
    probs = model.predict_proba([user_input])
    confidence = max(probs[0])
    
    # Ambil respons dasar
    import random
    base_response = random.choice(responses_data[predicted_intent])
    final_response = base_response

    # Groq Enhancement
    if client and confidence > 0.5:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"Kamu CS Cafe. User: '{user_input}'. Intent: '{predicted_intent}'. Jawab database: '{base_response}'. Perbaiki kalimatnya jadi ramah & gaul. Langsung jawab."
                    }
                ],
                model="llama3-8b-8192",
            )
            final_response = chat_completion.choices[0].message.content
        except:
            pass

    return jsonify({"response": final_response})

if __name__ == '__main__':
    app.run(debug=True)