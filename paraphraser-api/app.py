from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

# Using 't5-small' because it fits in Render's 512MB RAM limit
model_name = 't5-small' 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_paraphrase(text):
    # T5 expects a specific prefix
    input_text = "paraphrase: " + text 
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs, 
        max_length=128, 
        num_beams=5, 
        is_encoder_decoder=True,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/api/paraphrase', methods=['POST'])
def paraphrase_endpoint():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        result = get_paraphrase(data['text'])
        return jsonify({"original": data['text'], "paraphrased": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
