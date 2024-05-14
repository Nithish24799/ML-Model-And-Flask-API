from flask import Flask, request, jsonify
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import text_classification
from torchtext.experimental.datasets.text_classification import URLS
from torchtext.experimental.functional import vocab_func
import os

app = Flask(__name__)
model_path = "model.pth"
if not os.path.exists(model_path):
    torch.hub.download_url_to_file(URLS['AG_NEWS'], model_path)
model = torch.jit.load(model_path)
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(text_classification._csv_iterator(model_path))
text_pipeline = lambda x: vocab_func(tokenizer(x), vocab)

def predict(text):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text), dtype=torch.long)
        output = model(text.unsqueeze(0))
        return output.argmax(1).item() - 1 
      
@app.route('/predict', methods=['POST'])
def sentiment_analysis():
    data = request.json
    text = data['text']
    prediction = predict(text)
    return jsonify({'sentiment': 'positive' if prediction == 1 else 'negative'})

if __name__ == '__main__':
    app.run(debug=True)
