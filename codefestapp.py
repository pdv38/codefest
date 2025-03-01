from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertForTokenClassification
import torch
import sqlite3
import os
import requests


app = Flask(__name__)

# Initialize models and tokenizers
sentiment_model_name = './Sentiment-Analysis-for-Mental-Health-w-Deep-Learning/saved_model'
ner_model_path = 'dslim/distilbert-NER'

sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_name)

ner_model = DistilBertForTokenClassification.from_pretrained(ner_model_path)
ner_tokenizer = DistilBertTokenizerFast.from_pretrained(ner_model_path)

# Database setup
DATABASE = 'triggers.db'
NEWSAPI_KEY = 'your_newsapi_key'

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    if not os.path.exists(DATABASE):
        with app.app_context():
            db = get_db()
            db.execute('''
                CREATE TABLE triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    count INTEGER DEFAULT 1
                )
            ''')
            db.commit()

# Initialize the database
init_db()

def save_trigger(entity, entity_type):
    db = get_db()
    cursor = db.cursor()
    # Check if the trigger already exists
    cursor.execute('SELECT * FROM triggers WHERE entity = ? AND entity_type = ?', (entity, entity_type))
    trigger = cursor.fetchone()
    if trigger:
        # Update the count if it exists
        cursor.execute('UPDATE triggers SET count = count + 1 WHERE id = ?', (trigger['id'],))
    else:
        # Insert a new trigger if it doesn't exist
        cursor.execute('INSERT INTO triggers (entity, entity_type) VALUES (?, ?)', (entity, entity_type))
    db.commit()

def get_triggers():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT entity_type, SUM(count) as total FROM triggers GROUP BY entity_type')
    triggers = cursor.fetchall()
    return triggers

# Sentiment and NER functions (unchanged)
def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    class_labels = ['Normal', 'Depression', 'Suicidal', 'Anxiety', 'Stress', 'Bipolar', 'Personality Disorder']
    predicted_label = class_labels[prediction]
    return predicted_label

def predict_entities(text):
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = ner_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    entities = []
    for token, prediction in zip(tokens, predictions):
        entity = ner_model.config.id2label[prediction]
        if entity != 'O':
            entities.append((token, entity))
    return entities
def fetch_articles(sentiment):
    # Define a query based on the sentiment
    query = f"Tips of Dealing with emotions of {sentiment}"
    
    # Make a request to NewsAPI
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        # Return the top 2 articles
        return articles[:2]
    else:
        return []

# Routes
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    entities = predict_entities(text)
    
    # Save detected triggers to the database
    for entity, entity_type in entities:
        save_trigger(entity, entity_type)
    
    return jsonify({'sentiment': sentiment, 'entities': entities})

@app.route('/triggers')
def triggers():
    triggers = get_triggers()
    return jsonify([dict(row) for row in triggers])
@app.route('/recommend-articles', methods=['GET'])
def recommend_articles():
    # Get the most common sentiment
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT sentiment FROM sentiments ORDER BY count DESC LIMIT 1')
    most_common_sentiment = cursor.fetchone()
    
    if most_common_sentiment:
        # Fetch articles related to the most common sentiment
        articles = fetch_articles(most_common_sentiment['sentiment'])
        return jsonify(articles)
    else:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)