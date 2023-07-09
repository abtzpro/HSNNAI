import os
import re
import string
import logging
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
from unidecode import unidecode
from jsonschema import validate
from flask import Flask, request, jsonify
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import nltk
import json
import jsonschema
import emoji

# Constants
EMBEDDING_DIM = 50
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
EPOCHS = 100

# Define the schema for incoming JSON validation
schema = {
    "type" : "object",
    "properties" : {
        "utterance" : {"type" : "string"},
    },
}

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

# Setup logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def scrape_data(url):
    # Step 1: Data scraping
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        logging.error(f'Request failed: {e}')
        return None
    soup = BeautifulSoup(response.text, 'html.parser')

    # You would need to adjust this part to fit the actual structure of the website
    utterances = [element.text for element in soup.select('.utterance')]
    responses = [element.text for element in soup.select('.response')]

    df = pd.DataFrame({'utterance': utterances, 'response': responses})

    return df

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation, remove words containing numbers, handle emojis, and remove excess whitespaces.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords)
    text = unidecode(text) # handles accented characters
    text = ''.join(char for char in text if char not in emoji.UNICODE_EMOJI) # removes emojis
    text = re.sub(' +', ' ', text) # removes excess whitespaces
    return text

def process_data(df):
    try:
        df['utterance'] = df['utterance'].apply(lambda x: clean_text(x))
        df['response'] = df['response'].apply(lambda x: clean_text(x))
    except Exception as e:
        logging.error(f'Data processing failed: {e}')
        return None

    return df

def create_model(optimizer='adam', vocab_size=None, max_length=None):
    # Create the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_length))
    model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Scrape and process the data
url = 'https://website_to_scrape_from.com'  # replace with the actual website
df = scrape_data(url)
df = process_data(df)

# Step 2: Prepare the data and train the model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['utterance'] + df['response'])

X = tokenizer.texts_to_sequences(df['utterance'])
X = pad_sequences(X, padding='post')
y = tokenizer.texts_to_sequences(df['response'])
y = pad_sequences(y, padding='post')

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(x) for x in X) # or set your own max_length
# Reshape X to (num_samples, num_time_steps, 1) to match LSTM 3D input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Grid Search for Optimizer hyperparameter
model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam']
param_grid = dict(optimizer=optimizers)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, np.expand_dims(y_train, -1), callbacks=[es], epochs=EPOCHS, validation_data=(X_test, np.expand_dims(y_test, -1)))  # also added validation_data here
best_params = grid_result.best_params_
logging.info(f'Best parameters: {best_params}')

# Train model with best parameters
model = create_model(optimizer=best_params['optimizer'], vocab_size=vocab_size, max_length=max_length)
history = model.fit(X_train, np.expand_dims(y_train, -1), validation_data=(X_test, np.expand_dims(y_test, -1)), epochs=EPOCHS, verbose=1, callbacks=[es])  # y should be 3D for sparse_categorical_crossentropy

# Save the model and tokenizer
model.save('my_model')
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

# Step 3: Run the web server
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Validate incoming JSON
        validate(data, schema)
    except jsonschema.exceptions.ValidationError as e:
        logging.error(f'Invalid JSON received: {e}')
        return jsonify(error=str(e)), 400
    
    try:
        # Reload tokenizer and model
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            tokenizer = tokenizer_from_json(json_data)
        model = tf.keras.models.load_model('my_model')

        sequence = tokenizer.texts_to_sequences([clean_text(data['utterance'])])
        sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        sequence = np.reshape(sequence, (sequence.shape[0], sequence.shape[1], 1))  # reshaping sequence to match LSTM input
        prediction = model.predict(sequence)
        response_index = np.argmax(prediction, axis=-1)
        response = tokenizer.sequences_to_texts(response_index)
    except Exception as e:
        logging.error(f'Prediction failed: {e}')
        return jsonify(error=str(e)), 500

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
