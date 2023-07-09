# HSNNAI
A Neural Network, ChatBot, AI, And Trainer Written In Python.

## LSTM and Flask Incorporation

This project is a Flask-based chatbot application that uses a Long Short-Term Memory (LSTM) model for text prediction. The LSTM model is trained on a dataset that is scraped and processed from a specific website.

## Description

The script includes the following steps:

1. Scraping the data from a website, extracting the conversation elements, and storing them into a DataFrame.
2. Preprocessing the scraped data by cleaning and tokenizing the text, and preparing it for the LSTM model.
3. Training an LSTM model using the processed data, with early stopping and grid search for hyperparameter tuning.
4. Saving the trained model and the tokenizer for future use.
5. Setting up a Flask web server with a `/predict` endpoint that accepts a JSON object, validates it, uses the saved model to generate a prediction, and returns a response.

## Installation

The script uses a number of Python packages which need to be installed. You can install these packages using pip:

```
pip install numpy pandas requests beautifulsoup4 nltk scikit-learn gensim unidecode jsonschema flask tensorflow keras
```

Also, download the NLTK stopwords using the following Python command:

```python
import nltk
nltk.download('stopwords')
```

## Usage

To run the script, simply use the command:

```
python script_name.py
```

The Flask server will run on `http://localhost:5000/`.

To use the `/predict` endpoint, send a POST request with a JSON body to `http://localhost:5000/predict`. The JSON body should follow the schema:

```json
{
  "utterance": "your text here"
}
```

For example:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"utterance":"Hello!"}' http://localhost:5000/predict
```

This will return the bot's response in JSON format.

## Note

The `scrape_data` function needs to be modified to match the actual structure of the website you're scraping data from. The website's scraping policy should be considered before proceeding. The model parameters and training configurations such as `EMBEDDING_DIM`, `LSTM_UNITS`, `DROPOUT_RATE`, `EPOCHS` should be adjusted according to the specific use case.

Ensure you have the appropriate exception handling in place for production-level code. Always validate the incoming data before processing and handle exceptions appropriately to prevent the application from crashing and to provide meaningful error messages.
