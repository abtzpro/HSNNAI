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
python HSNNAI.py
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

## Script Breakdown

This script is divided into multiple sections, performing distinct steps in the process of building and deploying the chatbot. Here's a detailed breakdown of each section:

1. **Imports and Constants:** The script starts by importing the necessary Python libraries and defining a few constant values used in the model such as embedding dimensions, LSTM units, dropout rate, and epochs.

2. **JSON Schema Definition:** A JSON schema is defined for validating incoming requests to the Flask endpoint.

3. **Logging Setup:** Logging is set up to record events that happen while the script is running. This can be particularly useful for debugging or understanding the script's operations.

4. **Data Scraping Function:** `scrape_data(url)` is a function to scrape conversation data from a given URL. The data includes pairs of utterances and responses. Note that this function needs to be tailored based on the actual structure of the website you want to scrape data from.

5. **Text Cleaning Function:** `clean_text(text)` is a function to clean and preprocess the text data. It performs several tasks such as making text lowercase, removing text in square brackets, removing punctuation, handling emojis, and removing excess whitespaces.

6. **Data Processing Function:** `process_data(df)` is a function that applies the `clean_text(text)` function to the utterances and responses in the DataFrame.

7. **Model Creation Function:** `create_model(optimizer, vocab_size, max_length)` is a function to create the LSTM model with given optimizer, vocabulary size, and maximum sequence length.

8. **Data Scraping and Processing:** The script scrapes the data from a given URL and processes it using the defined functions.

9. **Data Preparation and Model Training:** The script tokenizes the text data, splits it into training and test sets, and fits the data to the model. It uses GridSearchCV for hyperparameter tuning and EarlyStopping for stopping the training when validation loss doesn't improve.

10. **Model Saving:** The trained model and tokenizer are saved to disk for future use.

11. **Flask Application:** A Flask application is created with a single endpoint `/predict` that accepts POST requests. The endpoint reloads the model and tokenizer, preprocesses the incoming data, predicts the response using the model, and returns it.

The script ends by running the Flask application. If the script is run directly, the Flask application will start listening for incoming requests on port 5000.

## So... Like, What is it actually??

 *Long Short-Term Memory (LSTM) Neural Networks*

Long Short-Term Memory networks, commonly known as LSTMs, are a type of Recurrent Neural Network (RNN) designed to learn from sequences of data where the temporal dynamics are relevant. They are especially known for their ability to overcome the vanishing gradient problem, a common issue in traditional RNNs, which makes them particularly suited for longer sequences of data.

In the context of our chatbot project, we use LSTMs for processing sequences of text. Each input sequence consists of a series of words represented by their indices in a dictionary (known as a 'vocabulary') created from the dataset. Each word in a sequence is connected to a time step, and the order of these steps matters because the meaning of a sentence can change dramatically based on the order of its words.

An LSTM network maintains a form of memory by using a series of 'gates'. These gates control the flow of information into and out of each LSTM cell, allowing it to keep or discard data based on its relevance to the task at hand.

For our chatbot, the LSTM network is trained to generate a response given a sequence of words (utterance). The trained model predicts the next word in a sequence, and can generate an entire sentence by iteratively feeding the predicted word back into the model as part of the input sequence.

We use a Bidirectional LSTM (BiLSTM) which involves duplicating the LSTM layer in the network so that there are now two layers side-by-side, then providing the input sequence as-is as input to the first layer, and providing a reversed copy of the input sequence to the second. This can provide additional context to the network and result in faster and even fuller learning on the problem.

The chatbot is trained by optimizing a loss function with respect to the model parameters (i.e., the weights of the connections in the network). The goal is to find the set of parameters that results in the model generating responses that are as close as possible to the actual human responses in the dataset.

After training, the LSTM model can generate a response to any given user input, making it the core component of our chatbot's ability to carry out meaningful conversations.

## How HSNNAI Learns

Our chatbot/NN learns and retains information using a Bidirectional LSTM (Long Short-Term Memory) model, a type of artificial recurrent neural network architecture.

First, we feed our model with pairs of utterances and responses scraped from a website. The model learns to map these input-output sequences during the training process. 

This learning process happens through several iterations (or epochs). In each epoch, the model makes predictions and updates its internal parameters based on the error it made. The aim is to minimize this error over time.

The LSTM's unique memory cell structure allows it to learn and remember long-term dependencies in the data. It can capture the context from both past (backwards) and future (forwards) time steps, which is crucial in understanding the context in a conversation.

Once the model is trained, it has learned to generate responses that are contextually appropriate to the input utterances. This trained model is then saved and used to make predictions when conversing with users.

## Developer Credits

Adam Rivers > 
https://abtzpro.github.io

Hello Security LLC > https://hellosecurityllc.github.io
