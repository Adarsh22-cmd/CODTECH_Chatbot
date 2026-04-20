# AI Chatbot using Python and Sentence Transformers

This project is an AI-powered chatbot built with Python.  
It uses semantic similarity with Sentence Transformers to understand user input, match it to the correct intent, and generate more natural responses.

## Features

- Intent-based AI chatbot
- Semantic understanding using Sentence Transformers
- Typo correction using RapidFuzz
- Better response handling for common user messages
- Feedback storage for future improvement
- Teach mode for adding new training examples
- Lightweight and easy to extend

## Project Structure

## ```bash
CODTECH_Chatbot-main/
│── chatbot.py
│── intents.json
│── feedback_data.json
│── train_data.json
│── pattern_embeddings.npy
│── requirements.txt
│── README.md

## Technologies Used
Python
NumPy
Sentence Transformers
RapidFuzz
JSON

## How It Works
The chatbot works in the following way:

1. Loads intents from intents.json
2. Loads a pre-trained Sentence Transformer model
3. Converts all intent patterns into embeddings
4. Converts user input into an embedding
5. Compares user input with stored intent patterns
6. Finds the closest matching intent
7. Generates a response based on that intent
8. Stores feedback for future learning

## Installation

Open terminal inside the project folder and run:

1. pip install -r requirements.txt

If needed, install missing package manually:
pip install rapidfuzz

## Run the Chatbot
python chatbot.py

## Example Usage
You: hi
Bot: Hi! How can I help you?

You: how are you
Bot: I am doing well, thank you. How can I help you?

You: who are you
Bot: I am an AI chatbot built with Python and semantic understanding.

You: good afternoon
Bot: Good afternoon! How can I help you?

## Teach the Chatbot
You can teach the chatbot new examples using this format:

teach:tag:sentence
Example:

teach:greeting:good night
teach:thanks:many many thanks
teach:goodbye:see you next week

These examples are stored in train_data.json.


## Files Description
- chatbot.py
Main chatbot program.

- intents.json
Contains chatbot intents, training patterns, and responses.

- feedback_data.json
Stores user chat history and predicted tags.

- train_data.json
Stores user-taught examples.

- pattern_embeddings.npy
Cached embeddings for faster startup.

- requirements.txt
Project dependencies.

## Main Improvements Over Basic Chatbot
Compared to a simple keyword-based chatbot, this version is better because:

- It understands meaning, not only exact words
- It handles small spelling mistakes
- It gives more natural responses
- It supports user teaching
- It is easier to improve and scale

## Future Improvements
Possible future enhancements:

- Add a web interface using Streamlit or Flask
- Add voice input and output
- Store chat history in database
- Use a larger transformer model
- Add multilingual support
- Add automatic retraining from feedback

## Project Summary

This project demonstrates how to build a more intelligent chatbot using modern NLP techniques.
It combines semantic matching, typo correction, and structured intent handling to provide better chatbot responses than traditional rule-based systems.