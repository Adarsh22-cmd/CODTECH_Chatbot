import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random

# Download necessary NLTK packages (run only first time)
nltk.download('punkt')

# -------------------------------
# TRAINING DATA (INTENTS)
# -------------------------------
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning", "What's up"],
            "responses": ["Hello! How can I help you?", "Hi there!", "Hey! What can I do for you today?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you", "Take care"],
            "responses": ["Goodbye!", "See you soon!", "Take care!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful"],
            "responses": ["You're welcome!", "Glad I could help!", "Anytime!"]
        },
        {
            "tag": "about",
            "patterns": ["Who are you?", "What are you?", "Tell me about yourself"],
            "responses": ["I am an AI chatbot created using Python and NLTK!", 
                          "I am your simple NLP-based chatbot created for CODTECH Internship."]
        }
    ]
}

# -------------------------------
# PREPROCESSING THE DATA
# -------------------------------
words = []
labels = []
docs_x = []
docs_y = []

# Collect and stem words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stemming
words = [stemmer.stem(w.lower()) for w in words if w.isalpha()]
words = sorted(list(set(words)))
labels = sorted(labels)

# Creating training data (Bag-of-Words)
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc if w.isalpha()]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# -------------------------------
# FUNCTION TO CONVERT USER INPUT TO BoW
# -------------------------------
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    bag = [0 for _ in range(len(words))]
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)

# -------------------------------
# CHAT LOOP
# -------------------------------
print("Chatbot is ready! Type 'quit' to exit.\n")

while True:
    inp = input("You: ")

    if inp.lower() == "quit":
        print("Chatbot: Goodbye! 👋")
        break

    bow = bag_of_words(inp)
    result = np.dot(output.T, bow)

    index = np.argmax(result)
    tag = labels[index]

    for tg in intents["intents"]:
        if tg["tag"] == tag:
            responses = tg["responses"]

    print("Chatbot:", random.choice(responses))
