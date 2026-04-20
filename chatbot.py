import json
import random
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import process

INTENTS_FILE = "intents.json"
FEEDBACK_FILE = "feedback_data.json"
TRAIN_DATA_FILE = "train_data.json"
EMBED_FILE = "pattern_embeddings.npy"

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents = json.load(f)

print("Loading AI model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

patterns = []
tags = []
responses_map = {}

for intent in intents["intents"]:
    tag = intent["tag"]
    responses_map[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(tag)

if os.path.exists(TRAIN_DATA_FILE):
    with open(TRAIN_DATA_FILE, "r", encoding="utf-8") as f:
        try:
            learned_data = json.load(f)
            for item in learned_data:
                if "tag" in item and "sentence" in item:
                    patterns.append(item["sentence"])
                    tags.append(item["tag"])
        except json.JSONDecodeError:
            pass

vocab = set()
for p in patterns:
    for w in p.lower().split():
        vocab.add(w)

print("Encoding training patterns...")
pattern_embeddings = model.encode(
    patterns,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=False
).astype("float32")

def save_feedback(user_text, predicted_tag, correct_tag=None):
    entry = {
        "user_text": user_text,
        "predicted_tag": predicted_tag,
        "correct_tag": correct_tag
    }

    data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    data.append(entry)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def teach_new_example(tag, sentence):
    data = []

    if os.path.exists(TRAIN_DATA_FILE):
        with open(TRAIN_DATA_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    data.append({
        "tag": tag,
        "sentence": sentence
    })

    with open(TRAIN_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def correct_typos(user_input):
    words = user_input.split()
    corrected_words = []

    for word in words:
        match = process.extractOne(word.lower(), vocab, score_cutoff=80)
        if match:
            corrected_words.append(match[0])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)

def predict_intent(user_input, threshold=0.70):
    corrected_input = correct_typos(user_input)

    user_embedding = model.encode(
        corrected_input,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    scores = pattern_embeddings @ user_embedding
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    predicted_tag = tags[best_idx]
    matched_pattern = patterns[best_idx]

    if best_score < threshold:
        return None, None, None

    return predicted_tag, matched_pattern, corrected_input

def generate_better_response(predicted_tag, matched_pattern, corrected_input):
    text = corrected_input.lower().strip()

    # more natural direct replies
    if predicted_tag == "greeting":
        if "good morning" in text:
            return "Good morning! How can I help you?"
        if "good afternoon" in text:
            return "Good afternoon! How can I help you?"
        if "good evening" in text:
            return "Good evening! How can I help you?"
        if "hello" in text:
            return "Hello! How can I help you?"
        if "hi" in text or "hey" in text:
            return "Hi! How can I help you?"

    if predicted_tag == "how_are_you":
        return "I am doing well, thank you. How can I help you?"

    if predicted_tag == "thanks":
        if "many thanks" in text:
            return "You are very welcome."
        return "You're welcome!"

    if predicted_tag == "goodbye":
        if "tomorrow" in text:
            return "See you tomorrow!"
        return "Goodbye! Take care."

    if predicted_tag == "about":
        if "who are you" in text:
            return "I am an AI chatbot built with Python and semantic understanding."
        if "what are you doing" in text or "what do you do" in text:
            return "I am here to answer your messages and assist you."
        return "I am an AI chatbot designed to understand intents and respond helpfully."

    if predicted_tag == "help":
        return "Sure. Please tell me what you need help with."

    return random.choice(responses_map.get(predicted_tag, ["I understood, but I do not have a good response yet."]))

def chatbot():
    print("\nEfficient AI Chatbot is ready!")
    print("Type 'quit' to exit.")
    print("Type: teach:tag:sentence")
    print("Example: teach:greeting:good afternoon\n")

    fallback_responses = [
        "Sorry, I did not understand that clearly.",
        "Can you please rephrase that?",
        "I am still learning. Please ask in another way.",
        "I am not sure what you mean yet."
    ]

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break

        if not user_input:
            print("Bot: Please type something.")
            continue

        if user_input.startswith("teach:"):
            try:
                _, tag, sentence = user_input.split(":", 2)
                tag = tag.strip()
                sentence = sentence.strip()
                teach_new_example(tag, sentence)
                print(f"Bot: Learned new example for '{tag}'.")
            except ValueError:
                print("Bot: Wrong format. Use: teach:tag:sentence")
            continue

        predicted_tag, matched_pattern, corrected_input = predict_intent(user_input)

        if predicted_tag is None:
            print("Bot:", random.choice(fallback_responses))
            save_feedback(user_input, predicted_tag=None, correct_tag=None)
        else:
            response = generate_better_response(predicted_tag, matched_pattern, corrected_input)
            print(f"Bot: {response}")
            save_feedback(user_input, predicted_tag=predicted_tag, correct_tag=predicted_tag)

if __name__ == "__main__":
    chatbot()