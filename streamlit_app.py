from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import torch
from sklearn.metrics import ndcg_score, jaccard_score
import numpy as np
import streamlit as st

# Load the GPT-Neo model and tokenizer
model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Load training and testing data
with open("Reddit_data_train.json", "r") as f:
    training_data = json.load(f)

with open("Reddit_data_test.json", "r") as f:
    testing_data = json.load(f)

# Function to generate user profile based on posts
def generate_user_profile(posts):
    user_text = ' '.join([post["text"] for post in posts])
    inputs = tokenizer.encode(user_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    topics = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return topics[:20]

# Function to generate post topics
def generate_post_topics(post):
    inputs = tokenizer.encode(post, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    topics = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return topics[:10]

# Function to recommend posts to users
def recommend_posts(users, posts):
    recommendations = {}

    for user_id, user_posts in users.items():
        user_profile = generate_user_profile(user_posts)

        for post_id, post_content in posts.items():
            post_topics = generate_post_topics(post_content)

            jaccard_sim = len(set(user_profile) & set(post_topics)) / len(set(user_profile) | set(post_topics))

            if user_id not in recommendations:
                recommendations[user_id] = []

            recommendations[user_id].append((post_id, jaccard_sim))

    return recommendations

# Function to evaluate recommendations
def evaluate_recommendations(recommendations, ground_truth):
    predicted_labels = []
    true_labels = []

    for user_id, user_recommendations in recommendations.items():
        user_ground_truth = ground_truth.get(user_id, [])

        predicted_labels.append([post_id for post_id, _ in user_recommendations])
        true_labels.append([post["dist_from_root"] for post in user_ground_truth])

    ndcg = ndcg_score(true_labels, predicted_labels)
    jaccard = np.mean([jaccard_score(set(true), set(pred)) for true, pred in zip(true_labels, predicted_labels)])

    return ndcg, jaccard

# Extract relevant data from training and testing data
users = {user_id: [post["text"] for post in user_posts] for user_id, user_posts in testing_data.items()}
posts = {item['text']: item['text'] for item in training_data}
ground_truth = {item['text']: item['GT'] for item in testing_data}

# Generate recommendations
recommendations = recommend_posts(users, posts)

# Evaluate recommendations
ndcg, jaccard = evaluate_recommendations(recommendations, ground_truth)

# Display results
print(f"NDCG Score: {ndcg}")
print(f"Jaccard Similarity Score: {jaccard}")
