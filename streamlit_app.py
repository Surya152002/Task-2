from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json
from sklearn.metrics import ndcg_score, jaccard_score
import numpy as np
import streamlit as st

def generate_user_profile(posts):
    user_text = ' '.join([post["text"] for post in posts])
    model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    inputs = tokenizer.encode(user_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    topics = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return topics[:20]

def generate_post_topics(post):
    model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    inputs = tokenizer.encode(post, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    topics = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return topics[:10]

def recommend_posts(users, posts, ground_truth):
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

# Load training and testing data
with open("Reddit_data_train.json", "r") as f:
    training_data = json.load(f)

with open("Reddit_data_test.json", "r") as f:
    testing_data = json.load(f)

# Extract relevant data
users = {user_id: [post["text"] for post in user_posts] for user_id, user_posts in testing_data.items()}
posts = {post["text"]: post["text"] for user_posts in testing_data.values() for post in user_posts}
ground_truth = {user_id: [post["dist_from_root"] for post in user_posts] for user_id, user_posts in testing_data.items()}


# Streamlit app
st.title("Reddit Recommendations Evaluation")

# Get user input for user_id
user_id = st.selectbox("Select a User ID", list(users.keys()))

# Generate recommendations for the selected user
user_recommendations = recommend_posts({user_id: users[user_id]}, posts, ground_truth)

# Evaluate recommendations
ndcg, jaccard = evaluate_recommendations(user_recommendations, ground_truth)

# Display results
st.write(f"NDCG Score: {ndcg}")
st.write(f"Jaccard Similarity Score: {jaccard}")
