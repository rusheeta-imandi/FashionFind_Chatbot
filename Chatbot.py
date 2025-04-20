import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tabulate import tabulate

# Step 2: Load the dataset (you can replace this with your own dataset)
# Assume you have a CSV file with columns: 'product_id', 'product_name', 'description', 'category', 'rating', 'review_count', 'price'
# Example dataset:
data = pd.read_csv('products_data.csv')

# Step 3: Preprocess the data
# For simplicity, let's use the 'description' column for recommendation
data['description'] = data['description'].fillna('')  # Fill missing descriptions with empty string

# Step 4: Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'])

# Step 5: Compute the cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Step 6: Define a function to get recommendations
def get_recommendations(product_name, num_recommendations=5):
    idx = data[data['product_name'] == product_name].index
    if len(idx) == 0:
        return pd.DataFrame()  # Return an empty DataFrame if product not found
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    product_indices = [i[0] for i in sim_scores]
    return data[['product_name', 'size', 'color', 'rating', 'review_count', 'price']].iloc[product_indices]

# Step 7: Define a function to handle user queries
def handle_user_query(user_input, num_recommendations=5):
    recommendations = get_recommendations(user_input, num_recommendations)
    return recommendations

# Example of how to use the chatbot function:
def chatbot():
    print("Welcome to the Product Recommender Chatbot!")
    while True:
        user_input = input("Please enter a product name (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        recommendations = handle_user_query(user_input)
        if recommendations.empty:
            print("Product not found. Please try again.")
        else:
            print(f"Recommended products for {user_input}:")
            print(tabulate(recommendations, headers='keys', tablefmt='pretty', colalign=("center",)))

# Run the chatbot
chatbot()
