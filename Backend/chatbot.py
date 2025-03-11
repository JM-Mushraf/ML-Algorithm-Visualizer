import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np

# Step 1: Create Dataset
data = {
    "question": [
        "What is regression?",
        "What is a decision tree?",
        "What is max_depth or max depth in decision trees?",
        "What is overfitting?",
        "What is a neural network?",
        "What is clustering?",
        "What is the difference between classification and regression?",
        "What is gradient descent?",
        "What is a loss function?",
        "What is regularization?"
    ],
    "answer": [
        "Regression is a statistical method used to predict a continuous outcome variable based on one or more predictor variables.",
        "A decision tree is a flowchart-like structure where each internal node represents a decision based on a feature, each branch represents the outcome of the decision, and each leaf node represents a class label or a continuous value.",
        "max_depth is a hyperparameter in decision trees that controls the maximum depth of the tree. A deeper tree can model more complex patterns but may lead to overfitting.",
        "Overfitting occurs when a model learns the training data too well, capturing noise and details, which negatively impacts its performance on unseen data.",
        "A neural network is a series of algorithms that attempts to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.",
        "Clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.",
        "Classification is used to predict discrete labels, while regression is used to predict continuous values.",
        "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent.",
        "A loss function measures how well a model's predictions match the actual target values.",
        "Regularization is a technique used to prevent overfitting by adding a penalty to the loss function based on the complexity of the model."
    ]
}

df = pd.DataFrame(data)

# Step 2: Preprocessing
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df['processed_question'] = df['question'].apply(preprocess_text)
df['processed_question_str'] = df['processed_question'].apply(lambda x: ' '.join(x))

# Step 3: Bag of Words
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(df['processed_question_str'])

# Step 4: Word2Vec
sentences = df['processed_question'].tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Function to get average Word2Vec vector for a sentence
def get_sentence_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Create Word2Vec vectors for all questions
df['word2vec_vector'] = df['processed_question'].apply(lambda x: get_sentence_vector(x, word2vec_model))

# Step 5: Chatbot Function
def chatbot(user_input, method='word2vec'):
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    
    if method == 'bow':
        # Convert user input to BoW
        input_bow = vectorizer.transform([' '.join(processed_input)])
        # Compute cosine similarity with all questions
        similarities = cosine_similarity(input_bow, bow_matrix)
        # Get the index of the most similar question
        best_match_idx = similarities.argmax()
    elif method == 'word2vec':
        # Convert user input to Word2Vec vector
        input_vector = get_sentence_vector(processed_input, word2vec_model)
        # Compute cosine similarity with all question vectors
        similarities = [cosine_similarity([input_vector], [q_vec])[0][0] for q_vec in df['word2vec_vector']]
        # Get the index of the most similar question
        best_match_idx = np.argmax(similarities)
    else:
        return "Invalid method. Choose 'bow' or 'word2vec'."

    # Return the best-matching answer
    return df.loc[best_match_idx, 'answer']

# Step 6: Test the Chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    response = chatbot(user_input, method='bow')  # Change to 'word2vec' for Word2Vec method
    print(f"Chatbot: {response}")