import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

base_dir = os.path.dirname(__file__)

# Step 1: Load the Data
def load_data():
    """Load training and validation datasets and rename columns for clarity."""
    training_file_path = os.path.join(base_dir, 'data', 'twitter_training.csv')
    validation_file_path = os.path.join(base_dir, 'data', 'twitter_validation.csv')

    # Load datasets
    training_data = pd.read_csv(training_file_path)
    validation_data = pd.read_csv(validation_file_path)

    # Rename columns
    training_data.columns = ['id', 'category', 'sentiment', 'tweet']
    validation_data.columns = ['id', 'category', 'sentiment', 'tweet']

print("Updated Columns in Training Data:")
print(training_data.columns)

print("\nFirst few rows of Training Data:")
print(training_data.head())

print("\nUpdated Columns in Validation Data:")
print(validation_data.columns)

print("\nFirst few rows of Validation Data:")
print(validation_data.head())

# Step 2: Data Preprocessing

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Remove URLs, mentions, hashtags, and non-alphabet characters
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)  # Convert to string
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

# Tokenize, lemmatize and remove stop words
    tokens = word_tokenize(text.lower())
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return cleaned_text

def preprocess_data(data):
    """Apply preprocessing to the 'tweet' column in the data frame."""
    training_data['tweet'] = training_data['tweet'].astype(str)
    validation_data['tweet'] = validation_data['tweet'].astype(str)

training_data['cleaned_text'] = training_data['tweet'].apply(preprocess_text)
validation_data['cleaned_text'] = validation_data['tweet'].apply(preprocess_text)

# Step 3: Feature Extraction
def extract_features(training_data, validation_data):
    """Extract TF-IDF features from the cleaned text data."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(training_data['cleaned_text'])
    y_train = training_data['sentiment']
    X_valid = vectorizer.transform(validation_data['cleaned_text'])
    y_valid = validation_data['sentiment']
    return X_train, y_train, X_valid, vectorizer

# Step 4: Model Training
def train_model(X_train, y_train):
    """Train the Multinomial Naive Bayes model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_valid, y_valid):
    """Evaluate the trained model using validation data and print results."""
    y_pred = model.predict(X_valid)
    print("Classification Report:")
    print(classification_report(y_valid, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))

# Step 6: Model Optimization
def optimize_model(X_train, y_train):
    """Optimize the model using GridSearchCV to find the best hyperparameters."""
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Step 7: Finalize and Save the Model
def save_model(model, vectorizer):
    """Save the trained model and the vectorizer to a file."""
    model_file_path = os.path.join(base_dir, 'sentiment_analysis_model.pkl')
    joblib.dump(grid_search.best_estimator_, model_file_path)
    print(f"Model saved as {model_file_path}")

def main():
    #Load data
    training_data, validation_data = load_data()
    print("Updated Columns in Training Data:")
    print(training_data.columns)
    print("\nFirst few rows of Training Data:")
    print(training_data.head())

    print("\nUpdated Columns in Validation Data:")
    print(validation_data.columns)
    print("\nFirst few rows of Validation Data:")
    print(validation_data.head())

    # Preprocess data
    training_data = preprocess_data(training_data)
    validation_data = preprocess_data(validation_data)

    # Extract features
    X_train, y_train, X_valid, y_valid, vectorizer = extract_features(training_data, validation_data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_valid, y_valid)

    # Optimize model
    best_model = optimize_model(X_train, y_train)

    # Save model
    save_model(best_model, vectorizer)

if __name__ == "__main__":
    main()
