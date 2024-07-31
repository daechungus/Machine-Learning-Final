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

# Step 1
training_file_path = os.path.join(base_dir, 'data', 'twitter_training.csv')
validation_file_path = os.path.join(base_dir, 'data', 'twitter_validation.csv')
training_data = pd.read_csv(training_file_path)
validation_data = pd.read_csv(validation_file_path)

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

# Step 2
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)  # Convert to string
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text.lower())
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return cleaned_text

training_data['tweet'] = training_data['tweet'].astype(str)
validation_data['tweet'] = validation_data['tweet'].astype(str)

training_data['cleaned_text'] = training_data['tweet'].apply(preprocess_text)
validation_data['cleaned_text'] = validation_data['tweet'].apply(preprocess_text)

# Step 3
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(training_data['cleaned_text'])
y_train = training_data['sentiment']
X_valid = vectorizer.transform(validation_data['cleaned_text'])
y_valid = validation_data['sentiment']

# Step 4
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)

print("Classification Report:")
print(classification_report(y_valid, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_pred))

# Step 5
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Step 6: Finalize and Save the Model
model_file_path = os.path.join(base_dir, 'sentiment_analysis_model.pkl')
joblib.dump(grid_search.best_estimator_, model_file_path)
print(f"Model saved as {model_file_path}")
