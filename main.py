import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0  # Fake news
real["label"] = 1  # Real news

# Combine datasets
data = pd.concat([fake, real])

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Print class distribution
print("Class distribution:\n", data["label"].value_counts())

# Features and Labels
X = data["text"]
y = data["label"]

# Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vect = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print additional evaluation metrics
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# if wanna use grind
"""
from sklearn.model_selection import GridSearchCV

params = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)
"""
