import tkinter as tk
from tkinter import messagebox
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import pandas as pd


df = pd.read_csv('Data/Twitter_Data.csv')
df = df.dropna(subset=['clean_text'])
df = df.sample(n=3000, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['category'] 

pca = PCA(n_components=0.95, random_state=42)
X_reduced = pca.fit_transform(X.toarray())  

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

def classify_tweet():
    tweet = entry.get()  
    
    if not tweet:
        messagebox.showwarning("Input Error", "Please enter a tweet.")  
    else:
        tweet_vector = vectorizer.transform([tweet])
        tweet_vector_reduced = pca.transform(tweet_vector.toarray())  # تطبيق PCA

        category = svm_model.predict(tweet_vector_reduced)
        
        sentiment_label = "Positive" if category == 1 else "Negative"
        messagebox.showinfo("Tweet Classification", f"The sentiment of the tweet is: {sentiment_label}")

window = tk.Tk()
window.title("Tweet Sentiment Classifier")

label = tk.Label(window, text="Enter a tweet:")
label.pack(pady=10)

entry = tk.Entry(window, width=50)
entry.pack(pady=10)

classify_button = tk.Button(window, text="Classify Tweet", command=classify_tweet)
classify_button.pack(pady=20)

window.mainloop()
