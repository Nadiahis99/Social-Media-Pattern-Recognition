import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

file_path = 'Data/Twitter_Data.csv'
df = pd.read_csv(file_path)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+', '', text)  
        text = re.sub(r'[^\x00-\x7F]+', '', text)   
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)  
        text = text.lower()                        
        text = ' '.join([word for word in text.split() if word not in stop_words])  
    return text

df['clean_text'] = df['clean_text'].apply(clean_text)

df = df.dropna(subset=['clean_text', 'category'])

vectorizer = TfidfVectorizer(stop_words='english')  
X = vectorizer.fit_transform(df['clean_text'])

y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression()

log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred))
