import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

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

print(df.head())
    
    



df = df.dropna(subset=['clean_text'])

vectorizer = TfidfVectorizer(stop_words='english')  
X = vectorizer.fit_transform(df['clean_text'])  

print(X.shape)  
