import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv('Data/Twitter_Data.csv')


df = df.dropna(subset=['clean_text'])
df = df.sample(n=3000, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# تحويل المصفوفة لـ مصفوفة كثيفة علشان PCA
X_dense = X.toarray()

pca = PCA()
pca.fit(X_dense)

cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')  # خط عند 95%
plt.legend() 

plt.show()