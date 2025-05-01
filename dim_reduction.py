import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('Data/Twitter_Data.csv')
df = df.dropna(subset=['clean_text'])
df = df.sample(n=10000
               , random_state=42)
df = df.dropna(subset=['clean_text', 'category'])  # نحذف أي صف فيه نص أو تصنيف ناقص

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['category']  

pca = PCA(n_components=2000, random_state=42)
X_reduced = pca.fit_transform(X)

print(f"Original number of features: {X.shape[1]}")
print(f"Number of features after PCA: {pca.n_components_}")

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Dimension Reduction: {accuracy:.4f}")
