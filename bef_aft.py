import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('Data/Twitter_Data.csv')
df = df.dropna(subset=['clean_text', 'category'])
df = df.sample(n=10000, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

# ============================
# 1. Accuracy قبل PCA
# ============================
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model_orig = SVC(kernel='linear')
svm_model_orig.fit(X_train_orig, y_train)
y_pred_orig = svm_model_orig.predict(X_test_orig)
accuracy_orig = accuracy_score(y_test, y_pred_orig)

print(f"Original number of features: {X.shape[1]}")
print(f"Accuracy before Dimension Reduction: {accuracy_orig:.4f}")

# ============================
# 2. Accuracy بعد PCA
# ============================
X_dense = X.toarray()

pca = PCA(n_components=.95, random_state=42)
X_reduced = pca.fit_transform(X_dense)

X_train_pca, X_test_pca, _, _ = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

svm_model_pca = SVC(kernel='linear')
svm_model_pca.fit(X_train_pca, y_train)
y_pred_pca = svm_model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print(f"Number of features after PCA: {pca.n_components_}")
print(f"Accuracy after Dimension Reduction: {accuracy_pca:.4f}")

