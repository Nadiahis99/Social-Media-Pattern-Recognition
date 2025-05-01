import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Data/Twitter_Data.csv'  
df = pd.read_csv(file_path)
def plot_category_distribution(df):
    label_map = {-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'}
    df['label'] = df['category'].map(label_map)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label', palette='Set2')
    plt.title('Distribution of Sentiment Categories')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    
    plt.savefig('outputs/category_distribution.png')
    plt.show()

plot_category_distribution(df)
