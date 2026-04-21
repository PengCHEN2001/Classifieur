import pandas as pd
import jieba
import os
from sklearn.model_selection import train_test_split

# 1. Charger les données
df = pd.read_csv('../data/taptap_game_reviews.csv') 

# Nettoyage de base
df = df[['review_content', 'sentiment']].dropna()

# 2. Diviser : Train, Dev + Test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Diviser les 20% restants : 10% Dev, 10% Test
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

def process_data(data, folder_name):
    for i, row in data.iterrows():
        label = 'pos' if row['sentiment'] == 1 else 'neg'
        
        # Créer les dossiers : corpus/train/pos, corpus/dev/neg, etc.
        path = f"../data/pretraitement/corpus/{folder_name}/{label}"
        os.makedirs(path, exist_ok=True)
        
        # Segmentation chinoise (les emojis sont conservés)
        words = jieba.cut(str(row['review_content']))
        content = " ".join(words) 
        
        with open(f"{path}/{i}.txt", 'w', encoding='utf-8') as f:
            f.write(content)

print("Segmentation et création des dossiers en cours...")
process_data(train_df, "train")
process_data(dev_df, "dev")
process_data(test_df, "test")