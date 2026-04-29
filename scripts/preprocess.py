import pandas as pd
import jieba
import re
from sklearn.model_selection import train_test_split


CSV_FILE = '../data/taptap_game_reviews.csv'
STOPWORDS_FILE = '../data/cn_stopwords.txt'
OUTPUT_DIR = '../data/pretraitement/'

# 1. Charger la liste des mots vides (stopwords) pour un filtrage rapide
def load_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Supprimer les sauts de ligne et les espaces
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Attention : Fichier introuvable {filepath}. Le filtrage des mots vides sera ignoré.")
        return set()

print("Chargement des mots vides en cours...")
stopwords = load_stopwords(STOPWORDS_FILE)

# 2. Lire les données CSV et supprimer les valeurs nulles
print("Lecture des données CSV...")
df = pd.read_csv(CSV_FILE)[['review_content', 'sentiment']].dropna()
df = df.sample(frac=0.5, random_state=42)
# 3. Diviser le corpus : 80% Train, 10% Dev, 10% Test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 4. Fonction principale pour traiter le texte et générer les fichiers ARFF
def create_string_arff(data, filename):
    filepath = f"{OUTPUT_DIR}{filename}.arff"
    with open(filepath, 'w', encoding='utf-8') as f:
        # En-tête du fichier ARFF (Format String)
        f.write("@relation taptap_reviews\n\n")
        f.write("@attribute text string\n")
        f.write("@attribute class {pos,neg}\n\n")
        f.write("@data\n")
        
        valid_count = 0
        for _, row in data.iterrows():
            raw_text = str(row['review_content'])
            
            # Nettoyage strict : conserver uniquement les caractères chinois, les lettres et les chiffres.
            # Cela élimine automatiquement les sauts de ligne (\n, \r), les guillemets (", ') 
            # et les symboles qui pourraient corrompre le format ARFF dans Weka.
            clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', raw_text)
            
            # Tokenisation du texte chinois avec jieba
            tokens = jieba.lcut(clean_text)
            
            # Filtrer les espaces vides et les mots appartenant à la liste des mots vides
            final_tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in stopwords]
            
            # Rejoindre les tokens filtrés avec un seul espace
            tokenized_text = " ".join(final_tokens)
            
            # Vérifier si le texte contient encore des mots valides après le nettoyage
            if len(tokenized_text) > 0:
                label = 'pos' if row['sentiment'] == 1 else 'neg'
                
                # Écriture de l'instance au format Weka (texte encapsulé dans des guillemets simples)
                f.write(f"'{tokenized_text}',{label}\n")
                valid_count += 1
                
    print(f"Fichier généré avec succès : {filepath} ({valid_count} instances valides)")

# 5. Exécution du pipeline
print("Début du prétraitement (nettoyage, tokenisation, génération ARFF)...")
create_string_arff(train_df, "train_string")
create_string_arff(dev_df, "dev_string")
create_string_arff(test_df, "test_string")