import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Charger les données
# =========================
df = pd.read_csv("../../data/taptap_game_reviews.csv")

print("Aperçu des données :")
print(df.head())
print("\nColonnes disponibles :", df.columns)

# =========================
# 2. Choisir les colonnes
# =========================


TEXT_COLUMN = "review_content"       # <-- texte des commentaires
LABEL_COLUMN = "sentiment"   # <-- 0 / 1

# Vérification rapide
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# =========================
# 3. Split train / test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. Vectorisation (TF-IDF)
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # unigram + bigram
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. Modèle Naive Bayes
# =========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# =========================
# 6. Évaluation
# =========================
y_pred = model.predict(X_test_vec)

print("\n====================")
print("Résultats")
print("====================")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# 7. Test sur nouveau texte
# =========================
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return pred, proba

# Exemple
test_comment = "烂游戏，讨厌死了，天天弹广告"
pred, proba = predict_sentiment(test_comment)

print("\n====================")
print("Test personnalisé")
print("====================")
print("Texte :", test_comment)
print("Prédiction :", pred)
print("Probabilités [0,1] :", proba)