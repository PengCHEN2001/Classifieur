import pandas as pd
import jieba

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

TEXT_COLUMN = "review_content"
LABEL_COLUMN = "sentiment"

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# =========================
# 2. TOKENISATION CHINOISE (JIEBA)
# =========================
def chinese_tokenizer(text):
    return jieba.lcut(text)

# =========================
# 3. SPLIT TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. TF-IDF AVEC JIEBA
# =========================
vectorizer = TfidfVectorizer(
    tokenizer=chinese_tokenizer, 
    token_pattern=None,          
    ngram_range=(1, 2),
    max_features=20000             
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. MODÈLE NAIVE BAYES
# =========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# =========================
# 6. ÉVALUATION
# =========================
y_pred = model.predict(X_test_vec)

print("\n====================")
print("Résultats")
print("====================")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# 7. TEST PERSONNALISÉ
# =========================
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return pred, proba

print("\n====================")
print("Test personnalisé")
print("====================")

test_comments = [
    "烂游戏，讨厌死了，天天弹广告", 
    "老是在关键时刻强制弹出广告 😡😡",
    "我可太喜欢这剧情了，太感人了",
    "角色好可爱！我要抽爆新角色！！！"
    ]

for test_comment in test_comments :
    pred, proba = predict_sentiment(test_comment)
    print("")
    print("Texte :", test_comment)
    print("Prédiction :", pred)
    print("Probabilités [0,1] :", proba)