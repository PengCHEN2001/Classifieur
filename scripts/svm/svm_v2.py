import pandas as pd
import jieba
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Charger le dataset
df = pd.read_csv("../../data/taptap_game_reviews.csv")

# 2. Garder uniquement les colonnes utiles
df = df[["review_content", "sentiment"]].dropna()

# 3. Stopwords chinois 
stopwords = set([
    "的", "了", "和", "是", "我", "也", "很", "就", "都", "而", "及", "与", "着"
])

# 4. Nettoyage + tokenisation
def preprocess(text):
    # garder uniquement chinois
    text = re.sub(r"[^\u4e00-\u9fff]", "", str(text))
    
    # jieba
    words = jieba.cut(text)
    
    # enlever stopwords
    words = [w for w in words if w not in stopwords and len(w) > 1]
    
    return " ".join(words)

df["clean_text"] = df["review_content"].apply(preprocess)

# 5. Features / labels
X = df["clean_text"]
y = df["sentiment"]

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. TF-IDF amélioré
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. SVM avec gestion déséquilibre
model = SVC(
    kernel="linear",
    class_weight="balanced",
    probability=True
)

model.fit(X_train_vec, y_train)

# 9. Prédiction
y_pred = model.predict(X_test_vec)

# 10. Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 11. Nouveaux commentaires à tester
# TEST : nouvelle prédiction
def predict_sentiment(text, model, vectorizer):
    # même preprocessing qu'avant !
    text = re.sub(r"[^\u4e00-\u9fff]", "", str(text))
    words = jieba.cut(text)
    words = [w for w in words if w not in stopwords and len(w) > 1]
    text = " ".join(words)

    # vectorisation (IMPORTANT : transform seulement)
    text_vec = vectorizer.transform([text])

    # prédiction
    pred = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec)[0]

    return pred, proba

print("\n====================")
print("Test personnalisé")
print("====================")


new_comment = [
    "烂游戏，讨厌死了，天天弹广告",
    "老是在关键时刻强制弹出广告 😡😡",
    "我可太喜欢这剧情了，太感人了",
    "角色好可爱！我要抽爆新角色！！！"
]

for c in new_comment:
    pred, proba = predict_sentiment(c, model, vectorizer)
    print("")
    print("Texte : ",  c)
    print("Prédiction : ", pred)
    print("Probabilités : ", proba)