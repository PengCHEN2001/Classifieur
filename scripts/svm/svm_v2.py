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

# 3. Stopwords chinois (exemple minimal, tu peux enrichir)
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
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# 9. Prédiction
y_pred = model.predict(X_test_vec)

# 10. Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 11. Nouveaux commentaires à tester
new_comment = [
    "烂游戏，讨厌死了，天天弹广告",
    "老是在关键时刻强制弹出广告 😡😡",
    "我可太喜欢这剧情了，太感人了",
    "角色好可爱！我要抽爆新角色！！！"
]

# 12. Préprocessing (identique à l'entraînement)
new_comment_clean = [preprocess(text) for text in new_comment]

# 13. Vectorisation (⚠️ transform seulement)
new_comment_vec = vectorizer.transform(new_comment_clean)

# 14. Prédictions
predictions = model.predict(new_comment_vec)

# 15. Affichage
for text, pred in zip(new_comment, predictions):
    sentiment = "1" if pred == 1 else "0"
    print(f"{text} -> {sentiment}")