import pandas as pd
import jieba

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Charger le dataset
df = pd.read_csv("../../data/taptap_game_reviews.csv")

# 2. Garder uniquement les colonnes utiles
df = df[["review_content", "sentiment"]].dropna()

# 3. Tokenisation avec jieba
def jieba_tokenizer(text):
    return " ".join(jieba.cut(text))

df["cut_text"] = df["review_content"].apply(jieba_tokenizer)

# 4. Séparer features et labels
X = df["cut_text"]
y = df["sentiment"]

# 5. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Vectorisation TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Modèle SVM
model = SVC(kernel="linear", probability=True)  # linear souvent meilleur pour texte
model.fit(X_train_vec, y_train)

# 8. Prédiction
y_pred = model.predict(X_test_vec)

# 9. Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. Nouveaux commentaires à tester
new_comment = [
    "烂游戏，讨厌死了，天天弹广告",
    "老是在关键时刻强制弹出广告 😡😡",
    "我可太喜欢这剧情了，太感人了",
    "角色好可爱！我要抽爆新角色！！！"
]


# TEST : nouvelle prédiction
def predict_sentiment(text, model, vectorizer):
    # même preprocessing qu'avant !
    words = jieba.cut(text)
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