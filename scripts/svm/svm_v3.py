import pandas as pd
import jieba
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Charger le dataset
df = pd.read_csv("../../data/taptap_game_reviews.csv")

# 2. Colonnes utiles
df = df[["review_content", "sentiment"]].dropna()

# 3. Stopwords chinois
stopwords = set([
    "的", "了", "和", "是", "我", "也", "很", "就", "都", "而", "及", "与", "着"
])

# 4. Préprocessing
def preprocess(text):
    text = re.sub(r"[^\u4e00-\u9fff]", "", str(text))
    words = jieba.cut(text)
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

# 7. TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. GridSearch pour SVM
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    SVC(class_weight="balanced"),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train_vec, y_train)

# 9. Meilleur modèle
best_model = grid.best_estimator_

print("Meilleurs paramètres :", grid.best_params_)

# 10. Prédiction
y_pred = best_model.predict(X_test_vec)

# 11. Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 12. Nouveaux commentaires à tester
new_comment = [
    "烂游戏，讨厌死了，天天弹广告",
    "老是在关键时刻强制弹出广告 😡😡",
    "我可太喜欢这剧情了，太感人了",
    "角色好可爱！我要抽爆新角色！！！"
]

# 13. Préprocessing (IDENTIQUE à l'entraînement)
new_comment_clean = [preprocess(text) for text in new_comment]

# 14. Vectorisation (ne jamais refit !)
new_comment_vec = vectorizer.transform(new_comment_clean)

# 15. Prédictions
predictions = best_model.predict(new_comment_vec)

# 16. Affichage des résultats
for text, pred in zip(new_comment, predictions):
    sentiment = "1" if pred == 1 else "1"
    print(f"{text} -> {sentiment}")