import pandas as pd
import jieba
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
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

# 4. Préprocessing amélioré
def preprocess(text):
    text = str(text)

    # garder chinois + lettres + chiffres
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)

    words = jieba.cut(text)

    # on garde les mots de longueur 1 (important en chinois)
    words = [w for w in words if w not in stopwords]

    return " ".join(words)

df["clean_text"] = df["review_content"].apply(preprocess)

# 5. Features / labels
X = df["clean_text"]
y = df["sentiment"]

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Pipeline 
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )),
    ("clf", SVC(class_weight="balanced", probability=True))
])

# 8. GridSearch corrigé
param_grid = [
    {
        "clf__kernel": ["linear"],
        "clf__C": [0.1, 1, 10]
    },
    {
        "clf__kernel": ["rbf"],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"]
    }
]

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# 9. Entraînement
grid.fit(X_train, y_train)

# 10. Meilleur modèle
best_model = grid.best_estimator_

print("Meilleurs paramètres :", grid.best_params_)

# 11. Prédiction
y_pred = best_model.predict(X_test)

# 12. Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 13. Fonction de prédiction propre
def predict_sentiment(text, model):
    text = preprocess(text)
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
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
    pred, proba = predict_sentiment(c, best_model)
    print("\nTexte :", c)
    print("Prédiction :", pred)
    print("Probabilités :", proba)