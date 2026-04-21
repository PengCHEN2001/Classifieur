import pandas as pd
import jieba
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Charger
df = pd.read_csv("../../data/taptap_game_reviews.csv")
df = df[["review_content", "sentiment"]].dropna()

# 2. Stopwords
stopwords = set([
    "的", "了", "和", "是", "我", "也", "很", "就", "都", "而", "及", "与", "着"
])

# 3. Préprocessing
def preprocess(text):
    text = re.sub(r"[^\u4e00-\u9fff]", "", str(text))
    words = jieba.cut(text)
    words = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(words)

df["clean_text"] = df["review_content"].apply(preprocess)

X = df["clean_text"]
y = df["sentiment"]

# 4. Split TEST (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Split TRAIN / DEV (80% → 60% train, 20% dev)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# 6. TF-IDF (fit uniquement sur TRAIN)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
X_test_vec = vectorizer.transform(X_test)

# 7. GridSearch sur TRAIN (validation interne = DEV via CV)
param_grid = {
    "C": [0.5, 1, 5],
    "kernel": ["linear"],
}

grid = GridSearchCV(
    SVC(class_weight="balanced"),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train_vec, y_train)

best_model = grid.best_estimator_
print("Meilleurs paramètres :", grid.best_params_)

# 8. Évaluation sur DEV (important)
y_dev_pred = best_model.predict(X_dev_vec)

print("\n=== DEV RESULTS ===")
print("Accuracy:", accuracy_score(y_dev, y_dev_pred))
print(classification_report(y_dev, y_dev_pred))

# 9. Réentraînement sur TRAIN + DEV
X_final = pd.concat([X_train, X_dev])
y_final = pd.concat([y_train, y_dev])

# IMPORTANT : refit vectorizer sur train+dev
X_final_vec = vectorizer.fit_transform(X_final)

final_model = SVC(**grid.best_params_, class_weight="balanced")
final_model.fit(X_final_vec, y_final)

# 10. Transformation du test (sans refit !)
X_test_vec = vectorizer.transform(X_test)

# 11. Évaluation finale
y_test_pred = final_model.predict(X_test_vec)

print("\n=== FINAL TEST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

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

    return pred

new_comment = [
    "烂游戏，讨厌死了，天天弹广告",
    "老是在关键时刻强制弹出广告 😡😡",
    "我可太喜欢这剧情了，太感人了",
    "角色好可爱！我要抽爆新角色！！！"
]

for c in new_comment:
    print(c, "->", predict_sentiment(c, final_model, vectorizer))