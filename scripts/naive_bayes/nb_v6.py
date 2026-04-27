import pandas as pd
import re
import emoji
import jieba

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../../data/taptap_game_reviews.csv")

TEXT_COLUMN = "review_content"
LABEL_COLUMN = "sentiment"

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# =========================
# 2. CLEANING OPTIMISÉ
# =========================
def clean_text(text):
    text = str(text).lower()

    # emojis → texte émotionnel
    text = emoji.demojize(text)

    # URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# =========================
# 3. SPLIT TRAIN / DEV / TEST
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Train:", len(X_train), "Dev:", len(X_dev), "Test:", len(X_test))

# =========================
# 4. JIEBA TOKENIZATION
# =========================
def chinese_tokenizer(text):
    return jieba.lcut(text)

# =========================
# 5. TF-IDF AVEC JIEBA
# =========================
vectorizer = TfidfVectorizer(
    tokenizer=chinese_tokenizer,  
    token_pattern=None,            
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 6. GESTION DÉSÉQUILIBRE
# =========================
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# =========================
# 7. TRAIN + TUNING ALPHA
# =========================
best_model = None
best_score = 0
best_alpha = None

for alpha in [0.1, 0.3, 0.5, 1.0]:

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_vec, y_train, sample_weight=sample_weights)

    dev_pred = model.predict(X_dev_vec)
    score = accuracy_score(y_dev, dev_pred)

    print(f"alpha={alpha} | DEV accuracy={score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_alpha = alpha

print("\n Best alpha:", best_alpha)

# =========================
# 8. FINAL TEST EVALUATION
# =========================
test_pred = best_model.predict(X_test_vec)

print("\n====================")
print("FINAL TEST RESULTS")
print("====================")

print("Accuracy TEST:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# =========================
# 9. PREDICTION FUNCTION
# =========================
def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = best_model.predict(vec)[0]
    proba = best_model.predict_proba(vec)[0]
    return pred, proba

# =========================
# 10. TEST
# =========================
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