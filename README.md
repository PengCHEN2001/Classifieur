# Classification de sentiments des avis de jeux mobiles chinois (TapTap)

Ce projet vise à réaliser une **classification binaire des sentiments (positif / négatif)** sur des avis de jeux mobiles rédigés en chinois, issus de la plateforme TapTap.

Plusieurs approches classiques de machine learning sont mises en œuvre et comparées à l’aide de **scikit-learn (Python)** et de **WEKA**.

Pour plus de détails concernant la méthodologie, les expériences et l’analyse des résultats, veuillez vous référer au rapport.

---

## Jeu de données

- Source : corpus TapTap (Kaggle)
- Langue : chinois
- Tâche : classification binaire de sentiments
- Classes :
  - Positif (1)
  - Négatif (0)
- Taille : environ 39985 avis
- Site ：https://www.kaggle.com/datasets/karwinwang/taptap-mobile-game-reviews-chinese/data
---

##  Modèles utilisés

###  scikit-learn
- Naïve Bayes
- SVM (noyau linéaire et RBF)

###  WEKA
- VotedPerceptron
- RandomForest (bagging avec RandomTree)


