# Analyse de Sentiment sur Critiques de Films IMDB

## Aperçu du Projet
Ce projet implémente une analyse de sentiment sur des critiques de films IMDB en utilisant diverses techniques d'apprentissage automatique. L'objectif est de classifier automatiquement les critiques comme positives ou négatives à partir de leur contenu textuel.

## Données
- **Source** : Collection de critiques de films IMDB
- **Structure** : Fichiers texte organisés dans deux dossiers
  - `imdb_smol/pos/` : Critiques positives
  - `imdb_smol/neg/` : Critiques négatives
- **Répartition** : 70% pour l'entraînement, 30% pour les tests

## Méthodologie

### Prétraitement du Texte
```python
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(contenus_train)
X_test = vectorizer.transform(contenus_test)
```

Le texte est transformé en vecteurs numériques via la méthode "sac de mots" (bag-of-words), qui compte les occurrences de chaque mot tout en ignorant les mots vides anglais.

### Modèles de Classification
1) SVM Linéaire
2) Régression Logistique
3) Arbre de Décision
4) Forêt Aléatoire (Random Forest)
5) Classificateur Bayésien Naïf



L'arbre de décision utilise l'entropie comme mesure de l'impureté dans les données :

$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

où :

$S$ est un ensemble d'exemples
$p_i$ est la proportion d'exemples de la classe $i$
$c$ est le nombre de classes (2 dans notre cas)
L'entropie mesure le "désordre" dans un ensemble de données :

Entropie minimale (0) : Tous les exemples appartiennent à la même classe
Entropie maximale (1 pour deux classes) : Répartition égale entre les classes
À chaque division, l'arbre sélectionne la caractéristique qui permet de maximiser le gain d'information (réduction d'entropie), créant ainsi des sous-ensembles plus "purs"


mportance des Caractéristiques
Le code extrait les mots les plus influents dans la prise de décision :

```python
indices_mots_frequents = np.argsort(clf_tree.feature_importances_)[-10:]
mots_frequents = vectorizer.get_feature_names_out()[indices_mots_frequents]
```

Grâce à ça on peut voir les mots les plus discriminants qui influent le plus sur le sentiment négatif ou positif dans l'arbre de décision.


## Sortie terminal


```bash
  from pandas.core import (
Nombre de fichiers de critiques négatives : 301
Exemple de fichier négatif : imdb_smol/neg/233_1.txt
Nombre de fichiers de critiques positives : 301
Exemple de fichier positif : imdb_smol/pos/138_7.txt
Nombre d'exemples d'entraînement: 421
Nombre d'exemples de test: 181
Sentiments dans l'ensemble d'entraînement: [201 220]
Sentiments dans l'ensemble de test: [100  81]

Dimensions de X_train après vectorisation: (421, 9572)
Dimensions de X_test après vectorisation: (181, 9572)
              precision    recall  f1-score   support

           0       0.92      0.76      0.83       100
           1       0.76      0.91      0.83        81

    accuracy                           0.83       181
   macro avg       0.84      0.84      0.83       181
weighted avg       0.84      0.83      0.83       181

score SVM
0.8287292817679558
score régression logistique
0.8066298342541437
score arbre de decision
0.5580110497237569
score random forest
0.7016574585635359
score bayesien
0.8342541436464088



```