import os
import glob
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import spacy


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Définition des chemins vers les dossiers de critiques
chemin_imdb = "imdb_smol"
chemin_neg = os.path.join(chemin_imdb, "neg")
chemin_pos = os.path.join(chemin_imdb, "pos")

# Récupération des fichiers négatifs et positifs
fichiers_neg = glob.glob(os.path.join(chemin_neg, "*.txt"))
fichiers_pos = glob.glob(os.path.join(chemin_pos, "*.txt"))

print(f"Nombre de fichiers de critiques négatives : {len(fichiers_neg)}")
print(f"Exemple de fichier négatif : {fichiers_neg[0] if fichiers_neg else 'Aucun'}")
print(f"Nombre de fichiers de critiques positives : {len(fichiers_pos)}")
print(f"Exemple de fichier positif : {fichiers_pos[0] if fichiers_pos else 'Aucun'}")

# Création des listes de fichiers
fichiers = fichiers_neg + fichiers_pos
# Création des labels: 0 pour négatif, 1 pour positif
all_labels = np.array([0] * len(fichiers_neg) + [1] * len(fichiers_pos))

# Division en ensembles d'entraînement et de test (70% train, 30% test)
fichiers_train, fichiers_test, labels_train, labels_test = train_test_split(
    fichiers, all_labels, test_size=0.3, random_state=42
)

print(f"Nombre d'exemples d'entraînement: {len(fichiers_train)}")
print(f"Nombre d'exemples de test: {len(fichiers_test)}")
print(f"Sentiments dans l'ensemble d'entraînement: {np.bincount(labels_train)}")
print(f"Sentiments dans l'ensemble de test: {np.bincount(labels_test)}")


def lire_fichier(chemin):
    
        with open(chemin, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
   

# Lecture du contenu des fichiers pour les ensembles d'entraînement et de test
contenus_train = [lire_fichier(fichier) for fichier in fichiers_train]
contenus_test = [lire_fichier(fichier) for fichier in fichiers_test]

# Vectorisation des textes (pas des chemins de fichiers)
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(contenus_train)  # Vectorise le contenu des fichiers
y_train = labels_train
X_test = vectorizer.transform(contenus_test)  # Vectorise le contenu des fichiers
y_test = labels_test

# Affichage des dimensions des données vectorisées
print(f"\nDimensions de X_train après vectorisation: {X_train.shape}")
print(f"Dimensions de X_test après vectorisation: {X_test.shape}")




# print(X_train[0, :])

# clf = LinearSVC(C=0.5)
clf = LinearSVC(dual=True, max_iter=8000) 

clf.fit(X_train, y_train)
clf.predict(X_test)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print('score SVM')
print(clf.score(X_test, y_test))





clf_logistique = LogisticRegression(max_iter=8000)  
clf_logistique.fit(X_train, y_train)  # y_train  0 pour neg et 1 pos
print('score régression logistique')
print(clf_logistique.score(X_test, y_test))  

y_pred_logistique = clf_logistique.predict(X_test)
y_proba_logistique = clf_logistique.predict_proba(X_test)[:, 1]  # Probabilité d'être positif

plt.figure(figsize=(10, 6))

indices_triés = np.argsort(y_proba_logistique)
probas_triées = y_proba_logistique[indices_triés]
labels_triés = y_test[indices_triés]

#  une droite des probabilités 
plt.plot(range(len(probas_triées)), probas_triées, 'b-', linewidth=2, label='Probabilités triées')

# Ajouter des points colorés pour montrer les vraies classes
plt.scatter(range(len(labels_triés)), probas_triées, 
           c=['green' if y==1 else 'red' for y in labels_triés])

# Ligne de seuil
plt.axhline(y=0.5, color='black', linestyle='--', label='Seuil de décision (0.5)')

# éléments du graphique
plt.title('Distribution des probabilités de la régression logistique')
plt.xlabel('Exemples triés par probabilité croissante')
plt.ylabel('Probabilité de sentiment positif')
plt.legend()
plt.grid(alpha=0.3)

# score
plt.text(len(y_test)*0.7, 0.2, 
         f'Précision: {clf_logistique.score(X_test, y_test):.2f}', 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()


#### Arbre de decisionn


clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=6)  # ou 'gini'
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)
print('score arbre de decision')
print(clf_tree.score(X_test, y_test))


# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_
# Cette propriété nous donne l'importance de chaque caractéristique (mot) dans la décision de l'arbre


# Affichage des 10 premiers mots les plus fréquents
indices_mots_frequents = np.argsort(clf_tree.feature_importances_)[-10:]
# Obtenir les noms des mots correspondants
mots_frequents = vectorizer.get_feature_names_out()[indices_mots_frequents]
# Obtenir les importances des mots
importances_mots = clf_tree.feature_importances_[indices_mots_frequents]
# affiche plot des mots les plus frequents et leur importance
plt.figure(figsize=(10, 6))
plt.barh(mots_frequents[::-1], importances_mots[::-1], color='skyblue')
plt.xlabel('Importance')
plt.title('Importance des 10 mots les plus fréquents dans l\'arbre de décision')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


### random Forest classifeur 
clf_forest = RandomForestClassifier(criterion='entropy', max_depth=6)  # ou 'gini'
clf_forest.fit(X_train, y_train)
y_pred_forest = clf_tree.predict(X_test)
print('score random forest')
print(clf_forest.score(X_test, y_test))


## modele bayesien naif
from sklearn.naive_bayes import MultinomialNB
clf_bayes = MultinomialNB()
clf_bayes.fit(X_train, y_train)
y_pred_bayes = clf_bayes.predict(X_test)
print('score bayesien')
print(clf_bayes.score(X_test, y_test))



### tokenisation avec spacy

doc_spacy = nlp(fichiers)
fichiers = words = [token.text.lower() for token in fichiers if not token.is_punct and not token.is_space]