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
# Ajout d'un modèle d'arbre de décision (DecisionTreeClassifier)
from sklearn.tree import DecisionTreeClassifier, plot_tree


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
all_files = fichiers_neg + fichiers_pos
# Création des labels: 0 pour négatif, 1 pour positif
all_labels = np.array([0] * len(fichiers_neg) + [1] * len(fichiers_pos))

# Division en ensembles d'entraînement et de test (70% train, 30% test)
files_train, files_test, labels_train, labels_test = train_test_split(
    all_files, all_labels, test_size=0.3, random_state=42
)

print(f"Nombre d'exemples d'entraînement: {len(files_train)}")
print(f"Nombre d'exemples de test: {len(files_test)}")
print(f"Distribution des sentiments dans l'ensemble d'entraînement: {np.bincount(labels_train)}")
print(f"Distribution des sentiments dans l'ensemble de test: {np.bincount(labels_test)}")

# Fonction pour lire le contenu d'un fichier
def lire_fichier(chemin):
    try:
        with open(chemin, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture de {chemin}: {e}")
        return ""

# Lecture du contenu des fichiers pour les ensembles d'entraînement et de test
contenus_train = [lire_fichier(fichier) for fichier in files_train]
contenus_test = [lire_fichier(fichier) for fichier in files_test]

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
clf = LinearSVC(dual=True, max_iter=8000) # initialize the classifier, costructeur

clf.fit(X_train, y_train)
clf.predict(X_test)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print(clf.score(X_test, y_test))







clf_logistique = LogisticRegression(max_iter=8000)  # Utiliser 'liblinear' pour les petits ensembles de données
clf_logistique.fit(X_train, y_train)  # y_train contient 0 ou 1

print(clf_logistique.score(X_test, y_test))  # y_test contient 0 ou 1

# Obtenir les prédictions de la régression logistique
y_pred_logistique = clf_logistique.predict(X_test)
y_proba_logistique = clf_logistique.predict_proba(X_test)[:, 1]  # Probabilité d'être positif

# Créer une figure simple
plt.figure(figsize=(10, 6))

# Trier les exemples par probabilité pour une visualisation plus claire
indices_triés = np.argsort(y_proba_logistique)
probas_triées = y_proba_logistique[indices_triés]
labels_triés = y_test[indices_triés]

# Tracer une droite des probabilités (triées)
plt.plot(range(len(probas_triées)), probas_triées, 'b-', linewidth=2, label='Probabilités triées')

# Ajouter des points colorés pour montrer les vraies classes
plt.scatter(range(len(labels_triés)), probas_triées, 
           c=['green' if y==1 else 'red' for y in labels_triés],
           alpha=0.5, s=20)

# Ligne de seuil
plt.axhline(y=0.5, color='black', linestyle='--', label='Seuil de décision (0.5)')

# Ajouter les éléments du graphique
plt.title('Distribution des probabilités de la régression logistique')
plt.xlabel('Exemples triés par probabilité croissante')
plt.ylabel('Probabilité de sentiment positif')
plt.legend()
plt.grid(alpha=0.3)

# Ajouter les informations de score
plt.text(len(y_test)*0.7, 0.2, 
         f'Précision: {clf_logistique.score(X_test, y_test):.2f}', 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()


#### Arbre de decisionn


clf_tree = DecisionTreeClassifier(criterion='entropy')  # ou 'gini'
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)
print(classification_report(y_test, y_pred_tree))
print(clf_tree.score(X_test, y_test))
# Visualisation de l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(clf_tree, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=["Négatif", "Positif"])
plt.title("Arbre de décision pour la classification des critiques de films")
plt.show()

# Affichage des 10 premiers mots les plus fréquents
# Obtenir les indices des 10 premiers mots les plus fréquents
indices_mots_frequents = np.argsort(clf_tree.feature_importances_)[-10:]
# Obtenir les noms des mots correspondants
mots_frequents = vectorizer.get_feature_names_out()[indices_mots_frequents]
# Obtenir les importances des mots
importances_mots = clf_tree.feature_importances_[indices_mots_frequents]
# Tracer les mots les plus fréquents
plt.figure(figsize=(10, 6))
plt.barh(mots_frequents[::-1], importances_mots[::-1], color='skyblue')
plt.xlabel('Importance')
plt.title('Importance des 10 mots les plus fréquents dans l\'arbre de décision')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
