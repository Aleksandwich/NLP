import os
import glob
import re
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from time import time

# Configuration
DATA_DIR = 'imdb_smol'
RANDOM_STATE = 42

def load_data(data_dir):
    """Charge les critiques de films depuis les dossiers positif et négatif"""
    print("Chargement des données...")
    
    # Chemins vers les dossiers positif et négatif
    pos_dir = os.path.join(data_dir, 'pos')
    neg_dir = os.path.join(data_dir, 'neg')
    
    # Liste des fichiers
    pos_files = glob.glob(os.path.join(pos_dir, '*.txt'))
    neg_files = glob.glob(os.path.join(neg_dir, '*.txt'))
    
    # Lecture des fichiers
    pos_reviews = []
    for file in pos_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            pos_reviews.append(f.read())
    
    neg_reviews = []
    for file in neg_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            neg_reviews.append(f.read())
    
    # Création du dataset
    reviews = pos_reviews + neg_reviews
    labels = [1] * len(pos_reviews) + [0] * len(neg_reviews)
    
    print(f"Nombre total de critiques: {len(reviews)}")
    print(f"Critiques positives: {len(pos_reviews)}")
    print(f"Critiques négatives: {len(neg_reviews)}")
    
    return reviews, labels

def preprocess_text(text):
    """Prétraitement basique du texte"""
    # Conversion en minuscules
    text = text.lower()
    # Suppression des balises HTML
    text = re.sub(r'<.*?>', '', text)
    # Suppression des caractères spéciaux et chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def vectorize_data(reviews, vectorizer_type='count', ngram_range=(1, 1), max_features=None):
    """Vectorise le texte selon la méthode choisie"""
    print(f"Vectorisation avec {vectorizer_type}...")
    
    # Prétraitement des textes
    preprocessed_reviews = [preprocess_text(review) for review in reviews]
    
    # Choix du vectoriseur
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    else:
        raise ValueError("Type de vectoriseur non reconnu")
    
    # Vectorisation
    X = vectorizer.fit_transform(preprocessed_reviews)
    
    print(f"Dimensions des données vectorisées: {X.shape}")
    return X, vectorizer

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, params=None):
    """Entraîne un modèle et évalue ses performances"""
    print(f"\nEntraînement du modèle {model_name}...")
    
    # Recherche des meilleurs hyperparamètres si params n'est pas None
    if params:
        print("Optimisation des hyperparamètres...")
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        start_time = time()
        grid_search.fit(X_train, y_train)
        training_time = time() - start_time
        
        # Récupération du meilleur modèle
        model = grid_search.best_estimator_
        print(f"Meilleurs paramètres: {grid_search.best_params_}")
    else:
        # Entraînement direct du modèle
        start_time = time()
        model.fit(X_train, y_train)
        training_time = time() - start_time
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Évaluation
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Temps d'entraînement: {training_time:.2f} secondes")
    print(f"Précision: {accuracy:.4f}")
    print("Rapport de classification:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'model': model,
        'accuracy': accuracy,
        'training_time': training_time
    }

def main():
    """Fonction principale"""
    # Chargement des données
    reviews, labels = load_data(DATA_DIR)
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Visualisation de la répartition des classes
    plt.figure(figsize=(8, 6))
    plt.hist([y_train, y_test], bins=2, label=['Train', 'Test'])
    plt.xticks([0.25, 0.75], ['Négatif', 'Positif'])
    plt.legend()
    plt.title("Répartition des classes d'entraînement et de test")
    plt.savefig("class_distribution.png")
    
    # Liste pour stocker les résultats
    results = []
    
    # Test des différentes combinaisons de vectorisation et de classifieurs
    for vectorizer_type in ['count', 'tfidf']:
        for ngram_range in [(1, 1), (1, 2)]:
            # Vectorisation
            X_train_vec, vectorizer = vectorize_data(
                X_train, 
                vectorizer_type=vectorizer_type, 
                ngram_range=ngram_range, 
                max_features=5000
            )
            X_test_vec = vectorizer.transform(X_test)
            
            # Configuration des modèles et leurs hyperparamètres
            models = [
                {
                    'name': 'SVM',
                    'model': LinearSVC(random_state=RANDOM_STATE, dual="auto"),
                    'params': {'C': [0.1, 1.0, 10.0]}
                },
                {
                    'name': 'Régression Logistique',
                    'model': LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'),
                    'params': {'C': [0.1, 1.0, 10.0]}
                },
                {
                    'name': 'Arbre de Décision',
                    'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
                    'params': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
                }
            ]
            
            # Entraînement et évaluation des modèles
            for model_config in models:
                result = train_and_evaluate(
                    X_train_vec, 
                    X_test_vec, 
                    y_train, 
                    y_test, 
                    f"{model_config['name']} ({vectorizer_type}, {ngram_range})", 
                    model_config['model'], 
                    model_config['params']
                )
                results.append(result)
    
    # Affichage et visualisation des résultats comparatifs
    print("\nRésultats comparatifs:")
    results_df = pd.DataFrame([
        {
            'Modèle': res['model_name'],
            'Précision': res['accuracy'],
            'Temps d\'entraînement (s)': res['training_time']
        }
        for res in results
    ])
    
    print(results_df.sort_values('Précision', ascending=False))
    
    # Visualisation des précisions
    plt.figure(figsize=(14, 8))
    bars = plt.bar(results_df['Modèle'], results_df['Précision'])
    plt.xticks(rotation=90)
    plt.ylim(0.7, 1.0)  # Ajusté pour mieux voir les différences
    plt.title('Comparaison des précisions des modèles')
    plt.ylabel('Précision')
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    
    # Identifie le meilleur modèle
    best_idx = results_df['Précision'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Modèle']
    best_model = next(res['model'] for res in results if res['model_name'] == best_model_name)
    
    print(f"\nLe meilleur modèle est: {best_model_name}")
    print(f"Avec une précision de: {results_df.loc[best_idx, 'Précision']:.4f}")

if __name__ == "__main__":
    main()



