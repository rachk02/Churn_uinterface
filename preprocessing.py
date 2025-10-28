"""
Module de prétraitement pour le projet de prédiction de churn
Contient le pipeline complet de transformation des données
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_engineering import extract_all_features
from config import MODEL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, COLUMNS_TO_DROP


def encode_categorical_features(df):
    """
    Encode les variables catégorielles avec One-Hot Encoding

    Args:
        df: DataFrame avec les colonnes catégorielles

    Returns:
        DataFrame: DataFrame avec les colonnes encodées
    """
    df = df.copy()

    print("Encodage des catégories...")

    # Encodage de Gender
    if 'Gender' in df.columns:
        print("  ├─ Encodage Gender...")
        gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender', drop_first=False)
        df = pd.concat([df, gender_dummies], axis=1)

    # Encodage de Segment
    if 'Segment' in df.columns:
        print("  ├─ Encodage Segment...")
        segment_dummies = pd.get_dummies(df['Segment'], prefix='Segment', drop_first=False)
        df = pd.concat([df, segment_dummies], axis=1)

    print("Encodage terminé")

    return df


def normalize_features(df, scaler=None, fit=False):
    """
    Normalise les features numériques avec StandardScaler

    Args:
        df: DataFrame avec les features numériques
        scaler: StandardScaler pré-entraîné (None pour en créer un nouveau)
        fit: Si True, fit le scaler sur les données (training), sinon transform seulement (prediction)

    Returns:
        tuple: (DataFrame normalisé, scaler)
    """
    df = df.copy()

    print("Normalisation des features...")

    # Identifier les features numériques présentes
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]

    if len(numerical_cols) == 0:
        print("  Aucune feature numérique trouvée")
        return df, scaler

    print(f"  ├─ {len(numerical_cols)} features à normaliser")

    # Créer un scaler si nécessaire
    if scaler is None:
        scaler = StandardScaler()

    # Fit et transform ou seulement transform
    if fit:
        print("  ├─ Fit + Transform")
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        print("  ├─ Transform seulement")
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    print("Normalisation terminée")

    return df, scaler


def clean_dataframe(df):
    """
    Supprime les colonnes inutiles après le preprocessing

    Args:
        df: DataFrame à nettoyer

    Returns:
        DataFrame: DataFrame nettoyé
    """
    df = df.copy()

    print("🧹 Nettoyage du DataFrame...")

    # Supprimer les colonnes qui existent dans le DataFrame
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]

    if cols_to_drop:
        print(f"  ├─ Suppression de {len(cols_to_drop)} colonnes")
        df = df.drop(columns=cols_to_drop, errors='ignore')

    print("Nettoyage terminé")

    return df


def select_model_features(df, required_features=None):
    """
    Sélectionne uniquement les features nécessaires pour le modèle

    Args:
        df: DataFrame avec toutes les features
        required_features: Liste des features requises (par défaut MODEL_FEATURES)

    Returns:
        DataFrame: DataFrame avec uniquement les features sélectionnées
    """
    if required_features is None:
        required_features = MODEL_FEATURES

    print(f"Sélection des features pour le modèle...")

    # Vérifier quelles features sont disponibles
    available_features = [f for f in required_features if f in df.columns]
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        print(f"  Features manquantes: {missing_features}")

    print(f"  ├─ {len(available_features)}/{len(required_features)} features disponibles")

    return df[available_features].copy()


def preprocess_for_training(df):
    """
    Pipeline complet de prétraitement pour l'entraînement du modèle

    Args:
        df: DataFrame brut avec les données originales

    Returns:
        tuple: (X_processed, y, scaler)
            - X_processed: Features prétraitées
            - y: Variable cible (si présente)
            - scaler: StandardScaler entraîné
    """
    print("Début du preprocessing pour entraînement...")
    print("="*60)

    # 1. Extraction des features depuis les champs JSON
    df = extract_all_features(df)

    # 2. Encodage des variables catégorielles
    df = encode_categorical_features(df)

    # 3. Normalisation des features numériques (avec fit)
    df, scaler = normalize_features(df, scaler=None, fit=True)

    # 4. Nettoyage du DataFrame
    df = clean_dataframe(df)

    # 5. Extraction de la variable cible (si présente)
    y = None
    if 'ChurnLabel' in df.columns:
        y = df['ChurnLabel'].copy()
        df = df.drop(columns=['ChurnLabel'])

    # 6. Sélection des features finales
    X = select_model_features(df)

    print("="*60)
    print(f"Preprocessing terminé: {X.shape[0]} lignes × {X.shape[1]} features")

    return X, y, scaler


def preprocess_for_prediction(df, scaler):
    """
    Pipeline complet de prétraitement pour la prédiction

    Args:
        df: DataFrame brut avec les données à prédire
        scaler: StandardScaler pré-entraîné

    Returns:
        DataFrame: Features prétraitées prêtes pour la prédiction
    """
    print("Début du preprocessing pour prédiction...")
    print("="*60)

    # 1. Extraction des features depuis les champs JSON
    df = extract_all_features(df)

    # 2. Encodage des variables catégorielles
    df = encode_categorical_features(df)

    # 3. Normalisation des features numériques (sans fit, juste transform)
    df, _ = normalize_features(df, scaler=scaler, fit=False)

    # 4. Nettoyage du DataFrame (garder les colonnes utiles pour l'affichage)
    # On ne nettoie pas complètement pour garder les infos clients

    # 5. Sélection des features finales pour le modèle
    X = select_model_features(df)

    print("="*60)
    print(f"Preprocessing terminé: {X.shape[0]} lignes × {X.shape[1]} features")

    return X


def validate_preprocessing(df_original, df_processed):
    """
    Valide que le preprocessing s'est correctement déroulé

    Args:
        df_original: DataFrame original avant preprocessing
        df_processed: DataFrame après preprocessing

    Returns:
        dict: Dictionnaire avec les résultats de validation
    """
    validation = {
        'original_shape': df_original.shape,
        'processed_shape': df_processed.shape,
        'missing_values': df_processed.isnull().sum().sum(),
        'features_created': list(df_processed.columns),
        'success': True
    }

    # Vérifier qu'on a toutes les features nécessaires
    missing_features = [f for f in MODEL_FEATURES if f not in df_processed.columns]
    if missing_features:
        validation['success'] = False
        validation['missing_features'] = missing_features

    # Vérifier qu'il n'y a pas de valeurs manquantes critiques
    if validation['missing_values'] > 0:
        validation['warning'] = f"{validation['missing_values']} valeurs manquantes détectées"

    return validation


def get_feature_importance_data(df):
    """
    Prépare les données pour afficher l'importance des features

    Args:
        df: DataFrame avec les features

    Returns:
        DataFrame: Statistiques sur les features
    """
    stats = pd.DataFrame({
        'Feature': df.columns,
        'Mean': df.mean(),
        'Std': df.std(),
        'Min': df.min(),
        'Max': df.max(),
        'Missing': df.isnull().sum()
    })

    return stats.reset_index(drop=True)
