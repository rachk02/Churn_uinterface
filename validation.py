"""
Module de validation des données et des transformations
"""
import pandas as pd
import numpy as np
from config import MODEL_FEATURES


def validate_raw_data(df):
    """Valide les données brutes"""
    errors = []
    warnings = []

    # Vérifier les colonnes requises
    required_cols = ['CustomerID', 'Age', 'Gender', 'Segment', 'NPS', 
                     'PaymentHistory', 'ServiceInteractions', 'EngagementMetrics',
                     'Feedback', 'WebsiteUsage', 'MarketingCommunication',
                     'PurchaseHistory', 'ClickstreamData']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Colonnes manquantes: {missing_cols}")

    # Vérifier les doublons
    if 'CustomerID' in df.columns:
        duplicates = df.duplicated(subset=['CustomerID']).sum()
        if duplicates > 0:
            warnings.append(f"{duplicates} CustomerID dupliqués détectés")

    # Vérifier les valeurs manquantes critiques
    critical_cols = ['CustomerID', 'ChurnLabel']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"{null_count} valeurs manquantes dans {col}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def validate_parsed_data(df_parsed, df_raw):
    """Valide les données après parsing"""
    errors = []
    warnings = []

    # Vérifier que de nouvelles colonnes ont été créées
    new_cols = [c for c in df_parsed.columns if c not in df_raw.columns]
    if len(new_cols) < 10:
        warnings.append(f"Seulement {len(new_cols)} nouvelles colonnes créées (attendu: 12+)")

    # Vérifier les features attendues
    expected_features = ['Total_Late_Payments', 'Nb_ServiceInteractions', 
                        'AvgLoginsPerMonth', 'Feedback_Score', 'Nb_PageViews',
                        'EmailOpenRate']

    missing_features = [f for f in expected_features if f not in df_parsed.columns]
    if missing_features:
        errors.append(f"Features manquantes: {missing_features}")

    # Vérifier les valeurs aberrantes
    numeric_cols = df_parsed.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col in df_parsed.columns:
            # Valeurs infinies
            inf_count = np.isinf(df_parsed[col]).sum()
            if inf_count > 0:
                errors.append(f"{inf_count} valeurs infinies dans {col}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def validate_normalized_data(df_normalized):
    """Valide les données après normalisation"""
    errors = []
    warnings = []

    # Vérifier que la normalisation a été appliquée
    numeric_cols = df_normalized.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        mean = df_normalized[col].mean()
        std = df_normalized[col].std()

        # La moyenne devrait être proche de 0
        if abs(mean) > 0.1:
            warnings.append(f"{col}: moyenne = {mean:.4f} (devrait être ~0)")

        # L'écart-type devrait être proche de 1
        if abs(std - 1) > 0.2:
            warnings.append(f"{col}: std = {std:.4f} (devrait être ~1)")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def validate_model_input(X):
    """Valide les données avant prédiction"""
    errors = []
    warnings = []

    # Vérifier le nombre de features
    if X.shape[1] != len(MODEL_FEATURES):
        errors.append(f"Nombre de features incorrect: {X.shape[1]} (attendu: {len(MODEL_FEATURES)})")

    # Vérifier les noms de colonnes
    missing_cols = [col for col in MODEL_FEATURES if col not in X.columns]
    if missing_cols:
        errors.append(f"Features manquantes: {missing_cols}")

    # Vérifier les valeurs manquantes
    null_count = X.isnull().sum().sum()
    if null_count > 0:
        errors.append(f"{null_count} valeurs manquantes dans les features")

    # Vérifier les valeurs infinies
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        errors.append(f"{inf_count} valeurs infinies dans les features")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
