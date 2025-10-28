"""
Module utilitaire pour le chargement et l'utilisation du modèle de prédiction
"""

import joblib
import pandas as pd
import numpy as np
from config import MODEL_PATH, SCALER_PATH, FEATURES_PATH
from preprocessing import preprocess_for_prediction


def load_model():
    """
    Charge le modèle de prédiction entraîné

    Returns:
        model: Modèle chargé ou None en cas d'erreur
    """
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modèle chargé: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None


def load_scaler():
    """
    Charge le StandardScaler entraîné

    Returns:
        scaler: Scaler chargé ou None en cas d'erreur
    """
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler chargé: {SCALER_PATH}")
        return scaler
    except Exception as e:
        print(f"Erreur lors du chargement du scaler: {e}")
        return None


def load_feature_names():
    """
    Charge les noms des features attendues par le modèle

    Returns:
        list: Liste des noms de features ou None en cas d'erreur
    """
    try:
        features = joblib.load(FEATURES_PATH)
        print(f"Features chargées: {len(features)} features")
        return features
    except Exception as e:
        print(f"Erreur lors du chargement des features: {e}")
        return None


def load_all_artifacts():
    """
    Charge tous les artifacts nécessaires (modèle, scaler, features)

    Returns:
        tuple: (model, scaler, features) ou (None, None, None) en cas d'erreur
    """
    print("Chargement des artifacts du modèle...")
    print("="*60)

    model = load_model()
    scaler = load_scaler()
    features = load_feature_names()

    if model is None or scaler is None or features is None:
        print("Échec du chargement des artifacts")
        return None, None, None

    print("="*60)
    print(f"Tous les artifacts chargés avec succès")
    print(f"├─ Modèle: {type(model).__name__}")
    print(f"├─ Scaler: {type(scaler).__name__}")
    print(f"└─ Features: {len(features)} features attendues")

    return model, scaler, features


def predict_churn(df, model, scaler):
    """
    Effectue la prédiction de churn sur un DataFrame

    Args:
        df: DataFrame avec les données brutes
        model: Modèle entraîné
        scaler: StandardScaler entraîné

    Returns:
        tuple: (predictions, probabilities, df_processed)
            - predictions: Array avec les prédictions (0 ou 1)
            - probabilities: Array avec les probabilités de churn
            - df_processed: DataFrame avec les features prétraitées
    """
    try:
        print("🔮 Début de la prédiction...")
        print("="*60)

        # Prétraitement des données
        X = preprocess_for_prediction(df, scaler)

        # Vérification
        if X.shape[0] == 0:
            print("Aucune donnée après preprocessing")
            return None, None, None

        # Prédictions
        print("🔮 Calcul des prédictions...")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        print("="*60)
        print(f"Prédictions effectuées pour {len(predictions)} clients")
        print(f"├─ Churn prédit: {predictions.sum()} clients ({predictions.mean()*100:.1f}%)")
        print(f"└─ Probabilité moyenne: {probabilities.mean():.3f}")

        return predictions, probabilities, X

    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def add_predictions_to_dataframe(df_original, predictions, probabilities):
    """
    Ajoute les prédictions et probabilités au DataFrame original

    Args:
        df_original: DataFrame original
        predictions: Array des prédictions
        probabilities: Array des probabilités

    Returns:
        DataFrame: DataFrame enrichi avec les prédictions
    """
    df = df_original.copy()

    # Ajouter les prédictions
    df['Prediction'] = predictions
    df['Probability_Churn'] = probabilities

    # Ajouter le niveau de risque
    df['Risk_Level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Faible', 'Moyen', 'Élevé']
    )

    print(f"Prédictions ajoutées au DataFrame")

    return df


def get_prediction_summary(predictions, probabilities):
    """
    Génère un résumé des prédictions

    Args:
        predictions: Array des prédictions
        probabilities: Array des probabilités

    Returns:
        dict: Dictionnaire avec les statistiques de prédiction
    """
    summary = {
        'total_clients': len(predictions),
        'churn_predicted': int(predictions.sum()),
        'churn_percentage': round(predictions.mean() * 100, 2),
        'avg_probability': round(probabilities.mean(), 3),
        'high_risk_count': int((probabilities > 0.7).sum()),
        'medium_risk_count': int(((probabilities > 0.3) & (probabilities <= 0.7)).sum()),
        'low_risk_count': int((probabilities <= 0.3).sum())
    }

    return summary


def get_high_risk_clients(df_with_predictions, top_n=10):
    """
    Extrait les clients à haut risque

    Args:
        df_with_predictions: DataFrame avec les prédictions
        top_n: Nombre de clients à retourner (par défaut 10)

    Returns:
        DataFrame: Top N clients à risque
    """
    # Filtrer les clients avec prédiction de churn
    high_risk = df_with_predictions[df_with_predictions['Prediction'] == 1].copy()

    # Trier par probabilité décroissante
    high_risk = high_risk.sort_values('Probability_Churn', ascending=False)

    # Retourner le top N
    return high_risk.head(top_n)


def calculate_accuracy(predictions, actual_labels):
    """
    Calcule l'accuracy si les vraies étiquettes sont disponibles

    Args:
        predictions: Prédictions du modèle
        actual_labels: Vraies étiquettes

    Returns:
        float: Accuracy en pourcentage
    """
    if actual_labels is None or len(actual_labels) == 0:
        return None

    accuracy = (predictions == actual_labels).sum() / len(predictions) * 100
    return round(accuracy, 2)


def save_predictions_to_csv(df_with_predictions, filename='predictions.csv'):
    """
    Sauvegarde les prédictions dans un fichier CSV

    Args:
        df_with_predictions: DataFrame avec les prédictions
        filename: Nom du fichier de sortie

    Returns:
        bool: True si succès, False sinon
    """
    try:
        df_with_predictions.to_csv(filename, index=False)
        print(f"Prédictions sauvegardées dans {filename}")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        return False
