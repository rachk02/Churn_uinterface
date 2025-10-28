"""
Configuration et constantes pour le projet de prédiction de churn
"""

# ============================================================
# CHEMINS DES MODÈLES
# ============================================================
MODEL_PATH = 'models/logistic_regression_churn_model.pkl'
SCALER_PATH = 'models/scaler_standardscaler.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

# ============================================================
# MAPPING DE FRÉQUENCE D'ENGAGEMENT
# ============================================================
# Conversion de la fréquence d'engagement en nombre de logins par mois
FREQUENCY_MAPPING = {
    'Daily': 30,      # 30 fois/mois
    'Weekly': 4,      # 4 fois/mois
    'Monthly': 1,     # 1 fois/mois
    'Rarely': 0.25    # 0.25 fois/mois
}

# ============================================================
# FEATURES FINALES POUR LE MODÈLE
# ============================================================
# Les 7 features que le modèle attend (dans cet ordre)
MODEL_FEATURES = [
    'Total_Late_Payments',      # Retards de paiement
    'Nb_ServiceInteractions',   # Interactions support
    'NPS',                       # Net Promoter Score
    'AvgLoginsPerMonth',        # Engagement utilisateur
    'Nb_PageViews',             # Engagement web
    'Feedback_Score',           # Satisfaction client
    'EmailOpenRate'             # Engagement marketing
]

# ============================================================
# FEATURES NUMÉRIQUES À NORMALISER
# ============================================================
NUMERICAL_FEATURES = [
    'Age',
    'NPS',
    'Total_Late_Payments',
    'Nb_ServiceInteractions',
    'AvgLoginsPerMonth',
    'Feedback_Score',
    'Nb_PageViews',
    'SubscriptionDuration',
    'PurchaseFrequency',
    'AvgTransactionAmount',
    'EmailOpenRate',
    'ResponseRate',
    'AvgSessionDuration',
    'BounceRate'
]

# ============================================================
# COLONNES CATÉGORIELLES À ENCODER
# ============================================================
CATEGORICAL_FEATURES = {
    'Gender': ['Male', 'Female'],
    'Segment': ['A', 'B', 'C']
}

# ============================================================
# COLONNES À SUPPRIMER APRÈS PREPROCESSING
# ============================================================
COLUMNS_TO_DROP = [
    # Variables d'identification
    'CustomerID', 'Name', 'Email', 'Phone', 'Address', 'Timestamp',
    
    # Variables redondantes (déjà encodées)
    'Gender', 'Segment', 'Location',
    
    # Variables JSON originales
    'PurchaseHistory', 'SubscriptionDetails', 'ServiceInteractions',
    'PaymentHistory', 'WebsiteUsage', 'ClickstreamData',
    'EngagementMetrics', 'Feedback', 'MarketingCommunication',
    
    # Variables parsées intermédiaires
    'Parsed_PaymentHistory', 'Parsed_ServiceInteractions',
    'Parsed_EngagementMetrics', 'Parsed_Feedback',
    'Parsed_WebsiteUsage', 'Parsed_MarketingCommunication',
    
    # Variables intermédiaires de calcul
    'Feedback_Comment', 'Freq_Engagement', 'FreqEngagement',
    'Nb_Logins', 'TimeSpent_Minutes', 'Nb_Emails_Opened'
]
