"""
Module de feature engineering pour le projet de prédiction de churn
Contient toutes les fonctions de parsing des champs JSON et création des variables dérivées
"""

import ast
import numpy as np
import pandas as pd
from config import FREQUENCY_MAPPING


# ============================================================
# FONCTIONS DE PARSING DES CHAMPS JSON
# ============================================================

def parse_payment_history(payment_str):
    """
    Parse le champ PaymentHistory et extrait le nombre total de paiements en retard

    Args:
        payment_str: String ou liste contenant l'historique des paiements

    Returns:
        int: Nombre total de paiements en retard
    """
    try:
        if isinstance(payment_str, str):
            parsed = ast.literal_eval(payment_str)
        else:
            parsed = payment_str

        if isinstance(parsed, list):
            return sum(d.get('Late_Payments', 0) for d in parsed)
        return 0
    except:
        return 0


def parse_service_interactions(service_str):
    """
    Parse le champ ServiceInteractions et compte le nombre d'interactions

    Args:
        service_str: String ou liste contenant les interactions de service

    Returns:
        int: Nombre total d'interactions
    """
    try:
        if isinstance(service_str, str):
            parsed = ast.literal_eval(service_str)
        else:
            parsed = service_str

        if isinstance(parsed, list):
            return len(parsed)
        return 0
    except:
        return 0


def parse_engagement_metrics(engagement_str):
    """
    Parse le champ EngagementMetrics et extrait le nombre de logins et la fréquence

    Args:
        engagement_str: String ou dict contenant les métriques d'engagement

    Returns:
        tuple: (nombre de logins, fréquence d'engagement)
    """
    try:
        if isinstance(engagement_str, str):
            parsed = ast.literal_eval(engagement_str)
        elif pd.notna(engagement_str):
            parsed = engagement_str
        else:
            return 0, 'Monthly'

        nb_logins = parsed.get('Logins', 0)
        frequency = parsed.get('Frequency', 'Monthly')
        return nb_logins, frequency
    except:
        return 0, 'Monthly'


def parse_feedback(feedback_str):
    """
    Parse le champ Feedback et extrait le score et le commentaire

    Args:
        feedback_str: String ou dict contenant le feedback

    Returns:
        tuple: (score, commentaire)
    """
    try:
        if isinstance(feedback_str, str):
            parsed = ast.literal_eval(feedback_str)
        elif pd.notna(feedback_str):
            parsed = feedback_str
        else:
            return None, None

        score = parsed.get('Rating', None)
        comment = parsed.get('Comment', None)
        return score, comment
    except:
        return None, None


def parse_website_usage(website_str):
    """
    Parse le champ WebsiteUsage et extrait les pages vues et le temps passé

    Args:
        website_str: String ou dict contenant l'usage du site web

    Returns:
        tuple: (nombre de pages vues, temps passé en minutes)
    """
    try:
        if isinstance(website_str, str):
            parsed = ast.literal_eval(website_str)
        elif pd.notna(website_str):
            parsed = website_str
        else:
            return 0, 0

        page_views = parsed.get('PageViews', 0)
        time_spent = parsed.get('TimeSpent(minutes)', 0)
        return page_views, time_spent
    except:
        return 0, 0


def parse_marketing_communication(marketing_str):
    """
    Parse le champ MarketingCommunication et calcule le taux d'ouverture d'emails
    et le taux de réponse

    Args:
        marketing_str: String ou liste contenant les communications marketing

    Returns:
        tuple: (taux d'ouverture en %, taux de réponse en %)
    """
    try:
        if isinstance(marketing_str, str):
            parsed = ast.literal_eval(marketing_str)
        else:
            parsed = marketing_str

        if isinstance(parsed, list) and len(parsed) > 0:
            total = len(parsed)

            # Taux d'ouverture
            opened = sum(1 for e in parsed if e.get('EmailOpened', 'No') == 'Yes' or e.get('Email_Opened', False))
            open_rate = round((opened / total) * 100, 2) if total > 0 else 0

            # Taux de réponse
            responded = sum(1 for e in parsed if e.get('Responded', 'No') == 'Yes')
            response_rate = round((responded / total) * 100, 2) if total > 0 else 0

            return open_rate, response_rate
        return 0, 0
    except:
        return 0, 0


def parse_purchase_history(purchase_str):
    """
    Parse le champ PurchaseHistory et calcule la fréquence et le montant moyen

    Args:
        purchase_str: String ou liste contenant l'historique d'achats

    Returns:
        tuple: (fréquence d'achats, montant moyen)
    """
    try:
        if isinstance(purchase_str, str):
            parsed = ast.literal_eval(purchase_str)
        elif pd.notna(purchase_str):
            parsed = purchase_str
        else:
            return 0, 0

        if isinstance(parsed, list):
            frequency = len(parsed)

            if frequency > 0:
                amounts = [item.get('Amount', 0) for item in parsed if 'Amount' in item]
                avg_amount = round(np.mean(amounts), 2) if amounts else 0
            else:
                avg_amount = 0

            return frequency, avg_amount
        return 0, 0
    except:
        return 0, 0


def parse_clickstream_data(clickstream_str):
    """
    Parse le champ ClickstreamData et calcule le taux de rebond

    Args:
        clickstream_str: String ou liste contenant les données de clickstream

    Returns:
        float: Taux de rebond en %
    """
    try:
        if isinstance(clickstream_str, str):
            parsed = ast.literal_eval(clickstream_str)
        else:
            parsed = clickstream_str

        if isinstance(parsed, list) and len(parsed) > 0:
            total = len(parsed)
            clicks_count = sum(1 for c in parsed if c.get('Action') == 'Click')
            return round((clicks_count / total) * 100, 2) if total > 0 else 0
        return 0
    except:
        return 0


# ============================================================
# FONCTIONS DE CRÉATION DES VARIABLES DÉRIVÉES
# ============================================================

def calculate_avg_logins_per_month(nb_logins, frequency):
    """
    Calcule la moyenne de logins par mois basée sur la fréquence d'engagement

    Args:
        nb_logins: Nombre total de logins
        frequency: Fréquence d'engagement ('Daily', 'Weekly', 'Monthly', 'Rarely')

    Returns:
        float: Moyenne de logins par mois
    """
    freq_value = FREQUENCY_MAPPING.get(frequency, 1)
    if freq_value > 0:
        return round(nb_logins / freq_value, 2)
    return 0


def calculate_subscription_duration(nb_logins):
    """
    Calcule une estimation de la durée d'abonnement basée sur l'activité

    Args:
        nb_logins: Nombre total de logins

    Returns:
        float: Durée d'abonnement estimée
    """
    return round(nb_logins * 2, 0)


# ============================================================
# FONCTION PRINCIPALE D'EXTRACTION DES FEATURES
# ============================================================

def extract_all_features(df):
    """
    Extrait toutes les features à partir des champs JSON bruts

    Args:
        df: DataFrame avec les colonnes JSON brutes

    Returns:
        DataFrame: DataFrame enrichi avec toutes les features extraites
    """
    df = df.copy()

    print("Extraction des features...")

    # 1. PaymentHistory
    if 'PaymentHistory' in df.columns:
        print("  ├─ Parsing PaymentHistory...")
        df['Parsed_PaymentHistory'] = df['PaymentHistory'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['Total_Late_Payments'] = df['Parsed_PaymentHistory'].apply(
            lambda l: sum(d.get('Late_Payments', 0) for d in l) if isinstance(l, list) else 0
        )

    # 2. ServiceInteractions
    if 'ServiceInteractions' in df.columns:
        print("  ├─ Parsing ServiceInteractions...")
        df['Parsed_ServiceInteractions'] = df['ServiceInteractions'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['Nb_ServiceInteractions'] = df['Parsed_ServiceInteractions'].apply(
            lambda l: len(l) if isinstance(l, list) else 0
        )

    # 3. EngagementMetrics
    if 'EngagementMetrics' in df.columns:
        print("  ├─ Parsing EngagementMetrics...")
        df['Parsed_EngagementMetrics'] = df['EngagementMetrics'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['Nb_Logins'] = df['Parsed_EngagementMetrics'].apply(
            lambda d: d.get('Logins', 0) if isinstance(d, dict) else 0
        )
        df['Freq_Engagement'] = df['Parsed_EngagementMetrics'].apply(
            lambda d: d.get('Frequency', 'Monthly') if isinstance(d, dict) else 'Monthly'
        )

        # Calcul de AvgLoginsPerMonth
        df['FreqEngagement'] = df['EngagementMetrics'].apply(
            lambda x: ast.literal_eval(x).get('Frequency', 'Monthly') if isinstance(x, str) 
            else x.get('Frequency', 'Monthly') if pd.notna(x) else 'Monthly'
        )
        df['AvgLoginsPerMonth'] = (df['Nb_Logins'] / df['FreqEngagement'].map(FREQUENCY_MAPPING)).round(2)

    # 4. Feedback
    if 'Feedback' in df.columns:
        print("  ├─ Parsing Feedback...")
        df['Parsed_Feedback'] = df['Feedback'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['Feedback_Score'] = df['Parsed_Feedback'].apply(
            lambda d: d.get('Rating', None) if isinstance(d, dict) else None
        )
        df['Feedback_Comment'] = df['Parsed_Feedback'].apply(
            lambda d: d.get('Comment', None) if isinstance(d, dict) else None
        )

    # 5. WebsiteUsage
    if 'WebsiteUsage' in df.columns:
        print("  ├─ Parsing WebsiteUsage...")
        df['Parsed_WebsiteUsage'] = df['WebsiteUsage'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['Nb_PageViews'] = df['Parsed_WebsiteUsage'].apply(
            lambda d: d.get('PageViews', 0) if isinstance(d, dict) else 0
        )
        df['TimeSpent_Minutes'] = df['Parsed_WebsiteUsage'].apply(
            lambda d: d.get('TimeSpent(minutes)', 0) if isinstance(d, dict) else 0
        )
        df['AvgSessionDuration'] = df['WebsiteUsage'].apply(
            lambda x: ast.literal_eval(x).get('TimeSpent(minutes)', 0) if isinstance(x, str) 
            else x.get('TimeSpent(minutes)', 0) if pd.notna(x) else 0
        )

    # 6. MarketingCommunication
    if 'MarketingCommunication' in df.columns:
        print("  ├─ Parsing MarketingCommunication...")
        df['Parsed_MarketingCommunication'] = df['MarketingCommunication'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['Nb_Emails_Opened'] = df['Parsed_MarketingCommunication'].apply(
            lambda l: sum(1 for d in l if d.get('Email_Opened', False)) if isinstance(l, list) else 0
        )

        # EmailOpenRate
        def extract_email_rate(marketing):
            try:
                emails = ast.literal_eval(marketing) if isinstance(marketing, str) else marketing
                if isinstance(emails, list) and len(emails) > 0:
                    total = len(emails)
                    opened = sum(1 for e in emails if e.get('EmailOpened', 'No') == 'Yes')
                    return round((opened / total) * 100, 2) if total > 0 else 0
                return 0
            except:
                return 0

        df['EmailOpenRate'] = df['MarketingCommunication'].apply(extract_email_rate)

        # ResponseRate
        def extract_response_rate(marketing):
            try:
                emails = ast.literal_eval(marketing) if isinstance(marketing, str) else marketing
                if isinstance(emails, list) and len(emails) > 0:
                    total = len(emails)
                    responded = sum(1 for e in emails if e.get('Responded', 'No') == 'Yes')
                    return round((responded / total) * 100, 2) if total > 0 else 0
                return 0
            except:
                return 0

        df['ResponseRate'] = df['MarketingCommunication'].apply(extract_response_rate)

    # 7. PurchaseHistory
    if 'PurchaseHistory' in df.columns:
        print("  ├─ Parsing PurchaseHistory...")
        df['PurchaseFrequency'] = df['PurchaseHistory'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) 
            else len(x) if pd.notna(x) else 0
        )

        def extract_avg_transaction(hist):
            try:
                purchases = ast.literal_eval(hist) if isinstance(hist, str) else hist
                if purchases:
                    amounts = [item.get('Amount', 0) for item in purchases if 'Amount' in item]
                    return round(np.mean(amounts), 2) if amounts else 0
                return 0
            except:
                return 0

        df['AvgTransactionAmount'] = df['PurchaseHistory'].apply(extract_avg_transaction)

    # 8. ClickstreamData
    if 'ClickstreamData' in df.columns:
        print("  ├─ Parsing ClickstreamData...")
        def extract_bounce_rate(clickstream):
            try:
                clicks = ast.literal_eval(clickstream) if isinstance(clickstream, str) else clickstream
                if isinstance(clicks, list) and len(clicks) > 0:
                    total = len(clicks)
                    clicks_count = sum(1 for c in clicks if c.get('Action') == 'Click')
                    return round((clicks_count / total) * 100, 2) if total > 0 else 0
                return 0
            except:
                return 0

        df['BounceRate'] = df['ClickstreamData'].apply(extract_bounce_rate)

    # 9. SubscriptionDuration
    if 'Nb_Logins' in df.columns:
        print("  ├─ Calcul SubscriptionDuration...")
        df['SubscriptionDuration'] = (df['Nb_Logins'] * 2).round(0)

    print("Extraction terminée")

    return df
