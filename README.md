# Plateforme de Prédiction de Churn Client

## Description du Projet

Cette application Streamlit implémente un pipeline complet de machine learning pour la prédiction du churn client. Elle permet d'analyser les données clients, d'effectuer un prétraitement transparent avec visualisation des étapes intermédiaires, et de prédire les clients à risque de départ.

### Fonctionnalités principales

- **Chargement de données** : Support des formats CSV et Excel (XLSX/XLS)
- **Exploration des données (EDA)** : Analyse statistique complète avec 5 modules d'analyse
- **Prétraitement transparent** : Visualisation de chaque étape de transformation des données
- **Prédiction ML** : Modèle de régression logistique pour prédire le churn
- **Dashboard analytique** : Visualisations interactives des résultats
- **Export des données** : Téléchargement des résultats et rapports

---

## Architecture du Projet

```
churn-prediction-app/
│
├── app.py                          # Application principale (page d'accueil)
│
├── pages/                          # Pages Streamlit multi-pages
│   ├── 01_Chargement.py               # Page d'upload des données
│   ├── 02_EDA.py                  # Page d'exploration des données
│   ├── 03_Prétraitement.py        # Page de prétraitement
│   ├── 04_Prediction.py           # Page de prédiction
│   ├── 05_Dashboard.py            # Dashboard analytique
│   └── 06_Export.py               # Page d'export
│
├── config.py                       # Configuration et constantes
├── feature_engineering.py          # Fonctions de parsing JSON et feature engineering
├── preprocessing.py                # Pipeline de prétraitement des données
├── model_utils.py                  # Utilitaires de chargement et prédiction du modèle
├── validation.py                   # Fonctions de validation des données
├── logger.py                       # Système de logging
│
├── models/                         # Modèles ML pré-entraînés
│   ├── logistic_regression_churn_model.pkl
│   ├── scaler_standardscaler.pkl
│   └── feature_names.pkl
│
├── logs/                           # Fichiers de logs (créé automatiquement)
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

---

## Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Git

---

## Installation

### Sur Windows

#### 1. Cloner le repository

Ouvrez PowerShell ou Command Prompt et exécutez :

```bash
git clone https://github.com/rachk02/Churn_uinterface.git
cd Churn_uinterface
```

#### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

#### 3. Activer l'environnement virtuel

```bash
venv\Scripts\activate
```

#### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`

---

### Sur macOS

#### 1. Cloner le repository

Ouvrez Terminal et exécutez :

```bash
git clone https://github.com/rachk02/Churn_uinterface.git
cd Churn_uinterface
```

#### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
```

#### 3. Activer l'environnement virtuel

```bash
source venv/bin/activate
```

#### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`

---

### Sur Linux (Ubuntu/Debian)

#### 1. Installer Python et pip (si nécessaire)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

#### 2. Cloner le repository

```bash
git clone https://github.com/rachk02/Churn_uinterface.git
cd Churn_uinterface
```

#### 3. Créer un environnement virtuel

```bash
python3 -m venv venv
```

#### 4. Activer l'environnement virtuel

```bash
source venv/bin/activate
```

#### 5. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 6. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`

---

## Configuration des Modèles

Les modèles pré-entraînés doivent être placés dans le dossier `models/`. Si vous n'avez pas encore de modèles, assurez-vous d'avoir les trois fichiers suivants :

- `logistic_regression_churn_model.pkl` : Modèle de régression logistique
- `scaler_standardscaler.pkl` : StandardScaler pour la normalisation
- `feature_names.pkl` : Noms des features attendues par le modèle

---

## Utilisation de l'Application

### 1. Page d'Accueil

La page d'accueil présente :
- Un aperçu des fonctionnalités
- Le statut du pipeline (étapes complétées)
- Les données actuellement en mémoire
- Des instructions pour commencer

### 2. Chargement des Données

**Navigation** : Cliquez sur "Chargement" dans la sidebar

**Fonctionnalités** :
- Chargement de fichiers CSV ou Excel (XLSX/XLS)
- Validation automatique des données
- Affichage des statistiques du fichier
- Aperçu des données brutes
- Sauvegarde en mémoire pour les étapes suivantes

**Format attendu** :
Le fichier doit contenir au minimum les colonnes suivantes :
- `CustomerID` : Identifiant unique du client
- `Age`, `Gender`, `Segment`, `NPS` : Informations client
- `PaymentHistory`, `ServiceInteractions`, `EngagementMetrics` : Données JSON
- `Feedback`, `WebsiteUsage`, `MarketingCommunication` : Données JSON
- `PurchaseHistory`, `ClickstreamData` : Données JSON

### 3. Exploration des Données (EDA)

**Navigation** : Cliquez sur "EDA" dans la sidebar

**Fonctionnalités** :

#### Onglet 1 : Vue d'ensemble
- Métriques globales (lignes, colonnes, valeurs manquantes)
- Distribution des types de données
- Statistiques descriptives
- Analyse des valeurs manquantes

#### Onglet 2 : Analyse univariée
- Sélection de variable individuelle
- Pour variables numériques :
  - Histogrammes avec distribution marginale
  - Boxplots avec détection d'outliers
  - Statistiques détaillées (moyenne, médiane, écart-type)
- Pour variables catégorielles :
  - Diagrammes en barres
  - Graphiques circulaires
  - Tableaux de fréquences

#### Onglet 3 : Analyse bivariée
- Comparaison des variables
- Pour variables numériques :
  - Boxplots par groupe
  - Violin plots
  - Tests statistiques (t-test)
  - Statistiques par groupe
- Pour variables catégorielles :
  - Tableaux croisés
  - Graphiques groupés

### 4. Prétraitement

**Navigation** : Cliquez sur "Prétraitement" dans la sidebar

Le preprocessing s'effectue en 4 étapes séquentielles avec sauvegarde de DataFrames intermédiaires :

#### Étape 1 : Parsing JSON
- Parse 8 champs JSON (PaymentHistory, ServiceInteractions, etc.)
- Crée 12+ variables dérivées
- Sauvegarde : `df_parsed`

**Variables créées** :
- `Total_Late_Payments` : Nombre total de retards de paiement
- `Nb_ServiceInteractions` : Nombre d'interactions avec le support
- `AvgLoginsPerMonth` : Moyenne de logins par mois
- `Feedback_Score` : Score de satisfaction client
- `Nb_PageViews` : Nombre de pages vues
- `EmailOpenRate` : Taux d'ouverture des emails (%)
- `ResponseRate` : Taux de réponse (%)
- `BounceRate` : Taux de rebond (%)
- `SubscriptionDuration` : Durée estimée d'abonnement
- `PurchaseFrequency` : Fréquence d'achat
- `AvgTransactionAmount` : Montant moyen des transactions
- `AvgSessionDuration` : Durée moyenne de session

#### Étape 2 : Encodage One-Hot
- Encode les variables catégorielles
- Gender : 2 colonnes binaires
- Segment : 3 colonnes binaires
- Sauvegarde : `df_encoded`

#### Étape 3 : Normalisation
- Applique StandardScaler sur 14 features numériques
- Transformation : mean = 0, std = 1
- Utilise le scaler pré-entraîné
- Sauvegarde : `df_normalized`

#### Étape 4 : Dataset Final
- Sélection des 7 features finales pour le modèle
- Ordre respecté selon la configuration
- Sauvegarde : `df_model`

**Features finales** :
1. Total_Late_Payments
2. Nb_ServiceInteractions
3. NPS
4. AvgLoginsPerMonth
5. Nb_PageViews
6. Feedback_Score
7. EmailOpenRate

### 5. Prédiction

**Navigation** : Cliquez sur "Prediction" dans la sidebar

**Prérequis** : Le preprocessing doit être complété

**Fonctionnalités** :
- Chargement automatique du modèle
- Prédiction en batch sur toutes les données
- Calcul des probabilités de churn
- Classification en niveaux de risque :
  - Faible : probabilité < 30%
  - Moyen : probabilité 30-70%
  - Élevé : probabilité > 70%

**Résultats affichés** :
- Métriques globales (total clients, taux de churn, distribution des risques)
- Top 10 clients à risque élevé
- Graphiques de distribution
- Filtres par niveau de risque
- Export CSV des prédictions

### 6. Dashboard Analytique

**Navigation** : Cliquez sur "Dashboard" dans la sidebar

**Prérequis** : Les prédictions doivent être effectuées

**Analyses disponibles** :
- KPIs globaux (taux de churn, probabilité moyenne, accuracy si ChurnLabel présent)
- Distribution des probabilités (histogramme et boxplot)
- Analyse par segment (taux de churn, graphiques)
- Distribution des niveaux de risque (pie chart et bar chart)
- Analyse des features (scatter plots NPS vs probabilité, Age vs probabilité)
- Matrice de confusion (si ChurnLabel présent)
- Métriques de performance (accuracy, precision, recall)

### 7. Export

**Navigation** : Cliquez sur "Export" dans la sidebar

**Exports disponibles** :

#### DataFrames intermédiaires
- Données brutes
- Après parsing
- Après encodage
- Après normalisation
- Dataset final
- Avec prédictions

#### Rapports
- Rapport textuel de synthèse
- Export par catégorie de risque (Faible/Moyen/Élevé)

#### Statistiques d'export
- Nombre de DataFrames disponibles
- Taille des données
- Métriques de mémoire

---

## Modules Python

### config.py
Contient toutes les constantes et configurations :
- Chemins des modèles
- Mapping de fréquence d'engagement
- Liste des features du modèle
- Features à normaliser
- Colonnes à supprimer

### feature_engineering.py
Fonctions de feature engineering :
- `parse_payment_history()` : Parse l'historique de paiement
- `parse_service_interactions()` : Parse les interactions support
- `parse_engagement_metrics()` : Parse les métriques d'engagement
- `parse_feedback()` : Parse les feedbacks clients
- `parse_website_usage()` : Parse l'usage du site web
- `parse_marketing_communication()` : Parse les communications marketing
- `parse_purchase_history()` : Parse l'historique d'achats
- `parse_clickstream_data()` : Parse les données de navigation
- `extract_all_features()` : Fonction principale d'extraction

### preprocessing.py
Pipeline de prétraitement :
- `encode_categorical_features()` : Encodage One-Hot
- `normalize_features()` : Normalisation StandardScaler
- `clean_dataframe()` : Nettoyage des colonnes
- `select_model_features()` : Sélection des features finales
- `preprocess_for_training()` : Pipeline complet pour entraînement
- `preprocess_for_prediction()` : Pipeline complet pour prédiction

### model_utils.py
Gestion du modèle :
- `load_model()` : Charge le modèle pré-entraîné
- `load_scaler()` : Charge le StandardScaler
- `load_feature_names()` : Charge les noms des features
- `load_all_artifacts()` : Charge tous les artifacts
- `predict_churn()` : Effectue les prédictions
- `add_predictions_to_dataframe()` : Ajoute les prédictions au DataFrame
- `get_prediction_summary()` : Génère un résumé des prédictions
- `get_high_risk_clients()` : Extrait les clients à haut risque

### validation.py
Validation des données :
- `validate_raw_data()` : Valide les données brutes
- `validate_parsed_data()` : Valide après parsing
- `validate_normalized_data()` : Valide après normalisation
- `validate_model_input()` : Valide avant prédiction

### logger.py
Système de logging :
- Configuration du logger
- Logs dans fichiers quotidiens
- Niveaux : DEBUG, INFO, WARNING, ERROR

---

## Dépendances

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
scipy>=1.11.0
openpyxl>=3.0.0
joblib>=1.3.0
```

---

## Résolution de Problèmes

### Problème : Les pages ne s'affichent pas dans la sidebar
**Solution** : Vérifiez que les fichiers dans le dossier `pages/` ont le format correct : `01_Upload.py`, `02_EDA.py`, etc.

### Problème : ModuleNotFoundError
**Solution** : Assurez-vous que tous les modules Python (config.py, feature_engineering.py, etc.) sont dans le dossier racine, pas dans `pages/`.

### Problème : Erreur de chargement du modèle
**Solution** : Vérifiez que le dossier `models/` contient les trois fichiers `.pkl` nécessaires.

### Problème : Erreur lors de l'upload de fichier Excel
**Solution** : Installez openpyxl : `pip install openpyxl`

### Problème : Application lente
**Solution** : 
- Limitez la taille des données uploadées
- Utilisez un sous-échantillon pour les tests
- Vérifiez que vous utilisez Python 3.8+

---

## Logs

Les logs de l'application sont automatiquement sauvegardés dans le dossier `logs/` avec un fichier par jour au format : `app_YYYYMMDD.log`

Pour consulter les logs :
```bash
# Windows
type logs\app_20251028.log

# macOS/Linux
cat logs/app_20251028.log
```

---

## Contribution

Pour contribuer au projet :

1. Fork le repository
2. Créez une branche pour votre feature (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## Auteur

Abdoul-rachid Boinzemwendé FOFANA
rachk02@outlook.fr
Août 2025

---

## Licence

Génie Logiciel Pure Developer - Option Analyse de données

---

## Remerciements

Ce projet a été développé dans le cadre d'une soutenance académique en Data Analyse / Machine Learning.
