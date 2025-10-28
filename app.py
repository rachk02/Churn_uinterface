"""
Application Streamlit - Plateforme Prédiction Churn Client
Version Refonte Complète
"""

import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Plateforme Prédiction Churn",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# INITIALISATION SESSION STATE
# ============================================================
def init_session_state():
    """Initialise les variables de session"""
    session_vars = [
        'df_raw', 'df_parsed', 'df_encoded',
        'df_normalized', 'df_model', 'df_with_predictions',
        'model', 'scaler', 'features'
    ]

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

init_session_state()

# ============================================================
# PAGE D'ACCUEIL
# ============================================================
st.title("Plateforme de Prédiction de Churn Client")
st.markdown("""
### Projet de Soutenance : Prédiction de Churn Client
**Système d'analyse et de machine learning pour l'identification des clients à risque**
""")
# Banner d'information
st.info("""
**Bienvenue !** Cette application te permet de :
- Charger tes données CSV
- Explorer les données avec des analyses complètes (EDA)
- Prétraiter les données étape par étape (avec visibilité)
- Prédire le churn client
- Analyser les résultats dans un dashboard
- Exporter les rapports et données
""")

# Instructions d'utilisation
with st.expander("Comment utiliser cette application ?", expanded=True):
    st.markdown("""
    ### Parcours utilisateur
    
    **1. Chargement des données** (Page 1)
    - Upload ton fichier CSV
    - Validation automatique des données
    - Aperçu des données brutes
    
    **2. Analyse exploratoire - EDA** (Page 2)
    - Vue d'ensemble (statistiques)
    - Analyse univariée (distributions)
    - Analyse bivariée (churn vs features)
    - Matrice de corrélation
    - Distribution du churn
    
    **3. Prétraitement** (Page 3)
    - Étape 1: Parsing JSON → df_parsed
    - Étape 2: Encodage → df_encoded
    - Étape 3: Normalisation → df_normalized
    - Étape 4: Dataset final → df_model
    
    **4. Prédiction** (Page 4)
    - Lance les prédictions
    - Visualise les résultats
    - Identifie les clients à risque
    
    **5. Dashboard** (Page 5)
    - KPIs globaux
    - Analyses par segment
    - Graphiques interactifs
    
    **6. Export** (Page 6)
    - Exporte les DataFrames en CSV
    - Génère des rapports
    """)

# Barre latérale: Statut du pipeline
st.sidebar.markdown("---")
st.sidebar.markdown("### Statut du Pipeline")

status_data = {
    "Étape": [
        "1. Chargement",
        "2. Prétraitement",
        "3. Encodage",
        "4. Normalisation",
        "5. Modèle",
        "6. Prédiction"
    ],
    "Statut": [
        "✅" if st.session_state['df_raw'] is not None else "⬜",
        "✅" if st.session_state['df_parsed'] is not None else "⬜",
        "✅" if st.session_state['df_encoded'] is not None else "⬜",
        "✅" if st.session_state['df_normalized'] is not None else "⬜",
        "✅" if st.session_state['df_model'] is not None else "⬜",
        "✅" if st.session_state['df_with_predictions'] is not None else "⬜"
    ]
}

st.sidebar.dataframe(
    pd.DataFrame(status_data),
    hide_index=True,
    use_container_width=True
)

# Informations sur les données en mémoire
st.sidebar.markdown("---")
st.sidebar.markdown("### Données en mémoire")

if st.session_state['df_raw'] is not None:
    st.sidebar.success(f"Données brutes: {st.session_state['df_raw'].shape[0]:,} lignes")
else:
    st.sidebar.warning("Aucune donnée chargée")

if st.session_state['df_model'] is not None:
    st.sidebar.success(f"Dataset prêt: {st.session_state['df_model'].shape[1]} features")

if st.session_state['df_with_predictions'] is not None:
    st.sidebar.success(f"Prédictions disponibles")

# Section: Prochaines étapes
st.markdown("---")
st.markdown("### Prochaines étapes")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state['df_raw'] is None:
        st.warning("Commencez par **Charger** dans la sidebar")
    else:
        st.success("Données chargées")

with col2:
    if st.session_state['df_raw'] is not None and st.session_state['df_model'] is None:
        st.warning("Allez dans **Preprocessing**")
    elif st.session_state['df_model'] is not None:
        st.success("Preprocessing terminé")

with col3:
    if st.session_state['df_model'] is not None and st.session_state['df_with_predictions'] is None:
        st.warning("Allez dans **Prédiction**")
    elif st.session_state['df_with_predictions'] is not None:
        st.success("Prédictions faites")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Application développée pour la prédiction de churn client</p>
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)