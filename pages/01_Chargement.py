import streamlit as st
import pandas as pd
from logger import logger
from validation import validate_raw_data

st.set_page_config(page_title="Chargement", layout="wide")

st.title("Chargement des Données")
st.markdown("---")

# Instructions
st.info("""
**Instructions:**
1. Choisis ton fichier CSV ou XLSX contenant les données clients
2. Le fichier sera validé automatiquement
3. Vous verrez un aperçu des données
4. Clique sur "Sauvegarder" pour continuer vers l'EDA
""")

# Upload fichier (CSV ou XLSX)
uploaded_file = st.file_uploader(
    "Choisissez un fichier CSV ou Excel",
    type=['csv', 'xlsx', 'xls'],
    help="Fichier CSV ou Excel avec les données clients"
)

if uploaded_file is not None:
    try:
        # Charger les données selon le type de fichier
        with st.spinner("Chargement du fichier..."):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
                logger.info(f"Fichier CSV chargé: {uploaded_file.name}")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
                logger.info(f"Fichier Excel chargé: {uploaded_file.name}")
            else:
                st.error("Format de fichier non supporté")
                st.stop()
            
            logger.info(f"Données chargées: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        st.success(f"Fichier chargé avec succès: **{df.shape[0]:,}** lignes × **{df.shape[1]}** colonnes")
        
        # Validation
        st.markdown("### Validation des données")
        
        with st.spinner("Validation en cours..."):
            validation_result = validate_raw_data(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if validation_result['valid']:
                st.success("**Validation réussie**")
            else:
                st.error("**Validation échouée**")
        
        with col2:
            if validation_result['warnings']:
                st.warning(f"⚠**{len(validation_result['warnings'])} avertissements**")
        
        # Afficher les erreurs
        if validation_result['errors']:
            with st.expander("Erreurs détectées", expanded=True):
                for error in validation_result['errors']:
                    st.error(f"• {error}")
        
        # Afficher les warnings
        if validation_result['warnings']:
            with st.expander("Avertissements"):
                for warning in validation_result['warnings']:
                    st.warning(f"• {warning}")
        
        # Statistiques du fichier
        st.markdown("### Statistiques du fichier")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lignes", f"{df.shape[0]:,}")
        with col2:
            st.metric("Colonnes", df.shape[1])
        with col3:
            null_count = df.isnull().sum().sum()
            st.metric("Valeurs manquantes", f"{null_count:,}")
        with col4:
            size_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Taille", f"{size_mb:.2f} MB")
        
        # Types de colonnes
        st.markdown("### Types de colonnes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Colonnes numériques:**")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            st.write(f"{len(numeric_cols)} colonnes")
            if len(numeric_cols) > 0:
                with st.expander("Voir la liste"):
                    for col in numeric_cols:
                        st.write(f"• {col}")
        
        with col2:
            st.write("**Colonnes textuelles:**")
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            st.write(f"{len(text_cols)} colonnes")
            if len(text_cols) > 0:
                with st.expander("Voir la liste"):
                    for col in text_cols:
                        st.write(f"• {col}")
        
        # Aperçu des données
        st.markdown("### Aperçu des données")
        
        # Sélecteur de nombre de lignes
        n_rows = st.slider("Nombre de lignes à afficher", 5, 100, 20)
        
        # Afficher le DataFrame
        st.dataframe(df.head(n_rows), use_container_width=True, height=400)
        
        # Statistiques descriptives
        with st.expander("Statistiques descriptives"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Informations sur les valeurs manquantes
        with st.expander("Détails des valeurs manquantes"):
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            
            if len(missing) > 0:
                missing_df = pd.DataFrame({
                    'Colonne': missing.index,
                    'Valeurs manquantes': missing.values,
                    'Pourcentage': (missing.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("Aucune valeur manquante dans le dataset")
        
        # Bouton pour sauvegarder
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("Sauvegarder et continuer vers l'EDA",
                        type="primary", 
                        use_container_width=True):
                st.session_state['df_raw'] = df
                logger.info("Données sauvegardées dans session_state")
                st.success("Données sauvegardées avec succès !")
                st.balloons()
                st.info("Vous pouvez maintenant aller dans **EDA** (sidebar)")
    
    except Exception as e:
        st.error(f"**Erreur lors du chargement du fichier**")
        st.error(f"Détails: {str(e)}")
        logger.error(f"Erreur chargement fichier: {e}")
        
        with st.expander("🔧 Aide au débogage"):
            st.write("""
            **Causes possibles:**
            - Fichier corrompu
            - Format CSV/Excel invalide
            - Encodage incorrect (pour CSV)
            - Feuille Excel vide
            
            **Solutions:**
            - Vérifie que le fichier n'est pas corrompu
            - Pour CSV: essaie d'ouvrir avec un éditeur de texte
            - Pour Excel: vérifie que les données sont dans la première feuille
            - Pour CSV: vérifie l'encodage (UTF-8 recommandé)
            """)
            
            # Afficher le traceback complet
            import traceback
            st.code(traceback.format_exc())

else:
    # Message si aucun fichier uploadé
    st.warning("**Chargez un fichier CSV ou Excel pour commencer**")
    
    st.markdown("### Format attendu")
    st.write("""
    Le fichier CSV ou Excel doit contenir au minimum ces colonnes:
    - `CustomerID` : Identifiant unique du client
    - `Age`, `Gender`, `Segment`, `NPS` : Informations client
    - `PaymentHistory`, `ServiceInteractions`, `EngagementMetrics` : Données JSON
    - `Feedback`, `WebsiteUsage`, `MarketingCommunication` : Données JSON
    - `PurchaseHistory`, `ClickstreamData` : Données JSON
    - `ChurnLabel` : Variable cible (0 ou 1) - optionnel pour la prédiction
    """)
    
    st.markdown("### Conseils")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Pour les fichiers CSV:**
        - Utilise l'encodage UTF-8
        - Séparateur: virgule (,)
        - Pas d'espaces dans les noms de colonnes
        """)
    
    with col2:
        st.info("""
        **Pour les fichiers Excel:**
        - Format: .xlsx ou .xls
        - Données dans la première feuille
        - Pas de cellules fusionnées
        """)

# Afficher l'état actuel
st.markdown("---")
st.markdown("### État actuel")

if st.session_state['df_raw'] is not None:
    df_info = st.session_state['df_raw']
    
    st.success(f"""
    **Données en mémoire**
    - Lignes: {df_info.shape[0]:,}
    - Colonnes: {df_info.shape[1]}
    - Taille: {df_info.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    """)
    
    # Bouton pour supprimer les données
    if st.button("Supprimer les données en mémoire", type="secondary"):
        st.session_state['df_raw'] = None
        st.session_state['df_parsed'] = None
        st.session_state['df_encoded'] = None
        st.session_state['df_normalized'] = None
        st.session_state['df_model'] = None
        st.session_state['df_with_predictions'] = None
        st.warning("Données supprimées. Recharge la page.")
        st.rerun()
else:
    st.info("Aucune donnée en mémoire pour le moment")
