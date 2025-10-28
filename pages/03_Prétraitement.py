import streamlit as st
from feature_engineering import extract_all_features
from preprocessing import encode_categorical_features, normalize_features, select_model_features
from model_utils import load_scaler
from config import MODEL_FEATURES

st.set_page_config(page_title="Pretraitement", layout="wide")

st.title("Prétraitement des Données")
st.markdown("### Transformations étape par étape avec DataFrames intermédiaires")

# Vérifier les données
if st.session_state['df_raw'] is None:
    st.warning("Aucune donnée chargée. Allez d'abord à la page **Chargement**")
    st.stop()

# Onglets pour chaque étape
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Parsing JSON",
    "2️⃣ Encodage",
    "3️⃣ Normalisation",
    "4️⃣ Dataset Final"
])

# TAB 1: Parsing
with tab1:
    st.subheader("Étape 1: Parsing des champs JSON")

    st.info("""
    Cette étape parse les 8 champs JSON et crée 12+ nouvelles variables:
    - Total_Late_Payments, Nb_ServiceInteractions
    - AvgLoginsPerMonth, Feedback_Score
    - EmailOpenRate, BounceRate, etc.
    """)

    if st.button("Lancer le Parsing", type="primary", key="parse"):
        with st.spinner("Parsing en cours..."):
            df_raw = st.session_state['df_raw']
            df_parsed = extract_all_features(df_raw)
            st.session_state['df_parsed'] = df_parsed

            st.success(f"Parsing terminé: {df_parsed.shape[1]} colonnes créées")

    # Afficher résultat
    if st.session_state['df_parsed'] is not None:
        df_parsed = st.session_state['df_parsed']

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avant", f"{st.session_state['df_raw'].shape[1]} colonnes")
        with col2:
            st.metric("Après", f"{df_parsed.shape[1]} colonnes")

        new_cols = [c for c in df_parsed.columns if c not in st.session_state['df_raw'].columns]

        with st.expander(f"{len(new_cols)} nouvelles colonnes créées"):
            for col in new_cols:
                st.write(f"• {col}")

        st.dataframe(df_parsed.head(), use_container_width=True)

# TAB 2: Encodage
with tab2:
    st.subheader("Étape 2: Encodage One-Hot")

    if st.session_state['df_parsed'] is None:
        st.warning("Fais d'abord l'étape 1 (Parsing)")
        st.stop()

    st.info("Encode Gender (2 colonnes) et Segment (3 colonnes)")

    if st.button("Lancer l'Encodage", type="primary", key="encode"):
        with st.spinner("Encodage en cours..."):
            df_parsed = st.session_state['df_parsed']
            df_encoded = encode_categorical_features(df_parsed)
            st.session_state['df_encoded'] = df_encoded

            st.success("Encodage terminé")

    if st.session_state['df_encoded'] is not None:
        df_encoded = st.session_state['df_encoded']

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avant", f"{st.session_state['df_parsed'].shape[1]} colonnes")
        with col2:
            st.metric("Après", f"{df_encoded.shape[1]} colonnes")

        encoded_cols = [c for c in df_encoded.columns if c.startswith('Gender_') or c.startswith('Segment_')]

        with st.expander(f"{len(encoded_cols)} colonnes encodées"):
            for col in encoded_cols:
                st.write(f"• {col}: {df_encoded[col].sum()} clients")

# TAB 3: Normalisation
with tab3:
    st.subheader("Étape 3: Normalisation StandardScaler")

    if st.session_state['df_encoded'] is None:
        st.warning("Faites d'abord l'étape 2 (Encodage)")
        st.stop()

    st.info("Normalise 14 features numériques (mean ≈ 0, std ≈ 1)")

    if st.button("Lancer la Normalisation", type="primary", key="normalize"):
        with st.spinner("Normalisation en cours..."):
            df_encoded = st.session_state['df_encoded']
            scaler = load_scaler()

            df_normalized, _ = normalize_features(df_encoded, scaler, fit=False)
            st.session_state['df_normalized'] = df_normalized

            st.success("Normalisation terminée")

    if st.session_state['df_normalized'] is not None:
        st.success("Données normalisées disponibles")
        st.dataframe(st.session_state['df_normalized'].describe(), use_container_width=True)

# TAB 4: Dataset Final
with tab4:
    st.subheader("Étape 4: Création du Dataset Final")

    if st.session_state['df_normalized'] is None:
        st.warning("Faites d'abord l'étape 3 (Normalisation)")
        st.stop()

    st.info(f"Sélection des {len(MODEL_FEATURES)} features finales pour le modèle")

    if st.button("Créer le Dataset Final", type="primary", key="final"):
        with st.spinner("Création du dataset..."):
            df_normalized = st.session_state['df_normalized']
            X = select_model_features(df_normalized, MODEL_FEATURES)
            st.session_state['df_model'] = X

            st.success(f"Dataset final créé: {X.shape[1]} features")
            st.balloons()

    if st.session_state['df_model'] is not None:
        X = st.session_state['df_model']

        st.metric("Features finales", X.shape[1])

        with st.expander("Liste des features"):
            for i, col in enumerate(X.columns, 1):
                st.write(f"{i}. {col}")

        st.dataframe(X.head(), use_container_width=True)

        st.success("Prétraitement terminé ! Allez dans **Prédiction**")