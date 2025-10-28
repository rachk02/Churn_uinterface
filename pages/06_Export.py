import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Export", layout="wide")

st.title("Export et Rapports")
st.markdown("### Télécharge tes données et rapports")

st.markdown("---")

# Section 1: Export des DataFrames
st.markdown("### Export des DataFrames")

st.info("""
Télécharge les DataFrames intermédiaires créés pendant le preprocessing.
Utile pour analyser les transformations ou déboguer.
""")

dataframes = {
    'df_raw': ('Données Brutes', 'donnees_brutes.csv'),
    'df_parsed': ('Après Parsing', 'apres_parsing.csv'),
    'df_encoded': ('Après Encodage', 'apres_encodage.csv'),
    'df_normalized': ('Après Normalisation', 'apres_normalisation.csv'),
    'df_model': ('Dataset Final', 'dataset_final.csv'),
    'df_with_predictions': ('Avec Prédictions', 'predictions_completes.csv')
}

col1, col2 = st.columns(2)

for i, (df_name, (label, filename)) in enumerate(dataframes.items()):
    if st.session_state[df_name] is not None:
        df = st.session_state[df_name]
        
        # Alterner entre colonnes
        with col1 if i % 2 == 0 else col2:
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label=f"{label}",
                data=csv,
                file_name=f"{filename}",
                mime="text/csv",
                use_container_width=True,
                help=f"Télécharger {label} ({df.shape[0]} lignes × {df.shape[1]} colonnes)"
            )

st.markdown("---")

# Section 2: Rapports
st.markdown("### Rapports de Synthèse")

if st.session_state['df_with_predictions'] is not None:
    df = st.session_state['df_with_predictions']
    
    # Génération du rapport texte
    st.markdown("#### Rapport Textuel")
    
    report = f"""
# Rapport de Prédiction de Churn
Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}

## Statistiques Globales

- **Total clients**: {len(df):,}
- **Churn prédit**: {(df['Prediction'] == 1).sum():,} clients ({(df['Prediction'] == 1).mean()*100:.1f}%)
- **Probabilité moyenne**: {df['Probability_Churn'].mean():.3f}

## Distribution des Risques

- **Risque Faible**: {(df['Risk_Level'] == 'Faible').sum():,} clients
- **Risque Moyen**: {(df['Risk_Level'] == 'Moyen').sum():,} clients
- **Risque Élevé**: {(df['Risk_Level'] == 'Élevé').sum():,} clients

## Top 5 Clients à Risque

"""
    
    # Ajouter le top 5
    high_risk = df[df['Prediction'] == 1].nlargest(5, 'Probability_Churn')
    
    for i, (idx, row) in enumerate(high_risk.iterrows(), 1):
        customer_id = row.get('CustomerID', 'N/A')
        prob = row['Probability_Churn']
        report += f"{i}. Client {customer_id} - Probabilité: {prob:.2%}\n"
    
    report += "\n"
    
    # Afficher le rapport
    st.text_area("Rapport", report, height=400)
    
    # Bouton de téléchargement
    st.download_button(
        label="Télécharger le Rapport (TXT)",
        data=report,
        file_name=f"rapport_churn_{datetime.now():%Y%m%d_%H%M%S}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
else:
    st.warning("Aucune prédiction disponible. Allez à la page de **Prédiction**")

st.markdown("---")

# Section 3: Export des résultats par catégorie
st.markdown("### Export par Catégorie de Risque")

if st.session_state['df_with_predictions'] is not None:
    df = st.session_state['df_with_predictions']
    
    col1, col2, col3 = st.columns(3)
    
    # Risque Faible
    with col1:
        df_low = df[df['Risk_Level'] == 'Faible']
        if len(df_low) > 0:
            csv = df_low.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Risque Faible ({len(df_low)})",
                data=csv,
                file_name="clients_risque_faible.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Risque Moyen
    with col2:
        df_medium = df[df['Risk_Level'] == 'Moyen']
        if len(df_medium) > 0:
            csv = df_medium.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Risque Moyen ({len(df_medium)})",
                data=csv,
                file_name="clients_risque_moyen.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Risque Élevé
    with col3:
        df_high = df[df['Risk_Level'] == 'Élevé']
        if len(df_high) > 0:
            csv = df_high.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Risque Élevé ({len(df_high)})",
                data=csv,
                file_name="clients_risque_eleve.csv",
                mime="text/csv",
                use_container_width=True
            )

st.markdown("---")

# Section 4: Statistiques d'export
st.markdown("### Statistiques d'Export")

if st.session_state['df_with_predictions'] is not None:
    df = st.session_state['df_with_predictions']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("DataFrames disponibles", 
                 sum(1 for k, v in st.session_state.items() if k.startswith('df_') and v is not None))
    
    with col2:
        st.metric("Total lignes exportables", f"{len(df):,}")
    
    with col3:
        st.metric("Total colonnes", df.shape[1])
    
    with col4:
        size_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Taille mémoire", f"{size_mb:.2f} MB")

st.markdown("---")
st.success("Tous les exports sont disponibles ci-dessus !")
