import streamlit as st
from model_utils import load_all_artifacts, predict_churn, add_predictions_to_dataframe
from model_utils import get_prediction_summary, get_high_risk_clients
import plotly.express as px

st.set_page_config(page_title="Prédiction", layout="wide")

st.title("Prédiction de Churn")
st.markdown("### Prédis quels clients vont partir")

# Vérifier que le preprocessing est fait
if st.session_state['df_raw'] is None:
    st.warning("Aucune donnée. Allez à la page de **Chargement**")
    st.stop()

if st.session_state['df_model'] is None:
    st.warning("Prétraitement non fait. Allez dans **Prétraitement**")
    st.stop()

# Charger le modèle si pas déjà chargé
if st.session_state['model'] is None:
    with st.spinner("Chargement du modèle..."):
        model, scaler, features = load_all_artifacts()
        
        if model is not None:
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['features'] = features
            st.success("Modèle chargé")
        else:
            st.error("Erreur de chargement du modèle")
            st.stop()

# Bouton de prédiction
st.markdown("### Lancer les prédictions")

if st.button("Prédire le Churn", type="primary", use_container_width=True):
    with st.spinner("Prédiction en cours..."):
        df_raw = st.session_state['df_raw']
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        
        # Prédire
        predictions, probabilities, X = predict_churn(df_raw, model, scaler)
        
        if predictions is not None:
            # Ajouter au DataFrame
            df_results = add_predictions_to_dataframe(df_raw, predictions, probabilities)
            st.session_state['df_with_predictions'] = df_results
            
            # Résumé
            summary = get_prediction_summary(predictions, probabilities)
            
            st.success("Prédictions effectuées !")
            st.balloons()
            
            # Afficher les métriques
            st.markdown("### Résultats Globaux")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Clients", f"{summary['total_clients']:,}")
            
            with col2:
                st.metric(
                    "Churn Prédit", 
                    f"{summary['churn_predicted']:,}",
                    f"{summary['churn_percentage']}%"
                )
            
            with col3:
                st.metric("Risque Élevé", f"{summary['high_risk_count']:,}")
            
            with col4:
                st.metric("Risque Moyen", f"{summary['medium_risk_count']:,}")
            
            # Distribution des risques
            st.markdown("### Distribution des Niveaux de Risque")
            
            risk_data = {
                'Niveau': ['Faible', 'Moyen', 'Élevé'],
                'Nombre': [
                    summary['low_risk_count'],
                    summary['medium_risk_count'],
                    summary['high_risk_count']
                ]
            }
            
            fig = px.pie(
                risk_data, 
                values='Nombre', 
                names='Niveau',
                title='Répartition des niveaux de risque',
                color='Niveau',
                color_discrete_map={
                    'Faible': 'green',
                    'Moyen': 'orange',
                    'Élevé': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Erreur lors de la prédiction")

# Afficher les résultats si disponibles
if st.session_state['df_with_predictions'] is not None:
    df_results = st.session_state['df_with_predictions']
    
    st.markdown("---")
    st.markdown("### Top 10 Clients à Risque Élevé")
    
    high_risk = get_high_risk_clients(df_results, top_n=10)
    
    if len(high_risk) > 0:
        # Colonnes à afficher
        display_cols = ['CustomerID']
        
        if 'Name' in high_risk.columns:
            display_cols.append('Name')
        if 'Email' in high_risk.columns:
            display_cols.append('Email')
        
        display_cols.extend(['Probability_Churn', 'Risk_Level'])
        
        # Filtrer les colonnes existantes
        display_cols = [c for c in display_cols if c in high_risk.columns]
        
        st.dataframe(
            high_risk[display_cols],
            use_container_width=True,
            height=400
        )
        
        # Graphique
        st.markdown("### Probabilités de Churn - Top 10")
        
        fig = px.bar(
            high_risk.head(10),
            x='CustomerID',
            y='Probability_Churn',
            title='Top 10 clients à risque',
            color='Probability_Churn',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tous les résultats
    st.markdown("---")
    st.markdown("### Tous les Résultats")
    
    # Filtres
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.multiselect(
            "Filtrer par niveau de risque",
            ['Faible', 'Moyen', 'Élevé'],
            default=['Élevé']
        )
    
    with col2:
        show_churn_only = st.checkbox("Afficher uniquement les churners prédits", value=True)
    
    # Appliquer les filtres
    df_filtered = df_results.copy()
    
    if risk_filter:
        df_filtered = df_filtered[df_filtered['Risk_Level'].isin(risk_filter)]
    
    if show_churn_only:
        df_filtered = df_filtered[df_filtered['Prediction'] == 1]
    
    st.write(f"**{len(df_filtered)}** clients affichés")
    
    st.dataframe(df_filtered, use_container_width=True, height=400)
    
    # Export
    st.markdown("---")
    st.markdown("### Exporter les Résultats")
    
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les prédictions (CSV)",
        data=csv,
        file_name="predictions_churn.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("Cliquez sur le bouton ci-dessus pour lancer les prédictions")
